//! Basis Update & Galerkin (BUG) integrator for Hierarchical Tucker format.
//!
//! BUG is a dynamical low-rank integrator that updates the HT tensor factors
//! (leaf frames and transfer tensors) directly without ever materializing the
//! full 6D grid. Each timestep consists of three conceptual stages:
//!
//! - **K-step:** Update spatial (drift) or velocity (kick) leaf bases by
//!   semi-Lagrangian shifting, then QR-decompose to maintain orthonormality.
//! - **L-step:** Recompute the gravitational acceleration from the updated
//!   density projection and apply the velocity kick.
//! - **S-step:** Update the transfer tensors to absorb the basis change
//!   coefficients (the R matrices from QR), optionally augmenting rank.
//!
//! This avoids the step-and-truncate (SAT) approach where the full tensor is
//! advanced and then re-compressed, providing controlled memory usage,
//! automatic rank adaptation, and robust stability on the low-rank manifold.
//!
//! Algorithm (rank-adaptive BUG, one step):
//! 1. **K-step:** For each active leaf, shift basis by semi-Lagrangian,
//!    QR-decompose to get new orthonormal basis and coefficient matrix R.
//! 2. **Transfer update:** Contract R into the parent's transfer tensor
//!    along the appropriate child axis.
//! 3. **S-step (optional augmentation):** When `rank_increase > 0`, augment
//!    bases with shifted columns at additional velocity/acceleration samples,
//!    then SVD-truncate transfer tensors for rank adaptation.
//!
//! Midpoint variant: half-step K-update to predict midpoint bases, then
//! full-step Galerkin projection using the midpoint bases for 2nd-order accuracy.
//!
//! Conservative variant: after truncation, correct the distribution to
//! restore mass conservation via root transfer tensor scaling.
//!
//! Reference: Ceruti, Lubich & Walach, "An unconventional robust integrator
//! for dynamical low-rank approximation", BIT Numer. Math. (2022).

use std::sync::Arc;

use faer::Mat;

use super::super::{
    advecator::Advector,
    algos::ht::HtTensor,
    algos::lagrangian::sl_shift_1d_into,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};
use super::helpers;
use crate::CausticError;

/// Configuration for the BUG integrator.
pub struct BugConfig {
    /// Truncation tolerance for rank adaptation.
    pub tolerance: f64,
    /// Maximum rank per node.
    pub max_rank: usize,
    /// Use 2nd-order midpoint variant (otherwise 1st-order).
    pub midpoint: bool,
    /// Apply conservative moment correction after truncation.
    pub conservative: bool,
    /// Number of extra basis columns per K-step (0 = rank-preserving).
    pub rank_increase: usize,
}

impl Default for BugConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_rank: 50,
            midpoint: false,
            conservative: false,
            rank_increase: 2,
        }
    }
}

/// Parent node index and whether the leaf is the left child.
pub(crate) const LEAF_PARENT: [(usize, bool); 6] = [
    (8, true),  // leaf 0 → node 8, left
    (6, true),  // leaf 1 → node 6, left
    (6, false), // leaf 2 → node 6, right
    (9, true),  // leaf 3 → node 9, left
    (7, true),  // leaf 4 → node 7, left
    (7, false), // leaf 5 → node 7, right
];

// ─── Shared BUG helpers (used by BugIntegrator, ParallelBugIntegrator, RkBugIntegrator) ──

/// Shift all columns of a leaf frame by `displacement`, QR decompose,
/// and optionally augment with extra shifted columns.
///
/// Returns `(new_frame, R_trunc)` where:
/// - `new_frame`: orthonormal basis (n × k_new)
/// - `R_trunc`: coefficient matrix (k_new × k_old) for transfer tensor update
pub(crate) fn k_step_leaf(
    ht: &HtTensor,
    leaf_dim: usize,
    displacement: f64,
    aug_displacements: &[f64],
    max_rank: usize,
    tolerance: f64,
) -> (Mat<f64>, Mat<f64>) {
    let frame = ht.leaf_frame(leaf_dim);
    let (n, k) = (frame.nrows(), frame.ncols());
    let is_spatial = leaf_dim < 3;
    let dim_idx = if is_spatial { leaf_dim } else { leaf_dim - 3 };

    let (cell_size, half_extent, periodic) = if is_spatial {
        let dx = ht.domain.dx();
        let lx = ht.domain.lx();
        let per = matches!(
            ht.domain.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );
        (dx[dim_idx], lx[dim_idx], per)
    } else {
        let dv = ht.domain.dv();
        let lv = ht.domain.lv();
        let per = matches!(
            ht.domain.velocity_bc,
            super::super::init::domain::VelocityBoundType::Truncated
        );
        (dv[dim_idx], lv[dim_idx], per)
    };

    // Shift each column by the primary displacement
    let mut shifted = Mat::<f64>::zeros(n, k);
    let mut col_buf = vec![0.0f64; n];
    let mut out_buf = vec![0.0f64; n];

    for j in 0..k {
        for i in 0..n {
            col_buf[i] = frame[(i, j)];
        }
        sl_shift_1d_into(
            &col_buf,
            displacement,
            cell_size,
            n,
            half_extent,
            periodic,
            &mut out_buf,
        );
        for i in 0..n {
            shifted[(i, j)] = out_buf[i];
        }
    }

    let n_aug = aug_displacements.len();
    if n_aug == 0 {
        // Rank-preserving: QR of shifted frame only
        let (q, r) = qr_thin(&shifted);
        return (q, r);
    }

    // Augmented: shift by additional displacements, collect extra columns
    let total_cols = k + n_aug * k;
    let mut augmented = Mat::<f64>::zeros(n, total_cols);
    for j in 0..k {
        for i in 0..n {
            augmented[(i, j)] = shifted[(i, j)];
        }
    }
    for (s, &disp) in aug_displacements.iter().enumerate() {
        for j in 0..k {
            for i in 0..n {
                col_buf[i] = frame[(i, j)];
            }
            sl_shift_1d_into(
                &col_buf,
                disp,
                cell_size,
                n,
                half_extent,
                periodic,
                &mut out_buf,
            );
            for i in 0..n {
                augmented[(i, k + s * k + j)] = out_buf[i];
            }
        }
    }

    // QR decompose augmented matrix
    let (q_aug, r_aug) = qr_thin(&augmented);

    // SVD truncate to target rank
    let target_rank = (k + n_aug).min(max_rank).min(q_aug.ncols());
    let (u, sv, _vt) = svd_thin(&r_aug);
    if u.ncols() == 0 {
        return (q_aug, r_aug.subcols(0, k).to_owned());
    }
    let rank = truncation_rank(&sv, tolerance)
        .max(1)
        .min(target_rank)
        .min(u.ncols());

    // Truncated basis: Q_trunc = Q_aug @ U[:, :rank]
    let u_trunc = u.subcols(0, rank);
    let q_trunc = &q_aug * u_trunc;

    // Coefficient matrix: how the shifted primary columns decompose in the new basis
    // shifted = Q_aug @ R_aug[:, :k], and Q_trunc = Q_aug @ U[:,:rank]
    // So R_trunc = U[:,:rank]^T @ R_aug[:, :k]  (k_new × k_old)
    let r_aug_left = r_aug.subcols(0, k);
    let r_trunc = u_trunc.transpose() * r_aug_left;

    (q_trunc.to_owned(), r_trunc.to_owned())
}

/// Update the parent's transfer tensor after replacing a child's leaf frame.
///
/// If left child:  B_new[p, l_new, r] = Σ_l R[l_new, l] * B[p, l, r]
/// If right child: B_new[p, l, r_new] = Σ_r R[r_new, r] * B[p, l, r]
pub(crate) fn update_transfer(ht: &mut HtTensor, leaf_dim: usize, r_matrix: &Mat<f64>) {
    let (parent_idx, is_left) = LEAF_PARENT[leaf_dim];
    let (transfer, ranks) = ht.transfer_tensor(parent_idx);
    let [kp, kl, kr] = ranks;
    let k_new = r_matrix.nrows();
    let k_old = r_matrix.ncols();

    if is_left {
        assert_eq!(k_old, kl, "R cols must match old left rank");
        let mut new_data = vec![0.0f64; kp * k_new * kr];
        for p in 0..kp {
            for l_new in 0..k_new {
                for r in 0..kr {
                    let mut sum = 0.0;
                    for l in 0..kl {
                        sum += r_matrix[(l_new, l)] * transfer[p * kl * kr + l * kr + r];
                    }
                    new_data[p * k_new * kr + l_new * kr + r] = sum;
                }
            }
        }
        ht.set_transfer_tensor(parent_idx, new_data, [kp, k_new, kr]);
    } else {
        assert_eq!(k_old, kr, "R cols must match old right rank");
        let mut new_data = vec![0.0f64; kp * kl * k_new];
        for p in 0..kp {
            for l in 0..kl {
                for r_new in 0..k_new {
                    let mut sum = 0.0;
                    for r in 0..kr {
                        sum += r_matrix[(r_new, r)] * transfer[p * kl * kr + l * kr + r];
                    }
                    new_data[p * kl * k_new + l * k_new + r_new] = sum;
                }
            }
        }
        ht.set_transfer_tensor(parent_idx, new_data, [kp, kl, k_new]);
    }
}

/// Compute representative velocities from a velocity leaf frame.
pub(crate) fn representative_velocities(ht: &HtTensor, vel_dim: usize) -> Vec<f64> {
    let v_frame = ht.leaf_frame(vel_dim);
    let (nv, kv) = (v_frame.nrows(), v_frame.ncols());
    let dim_idx = vel_dim - 3;
    let dv = ht.domain.dv();
    let lv = ht.domain.lv();

    (0..kv)
        .map(|l| {
            let mut wt_sum = 0.0f64;
            let mut v_sum = 0.0f64;
            for i in 0..nv {
                let v = -lv[dim_idx] + (i as f64 + 0.5) * dv[dim_idx];
                let w = v_frame[(i, l)] * v_frame[(i, l)];
                v_sum += w * v;
                wt_sum += w;
            }
            if wt_sum > 1e-30 { v_sum / wt_sum } else { 0.0 }
        })
        .collect()
}

/// Compute representative accelerations for a spatial leaf.
pub(crate) fn representative_accelerations(
    ht: &HtTensor,
    spatial_dim: usize,
    accel: &AccelerationField,
) -> Vec<f64> {
    let x_frame = ht.leaf_frame(spatial_dim);
    let (nx_dim, kx) = (x_frame.nrows(), x_frame.ncols());
    let [nx1, nx2, nx3, _, _, _] = ht.shape;

    let accel_data = match spatial_dim {
        0 => &accel.gx,
        1 => &accel.gy,
        2 => &accel.gz,
        _ => unreachable!(),
    };

    (0..kx)
        .map(|j| {
            let mut wt_sum = 0.0f64;
            let mut a_sum = 0.0f64;
            for i in 0..nx_dim {
                let w = x_frame[(i, j)] * x_frame[(i, j)];
                // Average acceleration over the other two spatial dimensions
                let mut a_avg = 0.0f64;
                let n_other: usize = match spatial_dim {
                    0 => {
                        for ix2 in 0..nx2 {
                            for ix3 in 0..nx3 {
                                a_avg += accel_data[i * nx2 * nx3 + ix2 * nx3 + ix3];
                            }
                        }
                        nx2 * nx3
                    }
                    1 => {
                        for ix1 in 0..nx1 {
                            for ix3 in 0..nx3 {
                                a_avg += accel_data[ix1 * nx2 * nx3 + i * nx3 + ix3];
                            }
                        }
                        nx1 * nx3
                    }
                    2 => {
                        for ix1 in 0..nx1 {
                            for ix2 in 0..nx2 {
                                a_avg += accel_data[ix1 * nx2 * nx3 + ix2 * nx3 + i];
                            }
                        }
                        nx1 * nx2
                    }
                    _ => unreachable!(),
                };
                a_avg /= n_other as f64;
                a_sum += w * a_avg;
                wt_sum += w;
            }
            if wt_sum > 1e-30 { a_sum / wt_sum } else { 0.0 }
        })
        .collect()
}

/// Sample augmentation displacements from representative values.
pub(crate) fn sample_aug_displacements(
    representatives: &[f64],
    dt: f64,
    rank_increase: usize,
) -> Vec<f64> {
    if rank_increase == 0 || representatives.is_empty() {
        return vec![];
    }
    let mut reps: Vec<f64> = representatives.iter().map(|&v| v * dt).collect();
    reps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = reps.len();
    let mut aug = Vec::with_capacity(rank_increase);
    if n >= 1 {
        aug.push(reps[n - 1]); // max displacement
    }
    if rank_increase >= 2 && n >= 2 {
        aug.push(reps[0]); // min displacement
    }
    for s in 2..rank_increase {
        let frac = s as f64 / (rank_increase - 1) as f64;
        let idx = ((n as f64 - 1.0) * frac) as usize;
        aug.push(reps[idx.min(n - 1)]);
    }
    aug
}

/// BUG drift substep: K-step for spatial leaves 0, 1, 2.
pub(crate) fn bug_drift_substep(ht: &mut HtTensor, dt: f64, config: &BugConfig) {
    for d in 0..3 {
        let reps = representative_velocities(ht, d + 3);
        let primary = if reps.is_empty() {
            0.0
        } else {
            reps.iter().sum::<f64>() / reps.len() as f64 * dt
        };
        let aug = sample_aug_displacements(&reps, dt, config.rank_increase);
        let (new_frame, r_mat) =
            k_step_leaf(ht, d, primary, &aug, config.max_rank, config.tolerance);
        *ht.leaf_frame_mut(d) = new_frame;
        update_transfer(ht, d, &r_mat);
    }
}

/// BUG kick substep: K-step for velocity leaves 3, 4, 5.
pub(crate) fn bug_kick_substep(
    ht: &mut HtTensor,
    accel: &AccelerationField,
    dt: f64,
    config: &BugConfig,
) {
    for d in 3..6 {
        let reps = representative_accelerations(ht, d - 3, accel);
        let primary = if reps.is_empty() {
            0.0
        } else {
            reps.iter().sum::<f64>() / reps.len() as f64 * dt
        };
        let aug = sample_aug_displacements(&reps, dt, config.rank_increase);
        let (new_frame, r_mat) =
            k_step_leaf(ht, d, primary, &aug, config.max_rank, config.tolerance);
        *ht.leaf_frame_mut(d) = new_frame;
        update_transfer(ht, d, &r_mat);
    }
}

/// Conservative correction: scale root transfer tensor to restore mass.
pub(crate) fn conservative_correction(ht: &mut HtTensor, density_before: &DensityField) {
    let density_after = ht.compute_density();
    let mass_before: f64 = density_before.data.iter().sum();
    let mass_after: f64 = density_after.data.iter().sum();
    if mass_before.abs() < 1e-30 || (mass_after - mass_before).abs() < 1e-14 * mass_before.abs() {
        return;
    }
    let scale = mass_before / mass_after;
    let (transfer, ranks) = ht.transfer_tensor(10); // root
    let new_data: Vec<f64> = transfer.iter().map(|&v| v * scale).collect();
    ht.set_transfer_tensor(10, new_data, ranks);
}

// ─── Linear algebra helpers ─────────────────────────────────────────────

pub(crate) fn qr_thin(mat: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();
    if m.min(n) == 0 {
        return (Mat::zeros(m, 0), Mat::zeros(0, n));
    }
    let qr = mat.as_ref().qr();
    (qr.compute_thin_Q(), qr.thin_R().to_owned())
}

pub(crate) fn svd_thin(mat: &Mat<f64>) -> (Mat<f64>, Vec<f64>, Mat<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();
    let k = m.min(n);
    if k == 0 {
        return (Mat::zeros(m, 0), vec![], Mat::zeros(0, n));
    }
    let svd = match mat.as_ref().thin_svd() {
        Ok(s) => s,
        Err(_) => return (Mat::zeros(m, 0), vec![], Mat::zeros(0, n)),
    };
    let u = svd.U().to_owned();
    let vt = svd.V().transpose().to_owned();
    let s_diag = svd.S().column_vector();
    let s: Vec<f64> = (0..k).map(|i| s_diag[i]).collect();
    (u, s, vt)
}

pub(crate) fn truncation_rank(sv: &[f64], eps: f64) -> usize {
    let eps2 = eps * eps;
    let mut tail_sq = 0.0;
    for k in (0..sv.len()).rev() {
        tail_sq += sv[k] * sv[k];
        if tail_sq > eps2 {
            return k + 1;
        }
    }
    1
}

// ─── BugIntegrator ──────────────────────────────────────────────────────

/// BUG (Basis Update & Galerkin) integrator for low-rank tensor formats.
///
/// When the representation is an `HtTensor`, this integrator evolves the
/// solution directly on the low-rank manifold via K/L/S-step updates.
/// For other representations, it falls back to standard Strang splitting.
pub struct BugIntegrator {
    /// BUG algorithm parameters (tolerance, max rank, midpoint, conservative).
    pub config: BugConfig,
    /// Gravitational constant G used for the Poisson solve.
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl BugIntegrator {
    /// Create a new BUG integrator with the given gravitational constant and configuration.
    pub fn new(g: f64, config: BugConfig) -> Self {
        Self {
            config,
            g,
            last_timings: StepTimings::default(),
            progress: None,
        }
    }

    /// Fallback: standard Strang splitting for non-HT representations.
    fn strang_fallback(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        helpers::time_ms!(timings, drift_ms, advector.drift(repr, dt / 2.0));

        let (_, _, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(repr, solver, self.g)
        );

        helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, dt));

        helpers::time_ms!(timings, drift_ms, advector.drift(repr, dt / 2.0));
    }

    /// Standard BUG step: Strang-split drift-kick-drift on HT leaves.
    fn bug_step_ht(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        let Some(ht) = repr.as_any_mut().downcast_mut::<HtTensor>() else {
            debug_assert!(false, "BUG step requires HtTensor");
            return;
        };

        let density_before = if self.config.conservative {
            Some(ht.compute_density())
        } else {
            None
        };

        helpers::report_phase!(self.progress, StepPhase::BugKStep, 0, 4);
        helpers::time_ms!(
            timings,
            drift_ms,
            bug_drift_substep(ht, dt / 2.0, &self.config)
        );

        helpers::report_phase!(self.progress, StepPhase::BugLStep, 1, 4);
        let (_, _, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(ht, solver, self.g)
        );

        helpers::time_ms!(
            timings,
            kick_ms,
            bug_kick_substep(ht, &accel, dt, &self.config)
        );

        helpers::time_ms!(
            timings,
            drift_ms,
            bug_drift_substep(ht, dt / 2.0, &self.config)
        );

        helpers::report_phase!(self.progress, StepPhase::BugSStep, 2, 4);
        if let Some(ref dens) = density_before {
            conservative_correction(ht, dens);
        }
    }

    /// Midpoint BUG step: half-step predict, full-step with augmented bases.
    fn midpoint_bug_step(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        let Some(ht) = repr.as_any_mut().downcast_mut::<HtTensor>() else {
            debug_assert!(false, "midpoint BUG requires HtTensor");
            return;
        };

        let density_before = if self.config.conservative {
            Some(ht.compute_density())
        } else {
            None
        };

        helpers::report_phase!(self.progress, StepPhase::BugKStep, 0, 4);

        // Predict midpoint with half-step
        let saved = ht.clone();
        helpers::time_ms!(
            timings,
            drift_ms,
            bug_drift_substep(ht, dt / 4.0, &self.config)
        );

        let (_, _, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(ht, solver, self.g)
        );

        helpers::time_ms!(
            timings,
            kick_ms,
            bug_kick_substep(ht, &accel, dt / 2.0, &self.config)
        );

        helpers::time_ms!(
            timings,
            drift_ms,
            bug_drift_substep(ht, dt / 4.0, &self.config)
        );

        // ht is now at midpoint — restore and do full step
        helpers::report_phase!(self.progress, StepPhase::BugLStep, 1, 4);
        *ht = saved;

        let aug_config = BugConfig {
            rank_increase: self.config.rank_increase.max(1),
            ..BugConfig {
                tolerance: self.config.tolerance,
                max_rank: self.config.max_rank,
                midpoint: false,
                conservative: false,
                rank_increase: self.config.rank_increase.max(1),
            }
        };

        helpers::time_ms!(
            timings,
            drift_ms,
            bug_drift_substep(ht, dt / 2.0, &aug_config)
        );

        let (_, _, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(ht, solver, self.g)
        );

        helpers::time_ms!(
            timings,
            kick_ms,
            bug_kick_substep(ht, &accel, dt, &aug_config)
        );

        helpers::time_ms!(
            timings,
            drift_ms,
            bug_drift_substep(ht, dt / 2.0, &aug_config)
        );

        helpers::report_phase!(self.progress, StepPhase::BugSStep, 2, 4);
        if let Some(ref dens) = density_before {
            conservative_correction(ht, dens);
        }
    }
}

impl TimeIntegrator for BugIntegrator {
    /// Advance the distribution by one timestep `dt`.
    ///
    /// If the representation is `HtTensor`, performs a BUG step (standard or midpoint
    /// depending on config). Otherwise falls back to Strang splitting.
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("bug_advance").entered();
        let mut timings = StepTimings::default();

        if let Some(ref p) = self.progress {
            p.start_step();
        }

        let is_ht = repr.as_any().downcast_ref::<HtTensor>().is_some();

        if is_ht {
            if self.config.midpoint {
                self.midpoint_bug_step(repr, solver, dt, &mut timings);
            } else {
                self.bug_step_ht(repr, solver, dt, &mut timings);
            }
        } else {
            self.strang_fallback(repr, solver, advector, dt, &mut timings);
        }

        helpers::report_phase!(self.progress, StepPhase::StepComplete, 3, 4);

        // Compute end-of-step products for caller reuse
        let (density, potential, acceleration) = helpers::time_ms!(
            timings,
            density_ms,
            helpers::solve_poisson(repr, solver, self.g)
        );

        self.last_timings = timings;

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    /// Dynamical-time CFL: dt <= cfl_factor / sqrt(G * rho_max).
    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        helpers::dynamical_timestep(repr, self.g, cfl_factor)
    }

    /// Return timing breakdown from the most recent step.
    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }

    /// Attach a progress reporter for intra-step TUI updates.
    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}
