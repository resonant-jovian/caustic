//! Step-level rank diagnostics for monitoring HT tensor rank amplification.
//!
//! `InstrumentedStrangSplitting` wraps `StrangSplitting` and records ranks
//! before/after each sub-step (drift, kick, Poisson). This lets us attribute
//! rank growth to specific algorithmic phases.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use super::super::{
    advecator::Advector,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
};
use crate::CausticError;

/// Per-step rank diagnostics, populated by `InstrumentedStrangSplitting`.
/// Fields are `None` when the representation is not `HtTensor`.
#[derive(Clone, Debug, Default)]
pub struct StepRankDiagnostics {
    /// Ranks of each HT node before the first drift half-step.
    pub pre_drift_ranks: Option<Vec<usize>>,
    /// Ranks after drift(Δt/2).
    pub post_drift_ranks: Option<Vec<usize>>,
    /// Ranks after the velocity kick (Poisson + advect_v).
    pub post_kick_ranks: Option<Vec<usize>>,
    /// Ranks after the second drift half-step (final state).
    pub post_final_ranks: Option<Vec<usize>>,
    /// Poisson rank amplification: max(post_kick_rank) / max(post_drift_rank).
    pub poisson_rank_amplification: Option<f64>,
    /// Advection rank amplification: max(post_drift_rank) / max(pre_drift_rank).
    pub advection_rank_amplification: Option<f64>,
    /// Exponential fit to recent rank history (last 10 steps).
    /// Positive = rank growing, > 0.5 = doubling every 2 steps.
    pub rank_growth_rate: Option<f64>,
    /// Per-node singular values from the most recent HSVD truncation.
    /// Outer Vec = nodes (11 for 6D HT), inner Vec = singular values.
    pub singular_value_spectrum: Option<Vec<Vec<f64>>>,
    /// Current max rank / max_rank budget (0.0 to 1.0).
    pub rank_budget_fraction: f64,
}

/// Extract per-node ranks from a `PhaseSpaceRepr` if it's an `HtTensor`.
fn extract_ranks(repr: &dyn PhaseSpaceRepr) -> Option<Vec<usize>> {
    use super::super::algos::ht::HtTensor;
    let any = repr.as_any();
    any.downcast_ref::<HtTensor>()
        .map(|ht| ht.nodes.iter().map(|n| n.rank()).collect())
}

/// Compute singular values of a matrix, returning an empty Vec on failure.
///
/// Falls back to computing column norms when `thin_svd` fails (which can
/// happen for very small or nearly rank-deficient matrices in faer 0.24).
fn singular_values(mat: &faer::Mat<f64>) -> Vec<f64> {
    let m = mat.nrows();
    let n = mat.ncols();
    let k = m.min(n);
    if k == 0 {
        return vec![];
    }
    match mat.as_ref().thin_svd() {
        Ok(svd) => {
            let s_diag = svd.S().column_vector();
            (0..k).map(|i| s_diag[i]).collect()
        }
        Err(_) => {
            // Fallback: compute singular values via column norms.
            // For a single-column matrix the sole singular value is the column norm.
            if n == 1 {
                let norm: f64 = (0..m)
                    .map(|i| mat[(i, 0)] * mat[(i, 0)])
                    .sum::<f64>()
                    .sqrt();
                if norm.is_finite() {
                    return vec![norm];
                }
                return vec![];
            }
            if m == 1 {
                let norm: f64 = (0..n)
                    .map(|j| mat[(0, j)] * mat[(0, j)])
                    .sum::<f64>()
                    .sqrt();
                if norm.is_finite() {
                    return vec![norm];
                }
                return vec![];
            }
            // General fallback: Frobenius norm as a conservative single-value estimate.
            let mut fro2 = 0.0;
            for i in 0..m {
                for j in 0..n {
                    fro2 += mat[(i, j)] * mat[(i, j)];
                }
            }
            let fro = fro2.sqrt();
            if fro.is_finite() { vec![fro] } else { vec![] }
        }
    }
}

/// Extract per-node singular values from an `HtTensor`.
///
/// For leaf nodes, computes the SVD of the basis frame U_μ and returns
/// the singular values. For interior nodes, reshapes the transfer tensor
/// B_t ∈ ℝ^{k_t × (k_left · k_right)} and computes its SVD.
///
/// Returns `None` if the representation is not `HtTensor`.
fn extract_singular_values(repr: &dyn PhaseSpaceRepr) -> Option<Vec<Vec<f64>>> {
    use super::super::algos::ht::{HtNode, HtTensor};
    let any = repr.as_any();
    any.downcast_ref::<HtTensor>().map(|ht| {
        ht.nodes
            .iter()
            .map(|node| match node {
                HtNode::Leaf { frame, .. } => singular_values(frame),
                HtNode::Interior {
                    transfer, ranks, ..
                } => {
                    let [kt, kl, kr] = *ranks;
                    let cols = kl * kr;
                    let mat = faer::Mat::from_fn(kt, cols, |i, j| transfer[i * cols + j]);
                    singular_values(&mat)
                }
            })
            .collect()
    })
}

/// Extract the max_rank budget from an `HtTensor`, returning `None` if
/// the representation is not `HtTensor`.
fn extract_max_rank_budget(repr: &dyn PhaseSpaceRepr) -> Option<usize> {
    use super::super::algos::ht::HtTensor;
    let any = repr.as_any();
    any.downcast_ref::<HtTensor>().map(|ht| ht.max_rank)
}

fn max_rank(ranks: &[usize]) -> f64 {
    ranks.iter().copied().max().unwrap_or(1).max(1) as f64
}

/// Number of entries retained in the rank history ring buffer.
const RANK_HISTORY_LEN: usize = 10;

/// Compute exponential growth rate from a ring buffer of max-rank values.
///
/// Fits log(rank) = a + b * step via ordinary least squares and returns
/// the slope b.  Returns `None` when fewer than 3 data points are available.
fn compute_growth_rate(history: &VecDeque<f64>) -> Option<f64> {
    let n = history.len();
    if n < 3 {
        return None;
    }
    let log_ranks: Vec<f64> = history.iter().map(|&r| r.max(1.0).ln()).collect();
    let mean_x = (n - 1) as f64 / 2.0;
    let mean_y: f64 = log_ranks.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &ly) in log_ranks.iter().enumerate() {
        let xi = i as f64 - mean_x;
        num += xi * (ly - mean_y);
        den += xi * xi;
    }
    let growth_rate = if den > 1e-30 { num / den } else { 0.0 };
    Some(growth_rate)
}

/// Instrumented version of `StrangSplitting` that records per-sub-step ranks and timings.
pub struct InstrumentedStrangSplitting {
    pub inner: super::strang::StrangSplitting,
    /// The diagnostics from the most recent `advance()` call.
    pub last_diagnostics: StepRankDiagnostics,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
    /// Ring buffer of the last `RANK_HISTORY_LEN` max-rank values for
    /// exponential growth rate estimation.
    rank_history: VecDeque<f64>,
}

impl InstrumentedStrangSplitting {
    pub fn new(g: f64) -> Self {
        Self {
            inner: super::strang::StrangSplitting::new(g),
            last_diagnostics: StepRankDiagnostics::default(),
            last_timings: StepTimings::default(),
            progress: None,
            rank_history: VecDeque::with_capacity(RANK_HISTORY_LEN),
        }
    }
}

impl TimeIntegrator for InstrumentedStrangSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("instrumented_strang_advance").entered();
        let mut timings = StepTimings::default();

        if let Some(ref p) = self.progress {
            p.start_step();
            p.set_phase(StepPhase::DriftHalf1);
            p.set_sub_step(0, 5);
        }

        let mut diag = StepRankDiagnostics {
            pre_drift_ranks: extract_ranks(&*repr),
            ..Default::default()
        };

        // Drift half-step
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        diag.post_drift_ranks = extract_ranks(&*repr);

        // Poisson solve + kick
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::PoissonSolve);
            p.set_sub_step(1, 5);
        }
        let t0 = Instant::now();
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.inner.g);
        let accel = solver.compute_acceleration(&potential);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(2, 5);
        }
        let t0 = Instant::now();
        advector.kick(repr, &accel, dt);
        timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        diag.post_kick_ranks = extract_ranks(&*repr);

        // Drift half-step
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::DriftHalf2);
            p.set_sub_step(3, 5);
        }
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        diag.post_final_ranks = extract_ranks(&*repr);

        // Compute amplification ratios
        if let (Some(pre), Some(post_drift)) = (&diag.pre_drift_ranks, &diag.post_drift_ranks) {
            let r_pre = max_rank(pre);
            let r_post = max_rank(post_drift);
            diag.advection_rank_amplification = Some(r_post / r_pre);
        }

        if let (Some(post_drift), Some(post_kick)) = (&diag.post_drift_ranks, &diag.post_kick_ranks)
        {
            let r_drift = max_rank(post_drift);
            let r_kick = max_rank(post_kick);
            diag.poisson_rank_amplification = Some(r_kick / r_drift);
        }

        // ── Rank explosion early-warning ──────────────────────────────────
        // Compute rank_budget_fraction, rank_growth_rate, and singular value spectrum.

        if let Some(ref final_ranks) = diag.post_final_ranks {
            let current_max = max_rank(final_ranks);

            // Update ring buffer
            if self.rank_history.len() >= RANK_HISTORY_LEN {
                self.rank_history.pop_front();
            }
            self.rank_history.push_back(current_max);

            // Rank growth rate (exponential fit over recent history)
            let growth_rate = compute_growth_rate(&self.rank_history);
            diag.rank_growth_rate = growth_rate;

            // Rank budget fraction: current max rank / max_rank budget
            if let Some(budget) = extract_max_rank_budget(&*repr) {
                let budget_f = budget.max(1) as f64;
                diag.rank_budget_fraction = current_max / budget_f;
            }

            // Singular value spectrum
            diag.singular_value_spectrum = extract_singular_values(&*repr);

            // Emit tracing warnings for rank explosion
            if let Some(gr) = growth_rate.filter(|&r| r > 0.5) {
                tracing::warn!(
                    "Rank explosion warning: growth rate {:.2} (doubling every {:.1} steps)",
                    gr,
                    0.693 / gr.max(0.01)
                );
            }

            if diag.rank_budget_fraction > 0.9 {
                tracing::warn!(
                    "Rank budget nearly saturated: {:.0}% of max_rank used",
                    diag.rank_budget_fraction * 100.0
                );
            }
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
            p.set_sub_step(4, 5);
        }

        let t0 = Instant::now();
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.inner.g);
        let acceleration = solver.compute_acceleration(&potential);
        timings.density_ms += t0.elapsed().as_secs_f64() * 1000.0;

        self.last_diagnostics = diag;
        self.last_timings = timings;

        Ok(StepProducts { density, potential, acceleration })
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        self.inner.max_dt(repr, cfl_factor)
    }

    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instrumented_strang_on_uniform_grid() {
        // When repr is UniformGrid6D, all rank fields should be None
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
        use crate::tooling::core::poisson::fft::FftPoisson;

        let domain = Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let mut grid = UniformGrid6D::new(domain.clone());
        // Fill with small constant to avoid trivial density
        for v in grid.data.iter_mut() {
            *v = 1.0;
        }

        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut integrator = InstrumentedStrangSplitting::new(1.0);

        integrator.advance(&mut grid, &poisson, &advector, 0.01).unwrap();

        // UniformGrid6D is not HtTensor, so all rank fields should be None
        assert!(integrator.last_diagnostics.pre_drift_ranks.is_none());
        assert!(integrator.last_diagnostics.post_drift_ranks.is_none());
        assert!(integrator.last_diagnostics.post_kick_ranks.is_none());
        assert!(
            integrator
                .last_diagnostics
                .poisson_rank_amplification
                .is_none()
        );
        assert!(
            integrator
                .last_diagnostics
                .advection_rank_amplification
                .is_none()
        );
        // New early-warning fields should also be None/default for non-HT
        assert!(integrator.last_diagnostics.rank_growth_rate.is_none());
        assert!(
            integrator
                .last_diagnostics
                .singular_value_spectrum
                .is_none()
        );
        assert_eq!(integrator.last_diagnostics.rank_budget_fraction, 0.0);
    }

    #[test]
    fn test_rank_growth_rate_zero_for_equilibrium() {
        // A rank-1 separable HT tensor under tiny time steps should maintain
        // nearly constant rank, giving a growth rate near zero.
        use crate::tooling::core::algos::ht::HtTensor;
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::poisson::fft::FftPoisson;

        let n = 4usize;
        let domain = Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(n as i128)
            .velocity_resolution(n as i128)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        // Rank-1 separable: f = product of 1D Gaussians
        let shape = [n; 6];
        let total = n.pow(6);
        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let idx = i0 * n.pow(5)
                                    + i1 * n.pow(4)
                                    + i2 * n.pow(3)
                                    + i3 * n.pow(2)
                                    + i4 * n
                                    + i5;
                                data[idx] = 1.0; // constant (rank-1)
                            }
                        }
                    }
                }
            }
        }

        let mut ht = HtTensor::from_full(&data, shape, &domain, 1e-10);
        ht.max_rank = 16;
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut integrator = InstrumentedStrangSplitting::new(1.0);

        // Run enough steps to populate the rank history (>= 3)
        let dt = 0.001;
        for _ in 0..5 {
            integrator.advance(&mut ht, &poisson, &advector, dt).unwrap();
        }

        let diag = &integrator.last_diagnostics;

        // Growth rate should be computed (we have >= 3 history entries)
        assert!(
            diag.rank_growth_rate.is_some(),
            "rank_growth_rate should be Some after 5 steps"
        );

        // For a constant (rank-1) tensor, the growth rate should be near zero
        let gr = diag.rank_growth_rate.unwrap();
        assert!(
            gr.abs() < 0.5,
            "rank growth rate should be near zero for equilibrium, got {gr:.4}"
        );
    }

    #[test]
    fn test_rank_budget_fraction_computed() {
        // Verify the budget fraction is between 0 and 1 for an HtTensor run.
        // Use G=0 (free streaming) to avoid Poisson-induced numerical issues
        // on the tiny 4^6 grid.
        use crate::tooling::core::algos::ht::HtTensor;
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::poisson::fft::FftPoisson;

        let n = 4usize;
        let domain = Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(n as i128)
            .velocity_resolution(n as i128)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        // Rank-1 separable product: well-conditioned for HT decomposition.
        let shape = [n; 6];
        let total = n.pow(6);
        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let idx = i0 * n.pow(5)
                                    + i1 * n.pow(4)
                                    + i2 * n.pow(3)
                                    + i3 * n.pow(2)
                                    + i4 * n
                                    + i5;
                                data[idx] = ((i0 + 1)
                                    * (i1 + 1)
                                    * (i2 + 1)
                                    * (i3 + 1)
                                    * (i4 + 1)
                                    * (i5 + 1)) as f64;
                            }
                        }
                    }
                }
            }
        }

        let mut ht = HtTensor::from_full(&data, shape, &domain, 1e-10);
        ht.max_rank = 16;
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        // G = 0: pure free streaming, no Poisson kick — keeps frames well-conditioned
        let mut integrator = InstrumentedStrangSplitting::new(0.0);

        integrator.advance(&mut ht, &poisson, &advector, 0.001).unwrap();

        let diag = &integrator.last_diagnostics;

        // Budget fraction should be in [0, 1]
        assert!(
            diag.rank_budget_fraction >= 0.0 && diag.rank_budget_fraction <= 1.0,
            "rank_budget_fraction should be in [0, 1], got {}",
            diag.rank_budget_fraction
        );

        // Singular value spectrum should be populated for HtTensor
        assert!(
            diag.singular_value_spectrum.is_some(),
            "singular_value_spectrum should be Some for HtTensor"
        );
        let svs = diag.singular_value_spectrum.as_ref().unwrap();
        assert_eq!(svs.len(), 11, "should have 11 nodes for 6D HT");
        // All reported singular values should be non-negative and finite
        for (i, node_svs) in svs.iter().enumerate() {
            for &sv in node_svs {
                assert!(
                    sv >= 0.0 && sv.is_finite(),
                    "node {i}: singular values should be non-negative and finite, got {sv}"
                );
            }
        }
    }
}
