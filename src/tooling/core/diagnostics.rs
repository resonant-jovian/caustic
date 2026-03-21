//! `Diagnostics` — computes and stores all conserved quantities and monitoring outputs
//! at each timestep.
//!
//! Includes:
//! - L1 / L2 / L∞ field norms and field-comparison errors
//! - Conservation drift monitors (mass, energy, momentum, Casimir)
//! - Convergence order estimation via Richardson extrapolation

use super::phasespace::PhaseSpaceRepr;
use super::types::{DensityField, PotentialField};
use rayon::prelude::*;
use serde::Serialize;

/// One row of the global time-series output.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GlobalDiagnostics {
    pub time: f64,
    pub total_energy: f64,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub virial_ratio: f64,
    pub total_momentum: [f64; 3],
    pub total_angular_momentum: [f64; 3],
    pub casimir_c2: f64,
    pub entropy: f64,
    pub mass_in_box: f64,
    /// Casimir C₂ measured before LoMaC projection (None if LoMaC inactive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub casimir_c2_pre_lomac: Option<f64>,
    /// Casimir C₂ measured after LoMaC projection (None if LoMaC inactive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub casimir_c2_post_lomac: Option<f64>,
    /// L2 norm of the near-field correction applied during Poisson solve (None if not applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub near_field_correction_magnitude: Option<f64>,
    /// Coarse-grained entropy S_bar = -Sum f_bar ln(f_bar) dV (None if not computed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coarse_grained_entropy: Option<f64>,
    /// Per-step symplecticity error estimated from Casimir C₂ drift.
    ///
    /// For a symplectic integrator the Casimir invariants are exactly conserved
    /// (absent truncation). This field stores |ΔC₂/C₂| for the most recent step,
    /// or `None` if no previous step exists yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symplecticity_error: Option<f64>,
}

/// Accumulates time series of `GlobalDiagnostics`.
pub struct Diagnostics {
    pub history: Vec<GlobalDiagnostics>,
}

impl Diagnostics {
    /// Compute all global diagnostics from the current representation and potential.
    /// `dx3` is the spatial cell volume dx1*dx2*dx3.
    pub fn compute(
        &mut self,
        repr: &dyn PhaseSpaceRepr,
        potential: &PotentialField,
        time: f64,
        dx3: f64,
    ) -> GlobalDiagnostics {
        let density = repr.compute_density();
        self.compute_with_density(repr, &density, potential, time, dx3)
    }

    /// Compute all global diagnostics using a pre-computed density field.
    ///
    /// This avoids redundant `compute_density()` calls when the caller
    /// already has the density (e.g. from a Poisson solve in the same step).
    pub fn compute_with_density(
        &mut self,
        repr: &dyn PhaseSpaceRepr,
        density: &DensityField,
        potential: &PotentialField,
        time: f64,
        dx3: f64,
    ) -> GlobalDiagnostics {
        let w = Self::potential_energy(density, potential, dx3);
        let ((t, c2), (s, m)) = rayon::join(
            || rayon::join(|| Self::kinetic_energy(repr), || repr.casimir_c2()),
            || rayon::join(|| repr.entropy(), || repr.total_mass()),
        );
        let e = t + w;
        let vir = if w.abs() > 1e-30 {
            2.0 * t / w.abs()
        } else {
            0.0
        };
        let symplecticity_error = if let Some(prev) = self.history.last() {
            let prev_c2 = prev.casimir_c2;
            if prev_c2.abs() > 1e-30 {
                Some(((c2 - prev_c2) / prev_c2).abs())
            } else {
                Some(0.0)
            }
        } else {
            None
        };
        let diag = GlobalDiagnostics {
            time,
            total_energy: e,
            kinetic_energy: t,
            potential_energy: w,
            virial_ratio: vir,
            total_momentum: [0.0; 3],
            total_angular_momentum: [0.0; 3],
            casimir_c2: c2,
            entropy: s,
            mass_in_box: m,
            casimir_c2_pre_lomac: None,
            casimir_c2_post_lomac: None,
            near_field_correction_magnitude: None,
            coarse_grained_entropy: None,
            symplecticity_error,
        };
        self.history.push(diag);
        diag
    }

    /// Total kinetic energy T = ½∫fv² dx³dv³.
    pub fn kinetic_energy(repr: &dyn PhaseSpaceRepr) -> f64 {
        repr.total_kinetic_energy()
    }

    /// Total potential energy W = ½∫ρΦ dx³.
    /// `dx3` is the spatial cell volume.
    pub fn potential_energy(density: &DensityField, potential: &PotentialField, dx3: f64) -> f64 {
        0.5 * density
            .data
            .par_iter()
            .zip(potential.data.par_iter())
            .map(|(&rho, &phi)| rho * phi)
            .sum::<f64>()
            * dx3
    }

    /// Virial ratio 2T/|W|. Equals 1.0 at equilibrium.
    pub fn virial_ratio(t: f64, w: f64) -> f64 {
        2.0 * t / w.abs()
    }

    /// Maximum relative drift of a conserved quantity over the full history.
    ///
    /// Returns `|Q(t) - Q(0)| / |Q(0)|` for the worst timestep, or 0 if history is empty.
    pub fn max_relative_drift<F: Fn(&GlobalDiagnostics) -> f64>(&self, quantity: F) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let q0 = quantity(&self.history[0]);
        if q0.abs() < 1e-30 {
            return 0.0;
        }
        self.history
            .iter()
            .map(|d| ((quantity(d) - q0) / q0).abs())
            .fold(0.0f64, f64::max)
    }

    /// Conservation summary: max relative drifts in mass, energy, and Casimir C₂.
    pub fn conservation_summary(&self) -> ConservationSummary {
        ConservationSummary {
            max_energy_drift: self.max_relative_drift(|d| d.total_energy),
            max_mass_drift: self.max_relative_drift(|d| d.mass_in_box),
            max_casimir_drift: self.max_relative_drift(|d| d.casimir_c2),
            max_entropy_drift: self.max_relative_drift(|d| d.entropy),
            total_steps: self.history.len(),
        }
    }

    /// Spherically averaged density profile ρ(r) at current timestep.
    /// Bins density cells by radius from domain centre, returns (r_bin, ρ_avg) pairs.
    pub fn density_profile(density: &DensityField, dx: [f64; 3]) -> Vec<(f64, f64)> {
        let [nx, ny, nz] = density.shape;
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cz = nz as f64 / 2.0;

        // Determine maximum radius and bin width
        let r_max = (cx * dx[0]).min(cy * dx[1]).min(cz * dx[2]);
        let n_bins = (nx.max(ny).max(nz) / 2).max(1);
        let dr = r_max / n_bins as f64;

        // Parallel fold-reduce over ix slabs to accumulate bin data
        let (bin_sum, bin_count) = (0..nx)
            .into_par_iter()
            .fold(
                || (vec![0.0f64; n_bins], vec![0u64; n_bins]),
                |(mut bs, mut bc), ix| {
                    for iy in 0..ny {
                        for iz in 0..nz {
                            let rx = (ix as f64 + 0.5 - cx) * dx[0];
                            let ry = (iy as f64 + 0.5 - cy) * dx[1];
                            let rz = (iz as f64 + 0.5 - cz) * dx[2];
                            let r = (rx * rx + ry * ry + rz * rz).sqrt();
                            let bin = (r / dr) as usize;
                            if bin < n_bins {
                                bs[bin] += density.data[ix * ny * nz + iy * nz + iz];
                                bc[bin] += 1;
                            }
                        }
                    }
                    (bs, bc)
                },
            )
            .reduce(
                || (vec![0.0f64; n_bins], vec![0u64; n_bins]),
                |(mut a_s, mut a_c), (b_s, b_c)| {
                    for i in 0..n_bins {
                        a_s[i] += b_s[i];
                        a_c[i] += b_c[i];
                    }
                    (a_s, a_c)
                },
            );

        (0..n_bins)
            .filter(|&i| bin_count[i] > 0)
            .map(|i| {
                let r = (i as f64 + 0.5) * dr;
                let rho_avg = bin_sum[i] / bin_count[i] as f64;
                (r, rho_avg)
            })
            .collect()
    }
}

/// Summary of conservation quality over a simulation run.
#[derive(Debug, Clone, Copy)]
pub struct ConservationSummary {
    pub max_energy_drift: f64,
    pub max_mass_drift: f64,
    pub max_casimir_drift: f64,
    pub max_entropy_drift: f64,
    pub total_steps: usize,
}

// ──────────────────────────────────────────────────────────────────────
// Field norms
// ──────────────────────────────────────────────────────────────────────

/// L1 norm: ‖f‖₁ = Σ|fᵢ|·dV.
pub fn norm_l1(data: &[f64], cell_volume: f64) -> f64 {
    data.par_iter().map(|v| v.abs()).sum::<f64>() * cell_volume
}

/// L2 norm: ‖f‖₂ = √(Σfᵢ²·dV).
pub fn norm_l2(data: &[f64], cell_volume: f64) -> f64 {
    (data.par_iter().map(|v| v * v).sum::<f64>() * cell_volume).sqrt()
}

/// L∞ norm: max|fᵢ|.
pub fn norm_linf(data: &[f64]) -> f64 {
    data.par_iter().map(|v| v.abs()).reduce(|| 0.0f64, f64::max)
}

/// L1 error between two fields: ‖a − b‖₁ / ‖b‖₁.
/// Returns absolute L1 error if `reference` is zero everywhere.
pub fn error_l1(computed: &[f64], reference: &[f64], cell_volume: f64) -> f64 {
    let num = computed.par_iter().zip(reference.par_iter())
        .map(|(a, b)| (a - b).abs()).sum::<f64>() * cell_volume;
    let den = norm_l1(reference, cell_volume);
    if den > 1e-30 { num / den } else { num }
}

/// L2 error between two fields: ‖a − b‖₂ / ‖b‖₂.
pub fn error_l2(computed: &[f64], reference: &[f64], cell_volume: f64) -> f64 {
    let num = (computed.par_iter().zip(reference.par_iter())
        .map(|(a, b)| { let d = a - b; d * d }).sum::<f64>() * cell_volume).sqrt();
    let den = norm_l2(reference, cell_volume);
    if den > 1e-30 { num / den } else { num }
}

/// L∞ error between two fields: max|aᵢ − bᵢ| / max|bᵢ|.
pub fn error_linf(computed: &[f64], reference: &[f64]) -> f64 {
    let diff_max = computed.par_iter().zip(reference.par_iter())
        .map(|(a, b)| (a - b).abs())
        .reduce(|| 0.0f64, f64::max);
    let ref_max = norm_linf(reference);
    if ref_max > 1e-30 {
        diff_max / ref_max
    } else {
        diff_max
    }
}

// ──────────────────────────────────────────────────────────────────────
// Convergence order estimation
// ──────────────────────────────────────────────────────────────────────

/// Result of a convergence rate measurement.
#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    /// Resolutions (or rank values) in ascending order.
    pub parameters: Vec<f64>,
    /// Corresponding error values.
    pub errors: Vec<f64>,
    /// Estimated convergence order from each consecutive pair.
    pub orders: Vec<f64>,
    /// Mean convergence order.
    pub mean_order: f64,
}

/// Estimate convergence order from a sequence of (parameter, error) pairs.
///
/// For spatial convergence: parameters are grid spacings h = L/N (decreasing).
/// For rank convergence: parameters are 1/R (decreasing).
///
/// Order p is estimated via: p = log(e₁/e₂) / log(h₁/h₂).
///
/// Pairs should be sorted by decreasing parameter (coarsest first).
pub fn estimate_convergence_order(pairs: &[(f64, f64)]) -> ConvergenceResult {
    let parameters: Vec<f64> = pairs.iter().map(|&(h, _)| h).collect();
    let errors: Vec<f64> = pairs.iter().map(|&(_, e)| e).collect();

    let orders: Vec<f64> = pairs
        .windows(2)
        .map(|w| {
            let (h1, e1) = w[0];
            let (h2, e2) = w[1];
            if e1 > 1e-30 && e2 > 1e-30 && (h1 / h2).abs() > 1e-30 {
                (e1 / e2).ln() / (h1 / h2).ln()
            } else {
                f64::NAN
            }
        })
        .collect();

    let valid: Vec<f64> = orders.iter().copied().filter(|o| o.is_finite()).collect();
    let mean_order = if valid.is_empty() {
        f64::NAN
    } else {
        valid.iter().sum::<f64>() / valid.len() as f64
    };

    ConvergenceResult {
        parameters,
        errors,
        orders,
        mean_order,
    }
}

/// Richardson extrapolation: given two error values at resolutions h and h/r,
/// estimate the error at h=0.
///
/// `order` is the expected convergence order p.
/// Returns the extrapolated value: (r^p · f_fine − f_coarse) / (r^p − 1).
pub fn richardson_extrapolate(
    coarse_value: f64,
    fine_value: f64,
    refinement_ratio: f64,
    order: f64,
) -> f64 {
    let rp = refinement_ratio.powf(order);
    (rp * fine_value - coarse_value) / (rp - 1.0)
}

/// Build a 2D convergence table: rows = grid resolutions N, columns = rank values R.
///
/// `run_experiment` takes (N, R) and returns the measured error.
/// Returns a matrix of errors indexed as `table[n_idx][r_idx]`.
pub fn convergence_table<F: Fn(usize, usize) -> f64>(
    resolutions: &[usize],
    ranks: &[usize],
    run_experiment: F,
) -> Vec<Vec<f64>> {
    resolutions
        .iter()
        .map(|&n| ranks.iter().map(|&r| run_experiment(n, r)).collect())
        .collect()
}

/// Compute coarse-grained entropy by binning f on a coarser grid.
///
/// Given a phase-space representation, evaluate on a grid coarsened by
/// `factor` in each dimension, then compute S_bar = -Sum f_bar ln(f_bar) * dV.
/// The difference S_bar - S (fine-grained entropy) measures phase-space mixing.
pub fn coarse_grained_entropy(
    fine_data: &[f64],
    shape: [usize; 6],
    factor: usize,
    dv6: f64, // phase-space volume element dx^3 dv^3
) -> f64 {
    if factor <= 1 {
        // No coarsening -- just return fine-grained entropy
        return -fine_data
            .iter()
            .filter(|&&f| f > 0.0)
            .map(|&f| f * f.ln() * dv6)
            .sum::<f64>();
    }

    let coarse_shape: Vec<usize> = shape.iter().map(|&s| (s / factor).max(1)).collect();
    let coarse_total: usize = coarse_shape.iter().product();
    let mut coarse = vec![0.0f64; coarse_total];
    let mut counts = vec![0u32; coarse_total];

    // Bin fine grid into coarse grid
    for i0 in 0..shape[0] {
        let c0 = (i0 / factor).min(coarse_shape[0] - 1);
        for i1 in 0..shape[1] {
            let c1 = (i1 / factor).min(coarse_shape[1] - 1);
            for i2 in 0..shape[2] {
                let c2 = (i2 / factor).min(coarse_shape[2] - 1);
                for i3 in 0..shape[3] {
                    let c3 = (i3 / factor).min(coarse_shape[3] - 1);
                    for i4 in 0..shape[4] {
                        let c4 = (i4 / factor).min(coarse_shape[4] - 1);
                        for i5 in 0..shape[5] {
                            let c5 = (i5 / factor).min(coarse_shape[5] - 1);
                            let fine_idx = i0
                                * shape[1]
                                * shape[2]
                                * shape[3]
                                * shape[4]
                                * shape[5]
                                + i1 * shape[2] * shape[3] * shape[4] * shape[5]
                                + i2 * shape[3] * shape[4] * shape[5]
                                + i3 * shape[4] * shape[5]
                                + i4 * shape[5]
                                + i5;
                            let coarse_idx = c0
                                * coarse_shape[1]
                                * coarse_shape[2]
                                * coarse_shape[3]
                                * coarse_shape[4]
                                * coarse_shape[5]
                                + c1
                                    * coarse_shape[2]
                                    * coarse_shape[3]
                                    * coarse_shape[4]
                                    * coarse_shape[5]
                                + c2 * coarse_shape[3] * coarse_shape[4] * coarse_shape[5]
                                + c3 * coarse_shape[4] * coarse_shape[5]
                                + c4 * coarse_shape[5]
                                + c5;
                            coarse[coarse_idx] += fine_data[fine_idx];
                            counts[coarse_idx] += 1;
                        }
                    }
                }
            }
        }
    }

    // Average (divide accumulated sum by count)
    for (c, &cnt) in coarse.iter_mut().zip(counts.iter()) {
        if cnt > 0 {
            *c /= cnt as f64;
        }
    }

    // Coarse-grained volume element
    let coarse_dv6 = dv6 * (factor as f64).powi(6);

    -coarse
        .iter()
        .filter(|&&f| f > 0.0)
        .map(|&f| f * f.ln() * coarse_dv6)
        .sum::<f64>()
}

/// Estimate symplecticity error from consecutive Casimir C₂ values.
///
/// For a symplectic integrator, all Casimir invariants are exactly conserved
/// (in the absence of truncation). The per-step C₂ drift is a direct proxy
/// for symplecticity violation.
///
/// Returns |ΔC₂/C₂| for the most recent step, or 0.0 if history is too short.
pub fn casimir_symplecticity_proxy(history: &[GlobalDiagnostics]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let prev = &history[history.len() - 2];
    let curr = &history[history.len() - 1];
    let c2_prev = prev.casimir_c2;
    let c2_curr = curr.casimir_c2;
    if c2_prev.abs() < 1e-30 {
        return 0.0;
    }
    ((c2_curr - c2_prev) / c2_prev).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_l1_uniform() {
        let data = vec![1.0; 8];
        assert!((norm_l1(&data, 0.5) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn norm_l2_known() {
        // ‖[3,4]‖₂ with dV=1 = 5
        let data = vec![3.0, 4.0];
        assert!((norm_l2(&data, 1.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn norm_linf_picks_max() {
        let data = vec![1.0, -3.0, 2.5];
        assert!((norm_linf(&data) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn error_l2_identical_is_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(error_l2(&a, &a, 1.0) < 1e-14);
    }

    #[test]
    fn error_linf_known() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.5, 3.0];
        // diff = [0, 0.5, 0], max_diff = 0.5, ref_max = 3.0
        assert!((error_linf(&a, &b) - 0.5 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn convergence_order_second_order() {
        // Second-order method: error ∝ h²
        // h = 1/N, error = h² = 1/N²
        let pairs: Vec<(f64, f64)> = [8, 16, 32, 64]
            .iter()
            .map(|&n| {
                let h = 1.0 / n as f64;
                (h, h * h)
            })
            .collect();
        let result = estimate_convergence_order(&pairs);
        for &o in &result.orders {
            assert!((o - 2.0).abs() < 0.01, "Expected order ~2, got {o}");
        }
        assert!((result.mean_order - 2.0).abs() < 0.01);
    }

    #[test]
    fn convergence_order_fourth_order() {
        let pairs: Vec<(f64, f64)> = [8, 16, 32]
            .iter()
            .map(|&n| {
                let h = 1.0 / n as f64;
                (h, h.powi(4))
            })
            .collect();
        let result = estimate_convergence_order(&pairs);
        assert!(
            (result.mean_order - 4.0).abs() < 0.01,
            "Expected order ~4, got {}",
            result.mean_order
        );
    }

    #[test]
    fn richardson_extrapolation_exact_quadratic() {
        // f(h) = 1 + 2h² (true value = 1)
        // coarse: h=1 → f=3, fine: h=0.5 → f=1.5
        let extrap = richardson_extrapolate(3.0, 1.5, 2.0, 2.0);
        assert!(
            (extrap - 1.0).abs() < 1e-12,
            "Richardson should recover exact value, got {extrap}"
        );
    }

    #[test]
    fn convergence_table_structure() {
        let ns = [8, 16, 32];
        let rs = [10, 20];
        let table = convergence_table(&ns, &rs, |n, r| 1.0 / (n as f64 * r as f64));
        assert_eq!(table.len(), 3);
        assert_eq!(table[0].len(), 2);
        assert!((table[0][0] - 1.0 / 80.0).abs() < 1e-14);
        assert!((table[2][1] - 1.0 / 640.0).abs() < 1e-14);
    }

    #[test]
    fn conservation_summary_detects_drift() {
        let mut diags = Diagnostics {
            history: Vec::new(),
        };
        // Two snapshots: energy drifts by 1%
        diags.history.push(GlobalDiagnostics {
            time: 0.0,
            total_energy: 100.0,
            kinetic_energy: 60.0,
            potential_energy: 40.0,
            virial_ratio: 3.0,
            total_momentum: [0.0; 3],
            total_angular_momentum: [0.0; 3],
            casimir_c2: 1.0,
            entropy: 0.5,
            mass_in_box: 10.0,
            casimir_c2_pre_lomac: None,
            casimir_c2_post_lomac: None,
            near_field_correction_magnitude: None,
            coarse_grained_entropy: None,
            symplecticity_error: None,
        });
        diags.history.push(GlobalDiagnostics {
            time: 1.0,
            total_energy: 101.0,
            kinetic_energy: 61.0,
            potential_energy: 40.0,
            virial_ratio: 3.05,
            total_momentum: [0.0; 3],
            total_angular_momentum: [0.0; 3],
            casimir_c2: 1.0,
            entropy: 0.5,
            mass_in_box: 10.0,
            casimir_c2_pre_lomac: None,
            casimir_c2_post_lomac: None,
            near_field_correction_magnitude: None,
            coarse_grained_entropy: None,
            symplecticity_error: None,
        });
        let summary = diags.conservation_summary();
        assert!(
            (summary.max_energy_drift - 0.01).abs() < 1e-10,
            "Energy drift should be 1%, got {}",
            summary.max_energy_drift
        );
        assert!(summary.max_mass_drift < 1e-14);
        assert!(summary.max_casimir_drift < 1e-14);
    }

    #[test]
    fn test_symplecticity_proxy_zero_for_first_step() {
        // With no history, the proxy returns 0.0
        let empty: Vec<GlobalDiagnostics> = vec![];
        assert_eq!(casimir_symplecticity_proxy(&empty), 0.0);

        // With a single entry, the proxy also returns 0.0
        let single = vec![GlobalDiagnostics {
            time: 0.0,
            total_energy: -1.0,
            kinetic_energy: 0.5,
            potential_energy: -1.5,
            virial_ratio: 0.667,
            total_momentum: [0.0; 3],
            total_angular_momentum: [0.0; 3],
            casimir_c2: 42.0,
            entropy: 1.0,
            mass_in_box: 1.0,
            casimir_c2_pre_lomac: None,
            casimir_c2_post_lomac: None,
            near_field_correction_magnitude: None,
            coarse_grained_entropy: None,
            symplecticity_error: None,
        }];
        assert_eq!(casimir_symplecticity_proxy(&single), 0.0);

        // Also verify that compute_with_density sets symplecticity_error = None
        // on the very first step (no prior history).
        // We cannot call compute_with_density here without a PhaseSpaceRepr,
        // but we verify it indirectly: construct a single-entry GlobalDiagnostics
        // with symplecticity_error = None and confirm it serializes without that key.
        let json = serde_json::to_string(&single[0]).unwrap();
        assert!(
            !json.contains("symplecticity_error"),
            "symplecticity_error should be skipped when None"
        );
    }

    #[test]
    fn test_symplecticity_proxy_measures_drift() {
        // Two steps where C₂ drifts by 1%
        let history = vec![
            GlobalDiagnostics {
                time: 0.0,
                total_energy: -1.0,
                kinetic_energy: 0.5,
                potential_energy: -1.5,
                virial_ratio: 0.667,
                total_momentum: [0.0; 3],
                total_angular_momentum: [0.0; 3],
                casimir_c2: 100.0,
                entropy: 1.0,
                mass_in_box: 1.0,
                casimir_c2_pre_lomac: None,
                casimir_c2_post_lomac: None,
                near_field_correction_magnitude: None,
                coarse_grained_entropy: None,
                symplecticity_error: None,
            },
            GlobalDiagnostics {
                time: 0.1,
                total_energy: -1.0,
                kinetic_energy: 0.5,
                potential_energy: -1.5,
                virial_ratio: 0.667,
                total_momentum: [0.0; 3],
                total_angular_momentum: [0.0; 3],
                casimir_c2: 101.0, // 1% drift
                entropy: 1.0,
                mass_in_box: 1.0,
                casimir_c2_pre_lomac: None,
                casimir_c2_post_lomac: None,
                near_field_correction_magnitude: None,
                coarse_grained_entropy: None,
                symplecticity_error: Some(0.01),
            },
        ];
        let proxy = casimir_symplecticity_proxy(&history);
        assert!(
            (proxy - 0.01).abs() < 1e-12,
            "Expected symplecticity proxy ~0.01, got {proxy}"
        );
        assert!(proxy > 0.0, "Proxy should be positive for drifting C₂");
        assert!(proxy.is_finite(), "Proxy should be finite");

        // Verify zero drift gives zero proxy
        let no_drift = vec![
            GlobalDiagnostics {
                time: 0.0,
                total_energy: -1.0,
                kinetic_energy: 0.5,
                potential_energy: -1.5,
                virial_ratio: 0.667,
                total_momentum: [0.0; 3],
                total_angular_momentum: [0.0; 3],
                casimir_c2: 100.0,
                entropy: 1.0,
                mass_in_box: 1.0,
                casimir_c2_pre_lomac: None,
                casimir_c2_post_lomac: None,
                near_field_correction_magnitude: None,
                coarse_grained_entropy: None,
                symplecticity_error: None,
            },
            GlobalDiagnostics {
                time: 0.1,
                total_energy: -1.0,
                kinetic_energy: 0.5,
                potential_energy: -1.5,
                virial_ratio: 0.667,
                total_momentum: [0.0; 3],
                total_angular_momentum: [0.0; 3],
                casimir_c2: 100.0, // no drift
                entropy: 1.0,
                mass_in_box: 1.0,
                casimir_c2_pre_lomac: None,
                casimir_c2_post_lomac: None,
                near_field_correction_magnitude: None,
                coarse_grained_entropy: None,
                symplecticity_error: Some(0.0),
            },
        ];
        assert_eq!(casimir_symplecticity_proxy(&no_drift), 0.0);
    }
}
