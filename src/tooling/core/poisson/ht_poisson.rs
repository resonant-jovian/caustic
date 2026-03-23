//! HT-format Poisson solver with isolated boundary conditions.
//!
//! Solves ∇²Φ = 4πGρ using the Braess-Hackbusch exponential sum in HT tensor format.
//! The Green's function 1/r ≈ Σ c_k exp(-α_k r²) is a sum of separable Gaussians,
//! each of rank 1 in the HT format. Convolution in Fourier space preserves rank:
//! multiplying a rank-r_ρ density by a rank-1 Green's term produces rank-r_ρ.
//! Summing R_G such terms gives rank R_G × r_ρ before truncation.
//!
//! Pipeline:
//! 1. Dense DensityField → HtTensor3D (via from_dense with tolerance)
//! 2. Zero-pad leaf frames to 2N
//! 3. FFT leaves → HtTensor3DComplex
//! 4. For each Gaussian term k: scale leaf columns by rank-1 diagonal
//! 5. Sum all R_G terms via add → rank = R_G × r_density
//! 6. IFFT leaves → HtTensor3D
//! 7. Truncate to re-compress
//! 8. Extract to dense PotentialField

use super::super::solver::PoissonSolver;
use super::super::types::*;
use super::exponential_sum::ExponentialSumCoefficients;
use super::utils::{finite_difference_acceleration, spectral_laplacian};
use crate::tooling::core::algos::ht3d::{HtNode3D, HtNode3DComplex, HtTensor3D, HtTensor3DComplex};
use faer::Mat;
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// HT-format Poisson solver for isolated BCs.
///
/// Pre-computes the exponential sum coefficients and the per-term 1D Fourier
/// factors so that each `solve()` only needs: from_dense → pad → FFT → multiply
/// → sum → IFFT → truncate → extract.
pub struct HtPoisson {
    pub shape: [usize; 3],
    pub dx: [f64; 3],
    /// Exponential sum: 1/r ≈ Σ c_k exp(-α_k r²).
    exp_c: Vec<f64>,
    exp_alpha: Vec<f64>,
    /// Number of exp-sum terms.
    r_g: usize,
    /// Padded shape [2N0, 2N1, 2N2].
    padded_shape: [usize; 3],
    /// Volume element dx³.
    dv: f64,
    /// Tolerance for HT compression of density.
    tolerance: f64,
    /// Max rank for HT compression.
    max_rank: usize,
    /// Hockney-Eastwood G(0) regularization value.
    g_origin: f64,
    /// Pre-computed: FFT of 1D Gaussian factors per exp-sum term per axis.
    /// Indexed as green_factors[k][axis] = Vec<Complex64> of length padded_shape[axis].
    green_factors: Vec<[Vec<Complex64>; 3]>,
    /// Pre-computed prefactor = -1/(4π).
    prefactor: f64,
    /// Near-field correction: ∫_{B_δ} [1/|y| - GS(|y|)] dy.
    near_field_integral: f64,
    /// 2nd-order near-field correction integral.
    near_field_integral_2: f64,
    /// Cached forward FFT plans for leaf frames [axis0, axis1, axis2] at padded size.
    leaf_fft_plans: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Cached inverse FFT plans for leaf frames.
    leaf_ifft_plans: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Whether near-field correction is applied during solve.
    near_field_enabled: bool,
    /// L2 norm of last near-field correction (stored as f64 bits for atomic access).
    last_near_field_l2: AtomicU64,
    /// Shared progress state for intra-phase reporting.
    progress: Option<Arc<super::super::progress::StepProgress>>,
}

impl HtPoisson {
    pub fn new(
        shape: [usize; 3],
        dx: [f64; 3],
        exp_sum_accuracy: f64,
        tolerance: f64,
        max_rank: usize,
    ) -> Self {
        let delta = dx[0].min(dx[1]).min(dx[2]);
        let r_max = ((shape[0] as f64 * dx[0]).powi(2)
            + (shape[1] as f64 * dx[1]).powi(2)
            + (shape[2] as f64 * dx[2]).powi(2))
        .sqrt();

        let exp_sum = ExponentialSumCoefficients::compute(delta, r_max, exp_sum_accuracy);
        let padded_shape = [shape[0] * 2, shape[1] * 2, shape[2] * 2];

        let dx_avg = (dx[0] + dx[1] + dx[2]) / 3.0;
        let g_origin = -2.38 * dx_avg / (4.0 * std::f64::consts::PI);
        let prefactor = -1.0 / (4.0 * std::f64::consts::PI);

        // Near-field correction integrals (Exl, Mauser & Zhang, JCP 2016)
        let near_field_integral = exp_sum.near_field_correction_integral(delta);
        let (_, near_field_integral_2) = exp_sum.near_field_correction_second_order(delta);

        // Pre-compute per-term FFT of 1D Gaussian factors:
        // For term k with α_k, the 3D Green's term is:
        //   G_k(x) = c_k * exp(-α_k * x₁²) * exp(-α_k * x₂²) * exp(-α_k * x₃²)
        // After zero-padding (Hockney), the 1D factor along axis d is:
        //   g_d[i] = exp(-α_k * (min_image(i, 2N_d) * dx_d)²)
        let mut green_factors = Vec::with_capacity(exp_sum.r_g);
        let mut planner = FftPlanner::new();

        for k in 0..exp_sum.r_g {
            let alpha_k = exp_sum.alpha[k];
            let mut factors: [Vec<Complex64>; 3] = [vec![], vec![], vec![]];

            for d in 0..3 {
                let pn = padded_shape[d];
                let fft = planner.plan_fft_forward(pn);
                let mut buf: Vec<Complex64> = (0..pn)
                    .map(|i| {
                        let dist = min_image_dist(i, pn) * dx[d];
                        Complex64::new((-alpha_k * dist * dist).exp(), 0.0)
                    })
                    .collect();
                fft.process(&mut buf);
                factors[d] = buf;
            }

            green_factors.push(factors);
        }

        // Precompute leaf FFT plans for padded dimensions
        let leaf_fft_plans = [
            planner.plan_fft_forward(padded_shape[0]),
            planner.plan_fft_forward(padded_shape[1]),
            planner.plan_fft_forward(padded_shape[2]),
        ];
        let leaf_ifft_plans = [
            planner.plan_fft_inverse(padded_shape[0]),
            planner.plan_fft_inverse(padded_shape[1]),
            planner.plan_fft_inverse(padded_shape[2]),
        ];

        Self {
            shape,
            dx,
            exp_c: exp_sum.c,
            exp_alpha: exp_sum.alpha,
            r_g: exp_sum.r_g,
            padded_shape,
            dv: dx[0] * dx[1] * dx[2],
            tolerance,
            max_rank,
            g_origin,
            green_factors,
            prefactor,
            near_field_integral,
            near_field_integral_2,
            near_field_enabled: true,
            last_near_field_l2: AtomicU64::new(0u64),
            leaf_fft_plans,
            leaf_ifft_plans,
            progress: None,
        }
    }

    pub fn from_domain(
        domain: &crate::tooling::core::init::domain::Domain,
        exp_sum_accuracy: f64,
        tolerance: f64,
        max_rank: usize,
    ) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];
        let dx = domain.dx();
        Self::new(shape, dx, exp_sum_accuracy, tolerance, max_rank)
    }

    /// Per-step rank diagnostics: returns (pre_truncation_ranks, post_truncation_ranks).
    pub fn last_rank_info(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        // Stored by solve if instrumentation is added later
        None
    }

    /// Enable or disable the near-field correction (builder pattern).
    pub fn with_near_field_enabled(mut self, enabled: bool) -> Self {
        self.near_field_enabled = enabled;
        self
    }

    /// Return the L2 norm of the most recent near-field correction.
    ///
    /// Returns 0.0 if no solve has been performed yet or if the correction is disabled.
    pub fn last_near_field_magnitude(&self) -> f64 {
        f64::from_bits(self.last_near_field_l2.load(Ordering::Relaxed))
    }
}

impl PoissonSolver for HtPoisson {
    fn set_progress(&mut self, p: std::sync::Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let [nx, ny, nz] = self.shape;
        let [px, py, pz] = self.padded_shape;

        // Step 1: Zero-pad density to (2N)³ (parallel over i0 slabs)
        let mut padded = vec![0.0f64; px * py * pz];
        padded
            .par_chunks_mut(py * pz)
            .enumerate()
            .for_each(|(i0, slab)| {
                if i0 < nx {
                    for i1 in 0..ny {
                        for i2 in 0..nz {
                            slab[i1 * pz + i2] = density.data[i0 * ny * nz + i1 * nz + i2];
                        }
                    }
                }
            });

        // Step 2: Convert to HtTensor3D via from_dense
        let rho_ht = HtTensor3D::from_dense(
            &padded,
            self.padded_shape,
            [self.dx[0], self.dx[1], self.dx[2]],
            self.tolerance,
            self.max_rank,
        );

        // Step 3: FFT leaves (using cached plans)
        let rho_hat = rho_ht.into_fft_with_plans(&self.leaf_fft_plans);

        // Step 4: For each Gaussian term, scale density in Fourier space by
        // the rank-1 Green's factor, then accumulate via add.
        // Each term: Φ̂_k = prefactor * c_k * diag(ĝ_k^(0)) ⊗ diag(ĝ_k^(1)) ⊗ diag(ĝ_k^(2)) · ρ̂
        // In HT format: scale each leaf column by the diagonal → preserves rank exactly.
        let mut phi_hat: Option<HtTensor3DComplex> = None;

        let total = self.r_g as u64;
        for k in 0..self.r_g {
            let c_k = self.exp_c[k];
            let scale = self.prefactor * c_k;

            // Scale rho_hat leaf frames by Green's factor diagonals
            let scaled = scale_complex_ht(&rho_hat, &self.green_factors[k], scale);

            phi_hat = Some(match phi_hat {
                None => scaled,
                Some(acc) => add_complex_ht(&acc, &scaled),
            });

            if let Some(ref p) = self.progress {
                p.set_intra_progress(k as u64 + 1, total);
            }
        }

        let phi_hat = match phi_hat {
            Some(h) => h,
            None => {
                // No terms — return zero potential
                return PotentialField {
                    data: vec![0.0; nx * ny * nz],
                    shape: [nx, ny, nz],
                };
            }
        };

        // Step 5: IFFT leaves (using cached plans)
        let mut phi_ht = phi_hat.into_ifft_with_plans(&self.leaf_ifft_plans);

        // Step 6: Truncate to re-compress (rank may be R_G × r_ρ)
        phi_ht.truncate(self.tolerance, self.max_rank);

        // Step 7: Extract N³ subgrid via batch Khatri-Rao and scale by 4πG·dx³
        let scale = 4.0 * std::f64::consts::PI * g * self.dv;
        let mut data = phi_ht.to_dense_subgrid([nx, ny, nz]);
        data.par_iter_mut().for_each(|v| *v *= scale);

        // Step 8: Apply near-field corrections (Exl, Mauser & Zhang, JCP 2016)
        let mut corr_l2_sq = 0.0f64;

        if self.near_field_enabled {
            // 0th-order: Φ_corr(x) = -G/(4π) · ρ(x) · I₀ · dx³
            if self.near_field_integral.abs() > 1e-30 {
                let nf_scale =
                    -g / (4.0 * std::f64::consts::PI) * self.near_field_integral * self.dv;
                corr_l2_sq += density
                    .data
                    .par_iter()
                    .map(|&rho| (nf_scale * rho).powi(2))
                    .sum::<f64>();
                data.par_iter_mut()
                    .zip(density.data.par_iter())
                    .for_each(|(phi, &rho)| {
                        *phi += nf_scale * rho;
                    });
            }

            // 2nd-order: Φ_corr_2(x) = -G · I₂ · ∇²ρ(x)
            if self.near_field_integral_2.abs() > 1e-30 {
                let lap_rho = spectral_laplacian(density, &self.dx);
                let nf2_scale = -g * self.near_field_integral_2;
                corr_l2_sq += lap_rho
                    .data
                    .par_iter()
                    .map(|&lap| (nf2_scale * lap).powi(2))
                    .sum::<f64>();
                data.par_iter_mut()
                    .zip(lap_rho.data.par_iter())
                    .for_each(|(phi, &lap)| {
                        *phi += nf2_scale * lap;
                    });
            }
        }

        self.last_near_field_l2
            .store(corr_l2_sq.sqrt().to_bits(), Ordering::Relaxed);

        PotentialField {
            data,
            shape: [nx, ny, nz],
        }
    }

    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        finite_difference_acceleration(potential, &self.dx)
    }
}

/// Scale a complex HT tensor's leaf frames by rank-1 diagonal factors.
/// For each leaf d, column j gets multiplied pointwise by factors[d][i].
/// This preserves rank exactly.
fn scale_complex_ht(
    ht: &HtTensor3DComplex,
    factors: &[Vec<Complex64>; 3],
    scalar: f64,
) -> HtTensor3DComplex {
    let mut nodes = Vec::with_capacity(ht.nodes.len());

    for (idx, node) in ht.nodes.iter().enumerate() {
        match node {
            HtNode3DComplex::Leaf {
                dim,
                frame_re,
                frame_im,
            } => {
                let n = frame_re.nrows();
                let k = frame_re.ncols();
                let factor = &factors[*dim];
                let mut new_re = Mat::zeros(n, k);
                let mut new_im = Mat::zeros(n, k);

                // Apply scalar to first leaf only (to avoid multiplying it 3 times)
                let s = if *dim == 0 { scalar } else { 1.0 };

                for col in 0..k {
                    for row in 0..n {
                        let f_re = factor[row].re;
                        let f_im = factor[row].im;
                        let v_re = frame_re[(row, col)];
                        let v_im = frame_im[(row, col)];
                        // Complex multiply: (f_re + i f_im) * (v_re + i v_im)
                        new_re[(row, col)] = s * (f_re * v_re - f_im * v_im);
                        new_im[(row, col)] = s * (f_re * v_im + f_im * v_re);
                    }
                }

                nodes.push(HtNode3DComplex::Leaf {
                    dim: *dim,
                    frame_re: new_re,
                    frame_im: new_im,
                });
            }
            HtNode3DComplex::Interior {
                left,
                right,
                transfer,
                ranks,
            } => {
                nodes.push(HtNode3DComplex::Interior {
                    left: *left,
                    right: *right,
                    transfer: transfer.clone(),
                    ranks: *ranks,
                });
            }
        }
    }

    HtTensor3DComplex {
        nodes,
        shape: ht.shape,
        dx: ht.dx,
    }
}

/// Add two complex HT tensors via rank concatenation.
/// Leaf and non-root interior ranks are concatenated block-diagonally.
/// The root node keeps rank 1 (scalar tensor): block-diagonal in (l, r) only,
/// matching HtTensor3D::add.
fn add_complex_ht(a: &HtTensor3DComplex, b: &HtTensor3DComplex) -> HtTensor3DComplex {
    assert_eq!(a.shape, b.shape);
    let mut nodes = Vec::with_capacity(a.nodes.len());
    let root_idx = a.nodes.len() - 1;

    for (idx, (na, nb)) in a.nodes.iter().zip(b.nodes.iter()).enumerate() {
        match (na, nb) {
            (
                HtNode3DComplex::Leaf {
                    dim: da,
                    frame_re: fra,
                    frame_im: fia,
                },
                HtNode3DComplex::Leaf {
                    dim: _,
                    frame_re: frb,
                    frame_im: fib,
                },
            ) => {
                let n = fra.nrows();
                let ka = fra.ncols();
                let kb = frb.ncols();
                let mut new_re = Mat::zeros(n, ka + kb);
                let mut new_im = Mat::zeros(n, ka + kb);

                // Copy A columns
                for col in 0..ka {
                    for row in 0..n {
                        new_re[(row, col)] = fra[(row, col)];
                        new_im[(row, col)] = fia[(row, col)];
                    }
                }
                // Copy B columns
                for col in 0..kb {
                    for row in 0..n {
                        new_re[(row, ka + col)] = frb[(row, col)];
                        new_im[(row, ka + col)] = fib[(row, col)];
                    }
                }

                nodes.push(HtNode3DComplex::Leaf {
                    dim: *da,
                    frame_re: new_re,
                    frame_im: new_im,
                });
            }
            (
                HtNode3DComplex::Interior {
                    left: la,
                    right: ra,
                    transfer: ta,
                    ranks: rka,
                },
                HtNode3DComplex::Interior {
                    left: _,
                    right: _,
                    transfer: tb,
                    ranks: rkb,
                },
            ) => {
                let [ka_t, ka_l, ka_r] = *rka;
                let [kb_t, kb_l, kb_r] = *rkb;
                let kl = ka_l + kb_l;
                let kr = ka_r + kb_r;

                if idx == root_idx {
                    // Root: keep kt=1, block-diagonal in (l, r) only.
                    // B_sum[0, l, r] = B_A[0, l, r] for l<ka_l, r<ka_r
                    // B_sum[0, ka_l+l, ka_r+r] = B_B[0, l, r] for l<kb_l, r<kb_r
                    let mut transfer = vec![0.0; kl * kr];

                    for l in 0..ka_l {
                        for r in 0..ka_r {
                            transfer[l * kr + r] = ta[l * ka_r + r];
                        }
                    }
                    for l in 0..kb_l {
                        for r in 0..kb_r {
                            transfer[(ka_l + l) * kr + (ka_r + r)] = tb[l * kb_r + r];
                        }
                    }

                    nodes.push(HtNode3DComplex::Interior {
                        left: *la,
                        right: *ra,
                        transfer,
                        ranks: [1, kl, kr],
                    });
                } else {
                    // Non-root interior: full block-diagonal in (t, l, r)
                    let kt = ka_t + kb_t;
                    let mut transfer = vec![0.0; kt * kl * kr];

                    for t in 0..ka_t {
                        for l in 0..ka_l {
                            for r in 0..ka_r {
                                transfer[t * kl * kr + l * kr + r] =
                                    ta[t * ka_l * ka_r + l * ka_r + r];
                            }
                        }
                    }
                    for t in 0..kb_t {
                        for l in 0..kb_l {
                            for r in 0..kb_r {
                                transfer[(ka_t + t) * kl * kr + (ka_l + l) * kr + (ka_r + r)] =
                                    tb[t * kb_l * kb_r + l * kb_r + r];
                            }
                        }
                    }

                    nodes.push(HtNode3DComplex::Interior {
                        left: *la,
                        right: *ra,
                        transfer,
                        ranks: [kt, kl, kr],
                    });
                }
            }
            _ => {
                debug_assert!(false, "Mismatched node types at index {idx}");
                // Carry forward node from `a` unchanged as a safe fallback
                nodes.push(na.clone());
            }
        }
    }

    HtTensor3DComplex {
        nodes,
        shape: a.shape,
        dx: a.dx,
    }
}

/// Minimum-image distance for index i in [0, 2N).
fn min_image_dist(i: usize, n: usize) -> f64 {
    let half = n / 2;
    if i <= half { i as f64 } else { (n - i) as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ht_poisson_construction() {
        let shape = [8, 8, 8];
        let dx = [0.5, 0.5, 0.5];
        let solver = HtPoisson::new(shape, dx, 1e-4, 1e-4, 15);

        assert_eq!(solver.shape, shape);
        assert_eq!(solver.padded_shape, [16, 16, 16]);
        assert!(solver.r_g > 5);
    }

    #[test]
    fn ht_poisson_uniform_density() {
        let shape = [8, 8, 8];
        let dx = [0.5, 0.5, 0.5];
        let solver = HtPoisson::new(shape, dx, 1e-4, 1e-3, 20);

        let density = DensityField {
            data: vec![1.0; 512],
            shape,
        };

        let potential = solver.solve(&density, 1.0);
        assert_eq!(potential.data.len(), 512);
        assert!(
            potential.data.iter().all(|v| v.is_finite()),
            "Potential has non-finite values"
        );

        // Bowl-shaped: center more negative than corner
        let center_idx = 4 * 64 + 4 * 8 + 4;
        let corner_idx = 0;
        let center = potential.data[center_idx];
        let corner = potential.data[corner_idx];
        assert!(
            center < corner,
            "Center ({center}) should be more negative than corner ({corner})"
        );
    }

    #[test]
    fn ht_poisson_vs_tensor_poisson() {
        use crate::tooling::core::poisson::tensor_poisson::TensorPoisson;

        let shape = [8, 8, 8];
        let dx = [1.0, 1.0, 1.0];

        // Gaussian density blob
        let mut rho = vec![0.0; 512];
        for i0 in 0..8 {
            for i1 in 0..8 {
                for i2 in 0..8 {
                    let x = (i0 as f64 - 3.5) * dx[0];
                    let y = (i1 as f64 - 3.5) * dx[1];
                    let z = (i2 as f64 - 3.5) * dx[2];
                    rho[i0 * 64 + i1 * 8 + i2] = (-(x * x + y * y + z * z) / 4.0).exp();
                }
            }
        }

        let density = DensityField { data: rho, shape };

        let tensor_solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15);
        let ht_solver = HtPoisson::new(shape, dx, 1e-4, 1e-3, 20);

        let phi_tensor = tensor_solver.solve(&density, 1.0);
        let phi_ht = ht_solver.solve(&density, 1.0);

        // Compare after mean subtraction
        let n = phi_tensor.data.len() as f64;
        let mean_t: f64 = phi_tensor.data.iter().sum::<f64>() / n;
        let mean_h: f64 = phi_ht.data.iter().sum::<f64>() / n;

        let max_range = phi_tensor
            .data
            .iter()
            .map(|v| (v - mean_t).abs())
            .fold(0.0f64, f64::max);

        let max_diff: f64 = phi_tensor
            .data
            .iter()
            .zip(phi_ht.data.iter())
            .map(|(a, b)| ((a - mean_t) - (b - mean_h)).abs())
            .fold(0.0, f64::max);

        let rel_diff = max_diff / (max_range + 1e-15);
        println!(
            "HtPoisson vs TensorPoisson: rel_diff={rel_diff:.2e}, max_diff={max_diff:.2e}, max_range={max_range:.2e}"
        );
        // With near-field correction, HT should closely match Tensor Poisson.
        // At 8³ with 1e-3 HT tolerance, compression error dominates over near-field.
        assert!(
            rel_diff < 0.3,
            "HtPoisson vs TensorPoisson relative difference too large: {rel_diff}"
        );
    }

    #[test]
    fn test_near_field_magnitude_nontrivial() {
        // A peaked density (single high-value cell) should produce a non-trivial
        // near-field correction magnitude.
        let shape = [8, 8, 8];
        let dx = [1.0, 1.0, 1.0];
        let solver = HtPoisson::new(shape, dx, 1e-4, 1e-3, 20);

        let n = 512;
        let mut rho = vec![0.0; n];
        let center = 4 * 64 + 4 * 8 + 4;
        rho[center] = 100.0;

        let density = DensityField { data: rho, shape };
        let _potential = solver.solve(&density, 1.0);

        let mag = solver.last_near_field_magnitude();
        assert!(
            mag > 0.0,
            "Near-field magnitude should be > 0 for peaked density, got {mag}"
        );
    }

    #[test]
    fn test_near_field_magnitude_smooth() {
        // For a uniform density, the near-field correction magnitude should be
        // small relative to the potential itself.
        let shape = [8, 8, 8];
        let dx = [0.5, 0.5, 0.5];
        let solver = HtPoisson::new(shape, dx, 1e-4, 1e-3, 20);

        let n = 512;
        let density = DensityField {
            data: vec![1.0; n],
            shape,
        };

        let potential = solver.solve(&density, 1.0);
        let mag = solver.last_near_field_magnitude();

        // Compute the L2 norm of the potential for comparison
        let phi_l2 = potential.data.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            mag < phi_l2,
            "Near-field magnitude ({mag}) should be smaller than potential L2 norm ({phi_l2}) for uniform density"
        );
    }

    #[test]
    fn test_near_field_disabled() {
        // When near-field correction is disabled, magnitude should be zero.
        let shape = [8, 8, 8];
        let dx = [1.0, 1.0, 1.0];
        let solver = HtPoisson::new(shape, dx, 1e-4, 1e-3, 20).with_near_field_enabled(false);

        let n = 512;
        let mut rho = vec![0.0; n];
        rho[4 * 64 + 4 * 8 + 4] = 100.0;

        let density = DensityField { data: rho, shape };
        let _potential = solver.solve(&density, 1.0);

        let mag = solver.last_near_field_magnitude();
        assert!(
            mag == 0.0,
            "Near-field magnitude should be 0 when disabled, got {mag}"
        );
    }
}
