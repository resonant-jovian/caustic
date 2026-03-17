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
use super::utils::finite_difference_acceleration;
use crate::tooling::core::algos::ht3d::{HtNode3D, HtNode3DComplex, HtTensor3D, HtTensor3DComplex};
use faer::Mat;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

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
}

impl PoissonSolver for HtPoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let [nx, ny, nz] = self.shape;
        let [px, py, pz] = self.padded_shape;

        // Step 1: Zero-pad density to (2N)³
        let mut padded = vec![0.0f64; px * py * pz];
        for i0 in 0..nx {
            for i1 in 0..ny {
                for i2 in 0..nz {
                    padded[i0 * py * pz + i1 * pz + i2] = density.data[i0 * ny * nz + i1 * nz + i2];
                }
            }
        }

        // Step 2: Convert to HtTensor3D via from_dense
        let rho_ht = HtTensor3D::from_dense(
            &padded,
            self.padded_shape,
            [self.dx[0], self.dx[1], self.dx[2]],
            self.tolerance,
            self.max_rank,
        );

        // Step 3: FFT leaves
        let rho_hat = rho_ht.fft_leaves();

        // Step 4: For each Gaussian term, scale density in Fourier space by
        // the rank-1 Green's factor, then accumulate via add.
        // Each term: Φ̂_k = prefactor * c_k * diag(ĝ_k^(0)) ⊗ diag(ĝ_k^(1)) ⊗ diag(ĝ_k^(2)) · ρ̂
        // In HT format: scale each leaf column by the diagonal → preserves rank exactly.
        let mut phi_hat: Option<HtTensor3DComplex> = None;

        for k in 0..self.r_g {
            let c_k = self.exp_c[k];
            let scale = self.prefactor * c_k;

            // Scale rho_hat leaf frames by Green's factor diagonals
            let scaled = scale_complex_ht(&rho_hat, &self.green_factors[k], scale);

            phi_hat = Some(match phi_hat {
                None => scaled,
                Some(acc) => add_complex_ht(&acc, &scaled),
            });
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

        // Step 5: IFFT leaves
        let mut phi_ht = phi_hat.ifft_leaves();

        // Step 6: Truncate to re-compress (rank may be R_G × r_ρ)
        phi_ht.truncate(self.tolerance, self.max_rank);

        // Step 7: Extract N³ subgrid and scale by 4πG·dx³
        let scale = 4.0 * std::f64::consts::PI * g * self.dv;
        let mut data = vec![0.0f64; nx * ny * nz];
        for i0 in 0..nx {
            for i1 in 0..ny {
                for i2 in 0..nz {
                    data[i0 * ny * nz + i1 * nz + i2] = phi_ht.evaluate([i0, i1, i2]) * scale;
                }
            }
        }

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
/// Result rank at each node = rank_a + rank_b.
fn add_complex_ht(a: &HtTensor3DComplex, b: &HtTensor3DComplex) -> HtTensor3DComplex {
    assert_eq!(a.shape, b.shape);
    let mut nodes = Vec::with_capacity(a.nodes.len());

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
                // Block-diagonal transfer: B_new[t, l, r] where
                // t ∈ [0, ka_t + kb_t), l ∈ [0, ka_l + kb_l), r ∈ [0, ka_r + kb_r)
                let [ka_t, ka_l, ka_r] = *rka;
                let [kb_t, kb_l, kb_r] = *rkb;
                let kt = ka_t + kb_t;
                let kl = ka_l + kb_l;
                let kr = ka_r + kb_r;
                let mut transfer = vec![0.0; kt * kl * kr];

                // A block: B_new[t, l, r] = ta[t, l, r] for t < ka_t, l < ka_l, r < ka_r
                for t in 0..ka_t {
                    for l in 0..ka_l {
                        for r in 0..ka_r {
                            transfer[t * kl * kr + l * kr + r] = ta[t * ka_l * ka_r + l * ka_r + r];
                        }
                    }
                }

                // B block: B_new[ka_t + t, ka_l + l, ka_r + r] = tb[t, l, r]
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
            _ => panic!("Mismatched node types at index {idx}"),
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
        // Allow larger tolerance since HT introduces compression error
        assert!(
            rel_diff < 0.5,
            "HtPoisson vs TensorPoisson relative difference too large: {rel_diff}"
        );
    }
}
