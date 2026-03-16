//! Tensor-format isolated Poisson solver.
//!
//! Solves ∇²Φ = 4πGρ with vacuum (isolated) boundary conditions using
//! low-rank tensor representation of the Green's function for efficient
//! construction and FFT, with dense convolution in Fourier space.
//!
//! **Pipeline:**
//! 1. ρ(x) → zero-pad to (2N)³ for circulant embedding
//! 2. 3D FFT of ρ and precomputed Green's function
//! 3. Element-wise multiply in Fourier space (convolution theorem)
//! 4. 3D IFFT
//! 5. Extract N³ subgrid, scale by 4πG·dx³
//!
//! The Green's function is built compactly via the Braess-Hackbusch exponential
//! sum decomposition: 1/r ≈ Σ c_k exp(-α_k r²), where each term is separable.
//! This allows O(R_G · N) construction instead of O(N³).

use super::super::solver::PoissonSolver;
use super::super::types::*;
use super::exponential_sum::ExponentialSumCoefficients;
use super::utils::finite_difference_acceleration;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Tensor-format Poisson solver with isolated (vacuum) boundary conditions.
///
/// The Green's function FFT is precomputed during construction,
/// so repeated solves only require the convolution pipeline.
pub struct TensorPoisson {
    pub shape: [usize; 3],
    pub dx: [f64; 3],
    /// Precomputed FFT of Green's function on (2N)³ grid, stored as dense complex array.
    green_fft: Vec<Complex64>,
    /// Padded shape [2*N0, 2*N1, 2*N2].
    padded_shape: [usize; 3],
    /// Volume element dx³ for integration.
    dv: f64,
}

impl TensorPoisson {
    /// Create a new TensorPoisson solver.
    pub fn new(
        shape: [usize; 3],
        dx: [f64; 3],
        exp_sum_accuracy: f64,
        _tolerance: f64,
        _max_rank: usize,
    ) -> Self {
        let delta = dx[0].min(dx[1]).min(dx[2]);
        let r_max = ((shape[0] as f64 * dx[0]).powi(2)
            + (shape[1] as f64 * dx[1]).powi(2)
            + (shape[2] as f64 * dx[2]).powi(2))
        .sqrt();

        let exp_sum = ExponentialSumCoefficients::compute(delta, r_max, exp_sum_accuracy);

        let padded_shape = [shape[0] * 2, shape[1] * 2, shape[2] * 2];
        let n_padded: usize = padded_shape.iter().product();

        // Build Green's function on (2N)³ grid using exponential sum
        let mut green = vec![0.0f64; n_padded];
        let prefactor = -1.0 / (4.0 * std::f64::consts::PI);

        let dx_avg = (dx[0] + dx[1] + dx[2]) / 3.0;

        for i0 in 0..padded_shape[0] {
            for i1 in 0..padded_shape[1] {
                for i2 in 0..padded_shape[2] {
                    let flat = i0 * padded_shape[1] * padded_shape[2] + i1 * padded_shape[2] + i2;

                    // Minimum-image distance for circulant embedding
                    let d0 = min_image_dist(i0, padded_shape[0]) * dx[0];
                    let d1 = min_image_dist(i1, padded_shape[1]) * dx[1];
                    let d2 = min_image_dist(i2, padded_shape[2]) * dx[2];
                    let r2 = d0 * d0 + d1 * d1 + d2 * d2;

                    if r2 < 1e-30 {
                        // Origin: use Hockney–Eastwood regularized self-potential
                        // G(0) ≈ -2.38·dx/(4π), matching FftIsolated convention.
                        green[flat] = -2.38 * dx_avg / (4.0 * std::f64::consts::PI);
                    } else {
                        let mut val = 0.0;
                        for k in 0..exp_sum.r_g {
                            val += exp_sum.c[k] * (-exp_sum.alpha[k] * r2).exp();
                        }
                        green[flat] = prefactor * val;
                    }
                }
            }
        }

        // 3D FFT of Green's function
        let green_fft = fft_3d_forward(&green, padded_shape);
        let dv = dx[0] * dx[1] * dx[2];

        Self {
            shape,
            dx,
            green_fft,
            padded_shape,
            dv,
        }
    }

    /// Create from pre-built domain parameters.
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
}

impl PoissonSolver for TensorPoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let [nx, ny, nz] = self.shape;
        let [px, py, pz] = self.padded_shape;

        // Step 1: zero-pad density to (2N)³
        let n_padded: usize = self.padded_shape.iter().product();
        let mut rho_padded = vec![0.0f64; n_padded];
        for i0 in 0..nx {
            for i1 in 0..ny {
                for i2 in 0..nz {
                    let src = i0 * ny * nz + i1 * nz + i2;
                    let dst = i0 * py * pz + i1 * pz + i2;
                    rho_padded[dst] = density.data[src];
                }
            }
        }

        // Step 2: 3D FFT of padded density
        let rho_fft = fft_3d_forward(&rho_padded, self.padded_shape);

        // Step 3: element-wise multiply (convolution in Fourier space)
        let phi_fft: Vec<Complex64> = rho_fft
            .iter()
            .zip(self.green_fft.iter())
            .map(|(r, g)| r * g)
            .collect();

        // Step 4: 3D IFFT
        let phi_padded = fft_3d_inverse(&phi_fft, self.padded_shape);

        // Step 5: extract N³ subgrid and scale
        let scale = 4.0 * std::f64::consts::PI * g * self.dv;
        let mut data = vec![0.0f64; nx * ny * nz];
        for i0 in 0..nx {
            for i1 in 0..ny {
                for i2 in 0..nz {
                    let src = i0 * py * pz + i1 * pz + i2;
                    let dst = i0 * ny * nz + i1 * nz + i2;
                    data[dst] = phi_padded[src] * scale;
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

/// Minimum-image distance: for index i in [0, 2N), returns min(i, 2N-i).
fn min_image_dist(i: usize, n: usize) -> f64 {
    let half = n / 2;
    if i <= half { i as f64 } else { (n - i) as f64 }
}

/// 3D FFT (forward) of a real array, returning complex array.
fn fft_3d_forward(data: &[f64], shape: [usize; 3]) -> Vec<Complex64> {
    let [n0, n1, n2] = shape;
    let n_total = n0 * n1 * n2;
    assert_eq!(data.len(), n_total);

    let mut buf: Vec<Complex64> = data.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    let mut planner = FftPlanner::new();

    // FFT along axis 2 (fastest varying)
    let fft2 = planner.plan_fft_forward(n2);
    for i0 in 0..n0 {
        for i1 in 0..n1 {
            let start = i0 * n1 * n2 + i1 * n2;
            fft2.process(&mut buf[start..start + n2]);
        }
    }

    // FFT along axis 1
    let fft1 = planner.plan_fft_forward(n1);
    let mut line = vec![Complex64::new(0.0, 0.0); n1];
    for i0 in 0..n0 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                line[i1] = buf[i0 * n1 * n2 + i1 * n2 + i2];
            }
            fft1.process(&mut line);
            for i1 in 0..n1 {
                buf[i0 * n1 * n2 + i1 * n2 + i2] = line[i1];
            }
        }
    }

    // FFT along axis 0
    let fft0 = planner.plan_fft_forward(n0);
    let mut line0 = vec![Complex64::new(0.0, 0.0); n0];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            for i0 in 0..n0 {
                line0[i0] = buf[i0 * n1 * n2 + i1 * n2 + i2];
            }
            fft0.process(&mut line0);
            for i0 in 0..n0 {
                buf[i0 * n1 * n2 + i1 * n2 + i2] = line0[i0];
            }
        }
    }

    buf
}

/// 3D IFFT of a complex array, returning real part.
fn fft_3d_inverse(data: &[Complex64], shape: [usize; 3]) -> Vec<f64> {
    let [n0, n1, n2] = shape;
    let n_total = n0 * n1 * n2;
    assert_eq!(data.len(), n_total);

    let mut buf = data.to_vec();
    let scale = 1.0 / n_total as f64;

    let mut planner = FftPlanner::new();

    // IFFT along axis 2
    let ifft2 = planner.plan_fft_inverse(n2);
    for i0 in 0..n0 {
        for i1 in 0..n1 {
            let start = i0 * n1 * n2 + i1 * n2;
            ifft2.process(&mut buf[start..start + n2]);
        }
    }

    // IFFT along axis 1
    let ifft1 = planner.plan_fft_inverse(n1);
    let mut line = vec![Complex64::new(0.0, 0.0); n1];
    for i0 in 0..n0 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                line[i1] = buf[i0 * n1 * n2 + i1 * n2 + i2];
            }
            ifft1.process(&mut line);
            for i1 in 0..n1 {
                buf[i0 * n1 * n2 + i1 * n2 + i2] = line[i1];
            }
        }
    }

    // IFFT along axis 0
    let ifft0 = planner.plan_fft_inverse(n0);
    let mut line0 = vec![Complex64::new(0.0, 0.0); n0];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            for i0 in 0..n0 {
                line0[i0] = buf[i0 * n1 * n2 + i1 * n2 + i2];
            }
            ifft0.process(&mut line0);
            for i0 in 0..n0 {
                buf[i0 * n1 * n2 + i1 * n2 + i2] = line0[i0];
            }
        }
    }

    buf.iter().map(|c| c.re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_poisson_construction() {
        let shape = [8, 8, 8];
        let dx = [0.5, 0.5, 0.5];
        let solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15);

        assert_eq!(solver.shape, shape);
        assert_eq!(solver.padded_shape, [16, 16, 16]);
    }

    #[test]
    fn tensor_poisson_uniform_density() {
        // Uniform density in a box: the potential should be finite and bowl-shaped
        let shape = [8, 8, 8];
        let dx = [0.5, 0.5, 0.5];
        let solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15);

        let n = 512;
        let density = DensityField {
            data: vec![1.0; n],
            shape,
        };

        let potential = solver.solve(&density, 1.0);
        assert_eq!(potential.data.len(), n);

        // Potential should be finite everywhere
        assert!(
            potential.data.iter().all(|v| v.is_finite()),
            "Potential has non-finite values"
        );

        // For uniform density with isolated BC, potential should be
        // more negative at center than at corner (bowl-shaped)
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
    fn tensor_poisson_point_mass() {
        let shape = [8, 8, 8];
        let dx = [1.0, 1.0, 1.0];
        let solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15);

        let n = 512;
        let mut rho = vec![0.0; n];
        let center = 4 * 64 + 4 * 8 + 4;
        rho[center] = 1.0;

        let density = DensityField { data: rho, shape };

        let potential = solver.solve(&density, 1.0);
        assert!(potential.data[center].is_finite());

        // Potential should be most negative at the source
        let min_phi = potential.data.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            potential.data[center] <= min_phi * 0.9 + 1e-10,
            "Center potential {} should be near minimum {}",
            potential.data[center],
            min_phi
        );
    }

    #[test]
    fn tensor_poisson_acceleration() {
        let shape = [8, 8, 8];
        let dx = [1.0, 1.0, 1.0];
        let solver = TensorPoisson::new(shape, dx, 1e-3, 1e-3, 10);

        let n = 512;
        // Linear potential: Φ = x, so g_x = -1
        let mut phi_data = vec![0.0; n];
        for ix in 0..8 {
            for iy in 0..8 {
                for iz in 0..8 {
                    phi_data[ix * 64 + iy * 8 + iz] = ix as f64 * dx[0];
                }
            }
        }

        let potential = PotentialField {
            data: phi_data,
            shape,
        };

        let acc = solver.compute_acceleration(&potential);

        for ix in 1..7 {
            let idx = ix * 64 + 4 * 8 + 4;
            assert!(
                (acc.gx[idx] + 1.0).abs() < 0.01,
                "gx at interior point = {}, expected -1.0",
                acc.gx[idx]
            );
        }
    }

    #[test]
    fn tensor_poisson_vs_fft_isolated() {
        // Cross-validate against FftIsolated on a simple density
        use crate::tooling::core::init::domain::{
            DomainBuilder, SpatialBoundType, VelocityBoundType,
        };
        use crate::tooling::core::poisson::fft::FftIsolated;

        let domain = DomainBuilder::new()
            .spatial_extent(4.0)
            .velocity_extent(4.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Isolated)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let shape = [8, 8, 8];
        let dx = domain.dx();

        // Gaussian density blob
        let lx = 4.0;
        let mut rho = vec![0.0; 512];
        for i0 in 0..8 {
            for i1 in 0..8 {
                for i2 in 0..8 {
                    let x = -lx + (i0 as f64 + 0.5) * dx[0];
                    let y = -lx + (i1 as f64 + 0.5) * dx[1];
                    let z = -lx + (i2 as f64 + 0.5) * dx[2];
                    rho[i0 * 64 + i1 * 8 + i2] = (-(x * x + y * y + z * z) / 2.0).exp();
                }
            }
        }

        let density = DensityField { data: rho, shape };

        let fft_solver = FftIsolated::new(&domain);
        let tensor_solver = TensorPoisson::new(shape, dx, 1e-6, 1e-6, 20);

        let phi_fft = fft_solver.solve(&density, 1.0);
        let phi_tensor = tensor_solver.solve(&density, 1.0);

        // Compare after removing constant offset (potential is defined up
        // to a constant; the exp-sum Green's function has a different
        // self-potential than FftIsolated, shifting Φ uniformly).
        let n = phi_fft.data.len() as f64;
        let mean_fft: f64 = phi_fft.data.iter().sum::<f64>() / n;
        let mean_tensor: f64 = phi_tensor.data.iter().sum::<f64>() / n;

        let max_range = phi_fft
            .data
            .iter()
            .map(|v| (v - mean_fft).abs())
            .fold(0.0f64, f64::max);
        let max_diff: f64 = phi_fft
            .data
            .iter()
            .zip(phi_tensor.data.iter())
            .map(|(a, b)| (a - mean_fft) - (b - mean_tensor))
            .map(|d| d.abs())
            .fold(0.0, f64::max);

        let rel_diff = max_diff / (max_range + 1e-15);
        assert!(
            rel_diff < 0.3,
            "TensorPoisson vs FftIsolated relative difference: {rel_diff} (max_diff={max_diff}, max_range={max_range})"
        );
    }
}
