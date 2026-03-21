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

use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::super::solver::PoissonSolver;
use super::super::types::*;
use super::exponential_sum::ExponentialSumCoefficients;
use super::utils::{finite_difference_acceleration, spectral_laplacian};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Tensor-format Poisson solver with isolated (vacuum) boundary conditions.
///
/// The Green's function FFT is precomputed during construction,
/// so repeated solves only require the convolution pipeline.
/// FFT plans are cached for reuse across `solve()` calls.
pub struct TensorPoisson {
    pub shape: [usize; 3],
    pub dx: [f64; 3],
    /// Precomputed FFT of Green's function on (2N)³ grid, stored as dense complex array.
    green_fft: Vec<Complex64>,
    /// Padded shape [2*N0, 2*N1, 2*N2].
    padded_shape: [usize; 3],
    /// Volume element dx³ for integration.
    dv: f64,
    /// Near-field correction: ∫_{B_δ} [1/|y| - GS(|y|)] dy.
    /// Applied as Φ_corr(x) = -G/(4π) * ρ(x) * near_field_integral * dx³.
    near_field_integral: f64,
    /// 2nd-order near-field correction: ∫_{B_δ} [1/|y| - GS(|y|)] r⁴/6 dy.
    /// Applied as Φ_corr_2(x) = -G · I_2 · ∇²ρ(x).
    near_field_integral_2: f64,
    /// Cached C2C forward plans for padded grid [axis0, axis1, axis2].
    fft_fwd: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Cached C2C inverse plans for padded grid [axis0, axis1, axis2].
    fft_inv: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Whether near-field correction is applied during solve.
    near_field_enabled: bool,
    /// L2 norm of last near-field correction (stored as f64 bits for atomic access).
    last_near_field_l2: AtomicU64,
    /// Shared progress state for intra-phase reporting.
    progress: Option<Arc<super::super::progress::StepProgress>>,
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

        let prefactor = -1.0 / (4.0 * std::f64::consts::PI);
        let dx_avg = (dx[0] + dx[1] + dx[2]) / 3.0;

        // Build Green's function on (2N)³ grid using exponential sum.
        // Parallelize over i0 slabs. Pre-extract exp_sum data for closure capture.
        let slab_size = padded_shape[1] * padded_shape[2];
        let n_padded: usize = padded_shape.iter().product();
        let exp_c: Vec<f64> = exp_sum.c.clone();
        let exp_alpha: Vec<f64> = exp_sum.alpha.clone();
        let exp_rg = exp_sum.r_g;

        let mut green = vec![0.0f64; n_padded];
        green
            .par_chunks_mut(slab_size)
            .enumerate()
            .for_each(|(i0, slab)| {
                let d0 = min_image_dist(i0, padded_shape[0]) * dx[0];
                for i1 in 0..padded_shape[1] {
                    let d1 = min_image_dist(i1, padded_shape[1]) * dx[1];
                    for i2 in 0..padded_shape[2] {
                        let d2 = min_image_dist(i2, padded_shape[2]) * dx[2];
                        let r2 = d0 * d0 + d1 * d1 + d2 * d2;
                        let flat = i1 * padded_shape[2] + i2;

                        if r2 < 1e-30 {
                            slab[flat] = -2.38 * dx_avg / (4.0 * std::f64::consts::PI);
                        } else {
                            let mut val = 0.0;
                            for k in 0..exp_rg {
                                val += exp_c[k] * (-exp_alpha[k] * r2).exp();
                            }
                            slab[flat] = prefactor * val;
                        }
                    }
                }
            });

        // Precompute FFT plans for the padded grid
        let mut planner = FftPlanner::new();
        let fft_fwd = [
            planner.plan_fft_forward(padded_shape[0]),
            planner.plan_fft_forward(padded_shape[1]),
            planner.plan_fft_forward(padded_shape[2]),
        ];
        let fft_inv = [
            planner.plan_fft_inverse(padded_shape[0]),
            planner.plan_fft_inverse(padded_shape[1]),
            planner.plan_fft_inverse(padded_shape[2]),
        ];

        // 3D FFT of Green's function
        let green_fft = fft_3d_forward(&green, padded_shape, &fft_fwd);
        let dv = dx[0] * dx[1] * dx[2];

        // Compute near-field correction integrals (Exl, Mauser & Zhang, JCP 2016)
        let near_field_integral = exp_sum.near_field_correction_integral(delta);
        let (_, near_field_integral_2) = exp_sum.near_field_correction_second_order(delta);

        Self {
            shape,
            dx,
            green_fft,
            padded_shape,
            dv,
            near_field_integral,
            near_field_integral_2,
            near_field_enabled: true,
            last_near_field_l2: AtomicU64::new(0u64),
            fft_fwd,
            fft_inv,
            progress: None,
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

impl PoissonSolver for TensorPoisson {
    fn set_progress(&mut self, p: std::sync::Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let [nx, ny, nz] = self.shape;
        let [px, py, pz] = self.padded_shape;

        // Step 1: zero-pad density to (2N)³ — parallel over i0 slabs
        let n_padded: usize = self.padded_shape.iter().product();
        let mut rho_padded = vec![0.0f64; n_padded];
        rho_padded
            .par_chunks_mut(py * pz)
            .enumerate()
            .for_each(|(i0, slab)| {
                if i0 < nx {
                    for i1 in 0..ny.min(py) {
                        for i2 in 0..nz.min(pz) {
                            let src = i0 * ny * nz + i1 * nz + i2;
                            slab[i1 * pz + i2] = density.data[src];
                        }
                    }
                }
            });

        // Step 2: 3D FFT of padded density
        let rho_fft = fft_3d_forward(&rho_padded, self.padded_shape, &self.fft_fwd);

        // Step 3: element-wise multiply (convolution in Fourier space) — parallel
        let total = rho_fft.len() as u64;
        let counter = AtomicU64::new(0);
        let report_interval = (total / 100).max(1);
        let phi_fft: Vec<Complex64> = rho_fft
            .par_iter()
            .zip(self.green_fft.par_iter())
            .map(|(r, g)| {
                let result = r * g;
                if let Some(ref p) = self.progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, total);
                    }
                }
                result
            })
            .collect();

        // Step 4: 3D IFFT
        let phi_padded = fft_3d_inverse(&phi_fft, self.padded_shape, &self.fft_inv);

        // Step 5: extract N³ subgrid and scale — parallel over i0 slabs
        let scale = 4.0 * std::f64::consts::PI * g * self.dv;
        let mut data = vec![0.0f64; nx * ny * nz];
        data.par_chunks_mut(ny * nz)
            .enumerate()
            .for_each(|(i0, chunk)| {
                for i1 in 0..ny {
                    for i2 in 0..nz {
                        let src = i0 * py * pz + i1 * pz + i2;
                        chunk[i1 * nz + i2] = phi_padded[src] * scale;
                    }
                }
            });

        // Step 6: Apply near-field corrections (Exl, Mauser & Zhang, JCP 2016)
        let mut corr_l2_sq = 0.0f64;

        if self.near_field_enabled {
            // 0th-order: Φ_corr(x) = -G/(4π) * ρ(x) * I_0 * dx³
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

            // 2nd-order: Φ_corr_2(x) = -G · I_2 · ∇²ρ(x)
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

/// Minimum-image distance: for index i in [0, 2N), returns min(i, 2N-i).
fn min_image_dist(i: usize, n: usize) -> f64 {
    let half = n / 2;
    if i <= half { i as f64 } else { (n - i) as f64 }
}

/// 3D FFT (forward) of a real array, returning complex array.
/// Delegates to shared scratch-buffer implementation.
fn fft_3d_forward(
    data: &[f64],
    shape: [usize; 3],
    plans: &[Arc<dyn rustfft::Fft<f64>>; 3],
) -> Vec<Complex64> {
    let n: usize = shape.iter().product();
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
    super::fft_utils::fft_3d_forward_scratch(data, &mut scratch, shape, plans)
}

/// 3D IFFT of a complex array, returning real part.
/// Delegates to shared scratch-buffer implementation.
fn fft_3d_inverse(
    data: &[Complex64],
    shape: [usize; 3],
    plans: &[Arc<dyn rustfft::Fft<f64>>; 3],
) -> Vec<f64> {
    let n: usize = shape.iter().product();
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
    super::fft_utils::fft_3d_inverse_scratch(data, &mut scratch, shape, plans)
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
    #[allow(deprecated)]
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

    #[test]
    fn test_near_field_magnitude_nontrivial() {
        // A peaked density (single high-value cell) should produce a non-trivial
        // near-field correction magnitude.
        let shape = [8, 8, 8];
        let dx = [1.0, 1.0, 1.0];
        let solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15);

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
        let solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15);

        let n = 512;
        let density = DensityField {
            data: vec![1.0; n],
            shape,
        };

        let potential = solver.solve(&density, 1.0);
        let mag = solver.last_near_field_magnitude();

        // Compute the L2 norm of the potential for comparison
        let phi_l2 = potential
            .data
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

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
        let solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15)
            .with_near_field_enabled(false);

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
