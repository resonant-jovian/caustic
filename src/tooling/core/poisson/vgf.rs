//! Vico–Greengard–Ferrando (VGF) Poisson solver for isolated boundary conditions.
//!
//! Replaces the Hockney-Eastwood real-space Green's function with a truncated
//! Green's function whose Fourier transform is smooth, yielding spectral
//! convergence at the same O(N³ log N) cost.
//!
//! Reference: Vico, Greengard & Ferrando, "Fast convolution with free-space
//! Green's functions", J. Comput. Phys. 323 (2016), 191–203.

use super::super::{init::domain::Domain, solver::PoissonSolver, types::*};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Vico–Greengard–Ferrando isolated Poisson solver.
///
/// Convolves the density with a truncated Green's function whose Fourier
/// transform `G_hat(k) = -(1 - cos(kR)) / (2π k²)` is smooth, giving
/// spectral accuracy without the real-space singularity regularization
/// required by Hockney-Eastwood.
///
/// The kernel is computed analytically in Fourier space — no FFT of a
/// real-space Green's function is needed.
pub struct VgfPoisson {
    shape: [usize; 3],
    dx: [f64; 3],
    /// VGF kernel on (2N)³ — precomputed DIRECTLY in Fourier space.
    green_hat: Vec<Complex<f64>>,
    /// Cached C2C plans for the (2N)³ padded grid.
    fwd: [Arc<dyn rustfft::Fft<f64>>; 3],
    inv: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Cached scratch buffer for (2N)³ FFT transposes.
    scratch_cache: Mutex<Vec<Complex<f64>>>,
    /// Shared progress state for intra-phase reporting.
    progress: Option<Arc<super::super::progress::StepProgress>>,
}

impl VgfPoisson {
    pub fn new(domain: &Domain) -> Self {
        use std::f64::consts::PI;
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];
        let dx = domain.dx();
        let [nx, ny, nz] = shape;
        let [nx2, ny2, nz2] = [2 * nx, 2 * ny, 2 * nz];
        let n2_total = nx2 * ny2 * nz2;

        // Truncation radius: domain diagonal ensures the computational
        // domain fits inside the truncation ball.
        let lx = nx as f64 * dx[0];
        let ly = ny as f64 * dx[1];
        let lz = nz as f64 * dx[2];
        let l_max = lx.max(ly).max(lz);
        let r = 3.0_f64.sqrt() * l_max;

        // Precompute FFT plans for the padded (2N)³ grid
        let mut planner = FftPlanner::new();
        let fwd = [
            planner.plan_fft_forward(nx2),
            planner.plan_fft_forward(ny2),
            planner.plan_fft_forward(nz2),
        ];
        let inv = [
            planner.plan_fft_inverse(nx2),
            planner.plan_fft_inverse(ny2),
            planner.plan_fft_inverse(nz2),
        ];

        // Build VGF kernel analytically in Fourier space on the (2N)³ grid.
        // G_hat(k) = -(1 - cos(kR)) / (2π k²)  for k ≠ 0
        // G_hat(0) = -R² / (4π)
        //
        // Parallelized over z-rows (nz2-contiguous chunks).
        let mut green_hat = vec![Complex::new(0.0, 0.0); n2_total];
        green_hat
            .par_chunks_mut(nz2)
            .enumerate()
            .for_each(|(row, chunk)| {
                let ix = row / ny2;
                let iy = row % ny2;
                let kx = wavenumber(ix, nx2, dx[0]);
                let ky = wavenumber(iy, ny2, dx[1]);
                for (iz, c) in chunk.iter_mut().enumerate() {
                    let kz = wavenumber(iz, nz2, dx[2]);
                    let k2 = kx * kx + ky * ky + kz * kz;
                    if k2 < 1e-30 {
                        *c = Complex::new(-r * r / (4.0 * PI), 0.0);
                    } else {
                        let k = k2.sqrt();
                        *c = Complex::new(
                            -1.0 / (2.0 * PI * k2) * (1.0 - (k * r).cos()),
                            0.0,
                        );
                    }
                }
            });

        Self {
            shape,
            dx,
            green_hat,
            fwd,
            inv,
            scratch_cache: Mutex::new(Vec::new()),
            progress: None,
        }
    }
}

/// Wavenumber for index `i` on a grid of size `n` with cell spacing `cell_size`.
///
/// k_m = 2π m / (N h), where m wraps: m = i for i < N/2, m = i − N otherwise.
#[inline]
fn wavenumber(i: usize, n: usize, cell_size: f64) -> f64 {
    use std::f64::consts::PI;
    let j = if i < n / 2 {
        i as i64
    } else {
        i as i64 - n as i64
    };
    2.0 * PI * j as f64 / (n as f64 * cell_size)
}

impl PoissonSolver for VgfPoisson {
    fn set_progress(&mut self, p: Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let _span = tracing::info_span!("vgf_poisson_solve").entered();
        use std::f64::consts::PI;
        let [nx, ny, nz] = self.shape;
        let [nx2, ny2, nz2] = [2 * nx, 2 * ny, 2 * nz];
        let n2_total = nx2 * ny2 * nz2;

        // 1. Zero-pad ρ from N³ into (2N)³ (parallel over z-rows)
        let mut rho_pad = vec![Complex::new(0.0, 0.0); n2_total];
        rho_pad
            .par_chunks_mut(nz2)
            .enumerate()
            .for_each(|(row, chunk)| {
                let ix = row / ny2;
                let iy = row % ny2;
                if ix < nx && iy < ny {
                    for (iz, c) in chunk.iter_mut().take(nz).enumerate() {
                        *c = Complex::new(density.data[ix * ny * nz + iy * nz + iz], 0.0);
                    }
                }
            });

        // Use cached scratch for (2N)³ FFTs (saves 2 large allocations per solve)
        let mut scratch = std::mem::take(&mut *self.scratch_cache.lock().unwrap());
        scratch.resize(n2_total, Complex::new(0.0, 0.0));

        // 2. Forward FFT
        super::fft_utils::fft_3d_c2c_scratch(
            &mut rho_pad,
            &mut scratch,
            [nx2, ny2, nz2],
            &self.fwd,
        );

        // 3. Pointwise multiply: Φ̂ = 4πG · dx³ · Ĝ_VGF · ρ̂ (parallel)
        let dx3 = self.dx[0] * self.dx[1] * self.dx[2];
        let factor = 4.0 * PI * g * dx3;
        let total = n2_total as u64;
        let counter = AtomicU64::new(0);
        let report_interval = (total / 100).max(1);
        rho_pad
            .par_iter_mut()
            .zip(self.green_hat.par_iter())
            .for_each(|(rho, green)| {
                *rho = factor * green * *rho;
                if let Some(ref p) = self.progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, total);
                    }
                }
            });

        // 4. Inverse FFT
        super::fft_utils::fft_3d_c2c_scratch(
            &mut rho_pad,
            &mut scratch,
            [nx2, ny2, nz2],
            &self.inv,
        );

        // Return scratch to cache
        *self.scratch_cache.lock().unwrap() = scratch;
        let norm = n2_total as f64;

        // 5. Extract N³ sub-grid (parallel over z-rows)
        let mut phi = vec![0.0f64; nx * ny * nz];
        phi.par_chunks_mut(nz).enumerate().for_each(|(row, chunk)| {
            let ix = row / ny;
            let iy = row % ny;
            let base = ix * ny2 * nz2 + iy * nz2;
            for iz in 0..nz {
                chunk[iz] = rho_pad[base + iz].re / norm;
            }
        });

        PotentialField {
            data: phi,
            shape: density.shape,
        }
    }

    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        super::utils::finite_difference_acceleration(potential, &self.dx)
    }
}
