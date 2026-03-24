//! FFT-based Poisson solvers for the gravitational Poisson equation nabla^2 Phi = 4*pi*G*rho.
//!
//! Provides two solvers, both O(N^3 log N):
//! - [`FftPoisson`]: periodic boundary conditions using R2C/C2R real-to-complex FFTs
//!   with Hermitian symmetry on the z-axis. Suited for cosmological periodic boxes.
//! - [`FftIsolated`]: isolated (vacuum) boundary conditions via the Hockney-Eastwood
//!   zero-padding method. Pads the domain to (2N)^3 and convolves with a precomputed
//!   free-space Green's function. *Deprecated in favour of `VgfPoisson`.*
//!
//! All FFT plans are precomputed at construction time and cached for reuse
//! across solves, avoiding redundant plan creation on each call. Scratch buffers
//! are similarly cached behind a `Mutex` to eliminate per-call heap allocations.

use super::super::{init::domain::Domain, solver::PoissonSolver, types::*};
use rayon::prelude::*;
use realfft::RealFftPlanner;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Three-component complex slab (gx, gy, gz) for spectral differentiation.
type ComplexSlab = (Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>);

// ─── Shared 3D FFT with precomputed plans ──────────────────────────────────

/// Perform a full complex 3D FFT using precomputed plans.
///
/// Uses scratch-buffer-based transpose approach to avoid per-line allocations.
/// `buf` is a flat row-major complex buffer of size `shape[0] * shape[1] * shape[2]`.
/// `plans` contains one precomputed plan per axis: [x, y, z].
fn fft_3d_c2c(
    buf: &mut [Complex<f64>],
    shape: [usize; 3],
    plans: &[Arc<dyn rustfft::Fft<f64>>; 3],
) {
    let n: usize = shape.iter().product();
    let mut scratch = vec![Complex::new(0.0, 0.0); n];
    super::fft_utils::fft_3d_c2c_scratch(buf, &mut scratch, shape, plans);
}

/// Periodic-boundary-condition Poisson solver using 3D FFTs, O(N^3 log N).
///
/// Solves nabla^2 Phi = 4*pi*G*rho spectrally: forward-transform rho, multiply by
/// the Green's function -1/k^2, and inverse-transform to obtain Phi. The z-axis
/// exploits Hermitian symmetry via R2C/C2R transforms; x and y use full C2C.
///
/// Best suited for cosmological periodic boxes where the domain wraps on all axes.
/// FFT plans are precomputed at construction and reused across all `solve()` calls.
pub struct FftPoisson {
    /// Number of grid cells along each spatial axis [nx, ny, nz].
    pub shape: [usize; 3],
    /// Grid spacing along each spatial axis [dx, dy, dz].
    pub dx: [f64; 3],
    // Cached R2C / C2R plans for z-axis (Hermitian-symmetric)
    r2c_z: Arc<dyn realfft::RealToComplex<f64>>,
    c2r_z: Arc<dyn realfft::ComplexToReal<f64>>,
    // Cached C2C plans: used for y/x in R2C path and full-complex (compute_acceleration)
    fwd: [Arc<dyn rustfft::Fft<f64>>; 3],
    inv: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Cached scratch buffer for FFT transposes. Avoids per-call allocation.
    /// Protected by Mutex for interior mutability (PoissonSolver::solve takes &self).
    scratch_cache: std::sync::Mutex<Vec<Complex<f64>>>,
    /// Shared progress state for intra-phase reporting.
    progress: Option<Arc<super::super::progress::StepProgress>>,
}

impl FftPoisson {
    /// Create a new periodic Poisson solver, precomputing all FFT plans for the given domain.
    pub fn new(domain: &Domain) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];
        let dx = domain.dx();
        let [nx, ny, nz] = shape;

        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c_z = real_planner.plan_fft_forward(nz);
        let c2r_z = real_planner.plan_fft_inverse(nz);

        let mut planner = FftPlanner::new();
        let fwd = [
            planner.plan_fft_forward(nx),
            planner.plan_fft_forward(ny),
            planner.plan_fft_forward(nz),
        ];
        let inv = [
            planner.plan_fft_inverse(nx),
            planner.plan_fft_inverse(ny),
            planner.plan_fft_inverse(nz),
        ];

        Self {
            shape,
            dx,
            r2c_z,
            c2r_z,
            fwd,
            inv,
            scratch_cache: std::sync::Mutex::new(Vec::new()),
            progress: None,
        }
    }

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

    /// 3D FFT via sequential 1D FFTs along z, y, x axes.
    /// Uses realfft for the z-axis (contiguous, exploits Hermitian symmetry) and
    /// standard C2C FFT for y and x axes.
    fn fft_3d_real_forward(&self, input: &[f64]) -> Vec<Complex<f64>> {
        let _span = tracing::info_span!("fft_forward").entered();
        let [nx, ny, nz] = self.shape;
        let nz_c = nz / 2 + 1; // Hermitian-symmetric complex output length
        let n_total_c = nx * ny * nz_c;

        // --- z-axis: R2C on contiguous rows → write directly via par_chunks_mut ---
        let r2c = &self.r2c_z;
        let mut buf = vec![Complex::new(0.0, 0.0); n_total_c];
        buf.par_chunks_mut(nz_c)
            .enumerate()
            .for_each(|(row, out_chunk)| {
                let start = row * nz;
                let mut inbuf = vec![0.0f64; nz];
                inbuf.copy_from_slice(&input[start..start + nz]);
                if let Err(e) = r2c.process(&mut inbuf, out_chunk) {
                    tracing::error!("FFT R2C process failed: {e}");
                }
            });

        // Take cached scratch (avoids per-call allocation)
        let mut scratch =
            std::mem::take(&mut *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()));
        scratch.resize(n_total_c, Complex::new(0.0, 0.0));

        // --- y-axis: C2C via tiled transpose on half-complex grid ---
        scratch[..n_total_c]
            .par_chunks_mut(nz_c * ny)
            .enumerate()
            .for_each(|(ix, slab)| {
                let src = &buf[ix * ny * nz_c..(ix + 1) * ny * nz_c];
                super::fft_utils::transpose_tiled(src, slab, ny, nz_c);
            });
        scratch[..nx * nz_c * ny]
            .par_chunks_mut(ny)
            .for_each(|row| {
                self.fwd[1].process(row);
            });
        buf.par_chunks_mut(ny * nz_c)
            .enumerate()
            .for_each(|(ix, slab)| {
                let src = &scratch[ix * nz_c * ny..(ix + 1) * nz_c * ny];
                super::fft_utils::transpose_tiled(src, slab, nz_c, ny);
            });

        // --- x-axis: C2C via transpose on half-complex grid ---
        scratch[..n_total_c]
            .par_chunks_mut(nz_c * nx)
            .enumerate()
            .for_each(|(iy, slab)| {
                for iz in 0..nz_c {
                    for ix in 0..nx {
                        slab[iz * nx + ix] = buf[ix * ny * nz_c + iy * nz_c + iz];
                    }
                }
            });
        scratch[..ny * nz_c * nx]
            .par_chunks_mut(nx)
            .for_each(|row| {
                self.fwd[0].process(row);
            });
        buf.par_chunks_mut(ny * nz_c)
            .enumerate()
            .for_each(|(ix, slab)| {
                for iy in 0..ny {
                    for iz in 0..nz_c {
                        slab[iy * nz_c + iz] = scratch[iy * nz_c * nx + iz * nx + ix];
                    }
                }
            });

        // Return scratch to cache
        *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()) = scratch;

        buf
    }

    /// Inverse 3D FFT: C2R via x,y axes (C2C inverse) then z-axis (C2R).
    fn fft_3d_real_inverse(&self, buf: &mut [Complex<f64>]) -> Vec<f64> {
        let _span = tracing::info_span!("fft_inverse").entered();
        let [nx, ny, nz] = self.shape;
        let nz_c = nz / 2 + 1;

        // Take cached scratch (avoids per-call allocation)
        let n_total_c = nx * ny * nz_c;
        let mut scratch =
            std::mem::take(&mut *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()));
        scratch.resize(n_total_c, Complex::new(0.0, 0.0));

        // --- x-axis: C2C inverse via transpose ---
        scratch[..n_total_c]
            .par_chunks_mut(nz_c * nx)
            .enumerate()
            .for_each(|(iy, slab)| {
                for iz in 0..nz_c {
                    for ix in 0..nx {
                        slab[iz * nx + ix] = buf[ix * ny * nz_c + iy * nz_c + iz];
                    }
                }
            });
        scratch[..ny * nz_c * nx]
            .par_chunks_mut(nx)
            .for_each(|row| {
                self.inv[0].process(row);
            });
        buf.par_chunks_mut(ny * nz_c)
            .enumerate()
            .for_each(|(ix, slab)| {
                for iy in 0..ny {
                    for iz in 0..nz_c {
                        slab[iy * nz_c + iz] = scratch[iy * nz_c * nx + iz * nx + ix];
                    }
                }
            });

        // --- y-axis: C2C inverse via tiled transpose ---
        scratch[..n_total_c]
            .par_chunks_mut(nz_c * ny)
            .enumerate()
            .for_each(|(ix, slab)| {
                let src = &buf[ix * ny * nz_c..(ix + 1) * ny * nz_c];
                super::fft_utils::transpose_tiled(src, slab, ny, nz_c);
            });
        scratch[..nx * nz_c * ny]
            .par_chunks_mut(ny)
            .for_each(|row| {
                self.inv[1].process(row);
            });
        buf.par_chunks_mut(ny * nz_c)
            .enumerate()
            .for_each(|(ix, slab)| {
                let src = &scratch[ix * nz_c * ny..(ix + 1) * nz_c * ny];
                super::fft_utils::transpose_tiled(src, slab, nz_c, ny);
            });

        // --- z-axis: C2R ---
        // Enforce Hermitian symmetry: DC and Nyquist bins must be real
        // (numerical rounding from C2C passes can introduce tiny imaginary parts)
        for row in 0..nx * ny {
            let base = row * nz_c;
            buf[base].im = 0.0;
            if nz_c > 1 && nz % 2 == 0 {
                buf[base + nz_c - 1].im = 0.0;
            }
        }

        // Return scratch to cache
        *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()) = scratch;

        let c2r = &self.c2r_z;
        let n_total_real = nx * ny * nz;

        // Write directly via par_chunks_mut — eliminates nx*ny output vector allocations
        let mut output = vec![0.0f64; n_total_real];
        output
            .par_chunks_mut(nz)
            .enumerate()
            .for_each(|(row, out_chunk)| {
                let base = row * nz_c;
                let mut inbuf = buf[base..base + nz_c].to_vec();
                if let Err(e) = c2r.process(&mut inbuf, out_chunk) {
                    tracing::error!("FFT C2R process failed: {e}");
                }
            });
        output
    }
}

impl PoissonSolver for FftPoisson {
    fn set_progress(&mut self, p: std::sync::Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    /// Solve nabla^2 Phi = 4*pi*G*rho with periodic BC via spectral Green's function.
    ///
    /// Forward-transforms rho (R2C), divides by -k^2 in Fourier space (zeroing the DC
    /// mode to remove the mean), and inverse-transforms (C2R) to obtain the potential.
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let _span = tracing::info_span!("fft_poisson_solve").entered();
        use std::f64::consts::PI;
        let [nx, ny, nz] = self.shape;
        let nz_c = nz / 2 + 1;

        // Forward R2C FFT
        let mut rho_hat = self.fft_3d_real_forward(&density.data);

        // Multiply by Green's function: Φ̂ = −4πG ρ̂ / k² (parallel over rows)
        {
            let _s = tracing::info_span!("green_multiply").entered();
            let dx = self.dx;
            let total = (nx * ny) as u64;
            let counter = AtomicU64::new(0);
            let report_interval = (total / 100).max(1);
            rho_hat
                .par_chunks_mut(nz_c)
                .enumerate()
                .for_each(|(row, chunk)| {
                    let ix = row / ny;
                    let iy = row % ny;
                    let kx = Self::wavenumber(ix, nx, dx[0]);
                    let ky = Self::wavenumber(iy, ny, dx[1]);
                    for (iz, c) in chunk.iter_mut().enumerate() {
                        let kz = 2.0 * std::f64::consts::PI * iz as f64 / (nz as f64 * dx[2]);
                        let k2 = kx * kx + ky * ky + kz * kz;
                        if k2 == 0.0 {
                            *c = Complex::new(0.0, 0.0);
                        } else {
                            *c *= -4.0 * PI * g / k2;
                        }
                    }
                    if let Some(ref p) = self.progress {
                        let c = counter.fetch_add(1, Ordering::Relaxed);
                        if c.is_multiple_of(report_interval) {
                            p.set_intra_progress(c, total);
                        }
                    }
                });
        }

        // Inverse C2R FFT + normalize in-place (avoids extra allocation)
        let mut phi = self.fft_3d_real_inverse(&mut rho_hat);
        let inv_n = 1.0 / (nx * ny * nz) as f64;
        phi.par_iter_mut().for_each(|v| *v *= inv_n);

        PotentialField {
            data: phi,
            shape: density.shape,
        }
    }

    /// Compute the gravitational acceleration g = -grad(Phi) via spectral differentiation.
    ///
    /// Forward-transforms Phi, multiplies each component by -i*k_j, and inverse-transforms
    /// to obtain gx, gy, gz. All three components are computed in a single parallel pass.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        let [nx, ny, nz] = self.shape;
        let n_total = nx * ny * nz;

        let mut phi_hat: Vec<Complex<f64>> = potential
            .data
            .iter()
            .map(|&p| Complex::new(p, 0.0))
            .collect();

        // Use cached scratch for all C2C FFTs (saves 4 allocations per call)
        let mut scratch =
            std::mem::take(&mut *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()));
        scratch.resize(n_total, Complex::new(0.0, 0.0));

        super::fft_utils::fft_3d_c2c_scratch(&mut phi_hat, &mut scratch, self.shape, &self.fwd);

        // Spectral differentiation: compute gx, gy, gz in a single parallel pass
        // (avoids 3 clones of phi_hat)
        let i_unit = Complex::new(0.0f64, 1.0);
        let dx = self.dx;
        let slab_size = ny * nz;

        // Pre-allocate contiguous output arrays and write directly via par_chunks_mut,
        // eliminating per-slab Vec allocations and the serial flatten step.
        let mut gx_hat = vec![Complex::new(0.0, 0.0); n_total];
        let mut gy_hat = vec![Complex::new(0.0, 0.0); n_total];
        let mut gz_hat = vec![Complex::new(0.0, 0.0); n_total];
        let total = nx as u64;
        let counter = AtomicU64::new(0);
        let report_interval = (total / 100).max(1);

        gx_hat
            .par_chunks_mut(slab_size)
            .zip(gy_hat.par_chunks_mut(slab_size))
            .zip(gz_hat.par_chunks_mut(slab_size))
            .enumerate()
            .for_each(|(ix, ((gx_slab, gy_slab), gz_slab))| {
                let kx = Self::wavenumber(ix, nx, dx[0]);
                let base = ix * slab_size;
                for iy in 0..ny {
                    let ky = Self::wavenumber(iy, ny, dx[1]);
                    for iz in 0..nz {
                        let kz = Self::wavenumber(iz, nz, dx[2]);
                        let p = phi_hat[base + iy * nz + iz];
                        let local = iy * nz + iz;
                        // g = −∇Φ → ĝ_x = −ikx Φ̂
                        gx_slab[local] = -i_unit * kx * p;
                        gy_slab[local] = -i_unit * ky * p;
                        gz_slab[local] = -i_unit * kz * p;
                    }
                }
                if let Some(ref p) = self.progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, total);
                    }
                }
            });

        super::fft_utils::fft_3d_c2c_scratch(&mut gx_hat, &mut scratch, self.shape, &self.inv);
        super::fft_utils::fft_3d_c2c_scratch(&mut gy_hat, &mut scratch, self.shape, &self.inv);
        super::fft_utils::fft_3d_c2c_scratch(&mut gz_hat, &mut scratch, self.shape, &self.inv);

        // Return scratch to cache
        *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()) = scratch;

        let inv_norm = 1.0 / n_total as f64;
        AccelerationField {
            gx: gx_hat.par_iter().map(|c| c.re * inv_norm).collect(),
            gy: gy_hat.par_iter().map(|c| c.re * inv_norm).collect(),
            gz: gz_hat.par_iter().map(|c| c.re * inv_norm).collect(),
            shape: potential.shape,
        }
    }
}

/// Isolated-boundary-condition Poisson solver using the Hockney-Eastwood zero-padding method.
///
/// Embeds the N^3 density field into a (2N)^3 periodic box, convolves with the precomputed
/// Fourier-space free-space Green's function G(r) = -1/(4*pi*r), and extracts the N^3
/// interior to obtain the potential with correct vacuum boundary conditions. O(N^3 log N).
///
/// FFT plans and the Green's function FFT are precomputed at construction time for the
/// (2N)^3 grid. Acceleration is computed via second-order finite differences (not spectral).
///
/// **Deprecated:** Prefer [`VgfPoisson`](super::vgf::VgfPoisson) which provides
/// spectral-accuracy isolated boundary conditions with lower memory overhead.
#[deprecated(
    since = "0.0.11",
    note = "use VgfPoisson for isolated BC; FftIsolated will be removed in a future release"
)]
pub struct FftIsolated {
    /// Number of grid cells along each spatial axis [nx, ny, nz] (physical grid, not padded).
    pub shape: [usize; 3],
    /// Grid spacing along each spatial axis [dx, dy, dz].
    pub dx: [f64; 3],
    /// Precomputed FFT of the free-space Green's function on the (2N)³ grid.
    green_hat: Vec<Complex<f64>>,
    /// Cached C2C plans for the (2N)³ padded grid.
    fwd: [Arc<dyn rustfft::Fft<f64>>; 3],
    inv: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Cached scratch buffer for (2N)³ FFT transposes.
    scratch_cache: std::sync::Mutex<Vec<Complex<f64>>>,
    /// Shared progress state for intra-phase reporting.
    progress: Option<Arc<super::super::progress::StepProgress>>,
}

#[allow(deprecated)]
impl FftIsolated {
    /// Create a new isolated-BC Poisson solver, precomputing FFT plans and the Green's function.
    ///
    /// Builds the free-space Green's function G(r) = -1/(4*pi*r) on the (2N)^3 padded grid
    /// with minimum-image wrapping, then forward-transforms it for reuse in every `solve()` call.
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

        // Precompute plans for the padded grid
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

        // Build Green's function G(r) = -1/(4πr) on the (2N)³ grid
        // with minimum-image wrapping for correct periodicity on the padded grid
        // Parallelized over z-rows (nz2-contiguous chunks)
        let mut green = vec![Complex::new(0.0, 0.0); n2_total];
        green
            .par_chunks_mut(nz2)
            .enumerate()
            .for_each(|(row, chunk)| {
                let ix = row / ny2;
                let iy = row % ny2;
                let rx = if ix <= nx {
                    ix as f64
                } else {
                    ix as f64 - nx2 as f64
                } * dx[0];
                let ry = if iy <= ny {
                    iy as f64
                } else {
                    iy as f64 - ny2 as f64
                } * dx[1];
                for (iz, c) in chunk.iter_mut().enumerate() {
                    let rz = if iz <= nz {
                        iz as f64
                    } else {
                        iz as f64 - nz2 as f64
                    } * dx[2];
                    let r = (rx * rx + ry * ry + rz * rz).sqrt();
                    if r > 1e-30 {
                        *c = Complex::new(-1.0 / (4.0 * PI * r), 0.0);
                    } else {
                        // Regularized self-potential: G(0) ≈ -2.38·dx/(4π)
                        let dx_avg = (dx[0] + dx[1] + dx[2]) / 3.0;
                        *c = Complex::new(-2.38 * dx_avg / (4.0 * PI), 0.0);
                    }
                }
            });

        // Forward FFT of Green's function
        fft_3d_c2c(&mut green, [nx2, ny2, nz2], &fwd);

        Self {
            shape,
            dx,
            green_hat: green,
            fwd,
            inv,
            scratch_cache: std::sync::Mutex::new(Vec::new()),
            progress: None,
        }
    }
}

#[allow(deprecated)]
impl PoissonSolver for FftIsolated {
    fn set_progress(&mut self, p: std::sync::Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    /// Solve nabla^2 Phi = 4*pi*G*rho with isolated BC via Hockney-Eastwood convolution.
    ///
    /// Zero-pads rho into (2N)^3, forward-FFTs, multiplies pointwise with the cached
    /// Green's function, inverse-FFTs, and extracts the N^3 physical sub-grid.
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let _span = tracing::info_span!("fft_isolated_solve").entered();
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
        let mut scratch =
            std::mem::take(&mut *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()));
        scratch.resize(n2_total, Complex::new(0.0, 0.0));

        // 2. Forward FFT
        super::fft_utils::fft_3d_c2c_scratch(
            &mut rho_pad,
            &mut scratch,
            [nx2, ny2, nz2],
            &self.fwd,
        );

        // 3. Pointwise multiply: Φ̂ = 4πG · Ĝ · ρ̂ (parallel)
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
        *self.scratch_cache.lock().unwrap_or_else(|e| e.into_inner()) = scratch;
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

    /// Compute gravitational acceleration via second-order centered finite differences.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        super::utils::finite_difference_acceleration(potential, &self.dx)
    }
}
