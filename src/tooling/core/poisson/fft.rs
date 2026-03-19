//! FFT-based Poisson solvers. Periodic (FftPoisson) and isolated (FftIsolated via
//! Hockney-Eastwood zero-padding method). Both O(N³ log N).
//!
//! All FFT plans are precomputed at construction time and cached for reuse
//! across solves, avoiding redundant plan creation on each call.

use super::super::{init::domain::Domain, solver::PoissonSolver, types::*};
use rayon::prelude::*;
use realfft::RealFftPlanner;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

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

/// Periodic-BC Poisson solver. O(N³ log N). For cosmological boxes.
///
/// FFT plans are precomputed at construction and reused across all `solve()` calls.
pub struct FftPoisson {
    pub shape: [usize; 3],
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
}

impl FftPoisson {
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
                r2c.process(&mut inbuf, out_chunk).unwrap();
            });

        // Take cached scratch (avoids per-call allocation)
        let mut scratch = std::mem::take(&mut *self.scratch_cache.lock().unwrap());
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
        *self.scratch_cache.lock().unwrap() = scratch;

        buf
    }

    /// Inverse 3D FFT: C2R via x,y axes (C2C inverse) then z-axis (C2R).
    fn fft_3d_real_inverse(&self, buf: &mut [Complex<f64>]) -> Vec<f64> {
        let _span = tracing::info_span!("fft_inverse").entered();
        let [nx, ny, nz] = self.shape;
        let nz_c = nz / 2 + 1;

        // Take cached scratch (avoids per-call allocation)
        let n_total_c = nx * ny * nz_c;
        let mut scratch = std::mem::take(&mut *self.scratch_cache.lock().unwrap());
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
        *self.scratch_cache.lock().unwrap() = scratch;

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
                c2r.process(&mut inbuf, out_chunk).unwrap();
            });
        output
    }
}

impl PoissonSolver for FftPoisson {
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

    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        let [nx, ny, nz] = self.shape;
        let n_total = nx * ny * nz;

        let mut phi_hat: Vec<Complex<f64>> = potential
            .data
            .iter()
            .map(|&p| Complex::new(p, 0.0))
            .collect();

        // Use cached scratch for all C2C FFTs (saves 4 allocations per call)
        let mut scratch = std::mem::take(&mut *self.scratch_cache.lock().unwrap());
        scratch.resize(n_total, Complex::new(0.0, 0.0));

        super::fft_utils::fft_3d_c2c_scratch(&mut phi_hat, &mut scratch, self.shape, &self.fwd);

        // Spectral differentiation: compute gx, gy, gz in a single parallel pass
        // (avoids 3 clones of phi_hat)
        let i_unit = Complex::new(0.0f64, 1.0);
        let dx = self.dx;
        let slab_size = ny * nz;

        let slabs: Vec<ComplexSlab> = (0..nx)
            .into_par_iter()
            .map(|ix| {
                let kx = Self::wavenumber(ix, nx, dx[0]);
                let mut gx = Vec::with_capacity(slab_size);
                let mut gy = Vec::with_capacity(slab_size);
                let mut gz = Vec::with_capacity(slab_size);
                let base = ix * slab_size;
                for iy in 0..ny {
                    let ky = Self::wavenumber(iy, ny, dx[1]);
                    for iz in 0..nz {
                        let kz = Self::wavenumber(iz, nz, dx[2]);
                        let p = phi_hat[base + iy * nz + iz];
                        // g = −∇Φ → ĝ_x = −ikx Φ̂
                        gx.push(-i_unit * kx * p);
                        gy.push(-i_unit * ky * p);
                        gz.push(-i_unit * kz * p);
                    }
                }
                (gx, gy, gz)
            })
            .collect();

        // Flatten slabs into contiguous arrays
        let mut gx_hat = Vec::with_capacity(n_total);
        let mut gy_hat = Vec::with_capacity(n_total);
        let mut gz_hat = Vec::with_capacity(n_total);
        for (gx, gy, gz) in slabs {
            gx_hat.extend_from_slice(&gx);
            gy_hat.extend_from_slice(&gy);
            gz_hat.extend_from_slice(&gz);
        }

        super::fft_utils::fft_3d_c2c_scratch(&mut gx_hat, &mut scratch, self.shape, &self.inv);
        super::fft_utils::fft_3d_c2c_scratch(&mut gy_hat, &mut scratch, self.shape, &self.inv);
        super::fft_utils::fft_3d_c2c_scratch(&mut gz_hat, &mut scratch, self.shape, &self.inv);

        // Return scratch to cache
        *self.scratch_cache.lock().unwrap() = scratch;

        let inv_norm = 1.0 / n_total as f64;
        AccelerationField {
            gx: gx_hat.par_iter().map(|c| c.re * inv_norm).collect(),
            gy: gy_hat.par_iter().map(|c| c.re * inv_norm).collect(),
            gz: gz_hat.par_iter().map(|c| c.re * inv_norm).collect(),
            shape: potential.shape,
        }
    }
}

/// Isolated-BC Poisson solver (Hockney-Eastwood zero-padding). Correct vacuum BC.
/// Pads density into (2N)³ box, convolves with precomputed Green's function, extracts N³ solution.
///
/// FFT plans are precomputed for the (2N)³ grid at construction time.
pub struct FftIsolated {
    pub shape: [usize; 3],
    pub dx: [f64; 3],
    /// Precomputed FFT of the free-space Green's function on the (2N)³ grid.
    green_hat: Vec<Complex<f64>>,
    /// Cached C2C plans for the (2N)³ padded grid.
    fwd: [Arc<dyn rustfft::Fft<f64>>; 3],
    inv: [Arc<dyn rustfft::Fft<f64>>; 3],
    /// Cached scratch buffer for (2N)³ FFT transposes.
    scratch_cache: std::sync::Mutex<Vec<Complex<f64>>>,
}

impl FftIsolated {
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
        }
    }
}

impl PoissonSolver for FftIsolated {
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
        let mut scratch = std::mem::take(&mut *self.scratch_cache.lock().unwrap());
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
        rho_pad
            .par_iter_mut()
            .zip(self.green_hat.par_iter())
            .for_each(|(rho, green)| {
                *rho = factor * green * *rho;
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
