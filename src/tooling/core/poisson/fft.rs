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
/// `buf` is a flat row-major complex buffer of size `shape[0] * shape[1] * shape[2]`.
/// `plans` contains one precomputed plan per axis: [x, y, z].
fn fft_3d_c2c(
    buf: &mut [Complex<f64>],
    shape: [usize; 3],
    plans: &[Arc<dyn rustfft::Fft<f64>>; 3],
) {
    let [nx, ny, nz] = shape;

    // Axis 2 (z): contiguous rows — in-place via par_chunks_mut
    buf.par_chunks_mut(nz).for_each(|row| {
        plans[2].process(row);
    });

    // Axis 1 (y): stride-nz access — gather, transform, scatter
    let y_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..nx * nz)
        .into_par_iter()
        .map(|idx| {
            let ix = idx / nz;
            let iz = idx % nz;
            let mut line: Vec<Complex<f64>> =
                (0..ny).map(|iy| buf[ix * ny * nz + iy * nz + iz]).collect();
            plans[1].process(&mut line);
            (ix, iz, line)
        })
        .collect();
    for (ix, iz, line) in y_results {
        for iy in 0..ny {
            buf[ix * ny * nz + iy * nz + iz] = line[iy];
        }
    }

    // Axis 0 (x): stride-ny*nz access — gather, transform, scatter
    let x_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..ny * nz)
        .into_par_iter()
        .map(|idx| {
            let iy = idx / nz;
            let iz = idx % nz;
            let mut line: Vec<Complex<f64>> =
                (0..nx).map(|ix| buf[ix * ny * nz + iy * nz + iz]).collect();
            plans[0].process(&mut line);
            (iy, iz, line)
        })
        .collect();
    for (iy, iz, line) in x_results {
        for ix in 0..nx {
            buf[ix * ny * nz + iy * nz + iz] = line[ix];
        }
    }
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

        // --- z-axis: R2C on contiguous rows ---
        let r2c = &self.r2c_z;

        // Process z-lines in parallel
        let z_results: Vec<(usize, Vec<Complex<f64>>)> = (0..nx * ny)
            .into_par_iter()
            .map(|row| {
                let ix = row / ny;
                let iy = row % ny;
                let start = ix * ny * nz + iy * nz;
                let mut inbuf: Vec<f64> = input[start..start + nz].to_vec();
                let mut outbuf = vec![Complex::new(0.0, 0.0); nz_c];
                r2c.process(&mut inbuf, &mut outbuf).unwrap();
                (row, outbuf)
            })
            .collect();

        // Assemble into half-complex 3D buffer [nx, ny, nz_c]
        let mut buf = vec![Complex::new(0.0, 0.0); n_total_c];
        for (row, data) in z_results {
            let ix = row / ny;
            let iy = row % ny;
            let base = ix * ny * nz_c + iy * nz_c;
            buf[base..base + nz_c].copy_from_slice(&data);
        }

        // --- y-axis: C2C on half-complex grid ---
        let fft_y = &self.fwd[1];

        let y_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..nx * nz_c)
            .into_par_iter()
            .map(|idx| {
                let ix = idx / nz_c;
                let iz = idx % nz_c;
                let mut line: Vec<Complex<f64>> = (0..ny)
                    .map(|iy| buf[ix * ny * nz_c + iy * nz_c + iz])
                    .collect();
                fft_y.process(&mut line);
                (ix, iz, line)
            })
            .collect();

        for (ix, iz, line) in y_results {
            for iy in 0..ny {
                buf[ix * ny * nz_c + iy * nz_c + iz] = line[iy];
            }
        }

        // --- x-axis: C2C on half-complex grid ---
        let fft_x = &self.fwd[0];

        let x_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..ny * nz_c)
            .into_par_iter()
            .map(|idx| {
                let iy = idx / nz_c;
                let iz = idx % nz_c;
                let mut line: Vec<Complex<f64>> = (0..nx)
                    .map(|ix| buf[ix * ny * nz_c + iy * nz_c + iz])
                    .collect();
                fft_x.process(&mut line);
                (iy, iz, line)
            })
            .collect();

        for (iy, iz, line) in x_results {
            for ix in 0..nx {
                buf[ix * ny * nz_c + iy * nz_c + iz] = line[ix];
            }
        }

        buf
    }

    /// Inverse 3D FFT: C2R via x,y axes (C2C inverse) then z-axis (C2R).
    fn fft_3d_real_inverse(&self, buf: &mut [Complex<f64>]) -> Vec<f64> {
        let _span = tracing::info_span!("fft_inverse").entered();
        let [nx, ny, nz] = self.shape;
        let nz_c = nz / 2 + 1;

        // --- x-axis: C2C inverse ---
        let ifft_x = &self.inv[0];

        let x_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..ny * nz_c)
            .into_par_iter()
            .map(|idx| {
                let iy = idx / nz_c;
                let iz = idx % nz_c;
                let mut line: Vec<Complex<f64>> = (0..nx)
                    .map(|ix| buf[ix * ny * nz_c + iy * nz_c + iz])
                    .collect();
                ifft_x.process(&mut line);
                (iy, iz, line)
            })
            .collect();

        for (iy, iz, line) in x_results {
            for ix in 0..nx {
                buf[ix * ny * nz_c + iy * nz_c + iz] = line[ix];
            }
        }

        // --- y-axis: C2C inverse ---
        let ifft_y = &self.inv[1];

        let y_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..nx * nz_c)
            .into_par_iter()
            .map(|idx| {
                let ix = idx / nz_c;
                let iz = idx % nz_c;
                let mut line: Vec<Complex<f64>> = (0..ny)
                    .map(|iy| buf[ix * ny * nz_c + iy * nz_c + iz])
                    .collect();
                ifft_y.process(&mut line);
                (ix, iz, line)
            })
            .collect();

        for (ix, iz, line) in y_results {
            for iy in 0..ny {
                buf[ix * ny * nz_c + iy * nz_c + iz] = line[iy];
            }
        }

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

        let c2r = &self.c2r_z;
        let n_total_real = nx * ny * nz;

        let z_results: Vec<(usize, Vec<f64>)> = (0..nx * ny)
            .into_par_iter()
            .map(|row| {
                let ix = row / ny;
                let iy = row % ny;
                let base = ix * ny * nz_c + iy * nz_c;
                let mut inbuf = buf[base..base + nz_c].to_vec();
                let mut outbuf = vec![0.0f64; nz];
                c2r.process(&mut inbuf, &mut outbuf).unwrap();
                (row, outbuf)
            })
            .collect();

        let mut output = vec![0.0f64; n_total_real];
        for (row, data) in z_results {
            let ix = row / ny;
            let iy = row % ny;
            let start = ix * ny * nz + iy * nz;
            output[start..start + nz].copy_from_slice(&data);
        }
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
        fft_3d_c2c(&mut phi_hat, self.shape, &self.fwd);

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

        fft_3d_c2c(&mut gx_hat, self.shape, &self.inv);
        fft_3d_c2c(&mut gy_hat, self.shape, &self.inv);
        fft_3d_c2c(&mut gz_hat, self.shape, &self.inv);

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

        // 2. Forward FFT
        fft_3d_c2c(&mut rho_pad, [nx2, ny2, nz2], &self.fwd);

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
        fft_3d_c2c(&mut rho_pad, [nx2, ny2, nz2], &self.inv);
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
