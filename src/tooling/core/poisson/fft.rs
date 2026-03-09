//! FFT-based Poisson solvers. Periodic (FftPoisson) and isolated (FftIsolated via
//! Hockney-Eastwood zero-padding method). Both O(N³ log N).

use super::super::{init::domain::Domain, solver::PoissonSolver, types::*};
use rayon::prelude::*;
use realfft::RealFftPlanner;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

/// Periodic-BC Poisson solver. O(N³ log N). For cosmological boxes.
pub struct FftPoisson {
    pub shape: [usize; 3],
    pub dx: [f64; 3],
}

impl FftPoisson {
    pub fn new(domain: &Domain) -> Self {
        Self {
            shape: [
                domain.spatial_res.x1 as usize,
                domain.spatial_res.x2 as usize,
                domain.spatial_res.x3 as usize,
            ],
            dx: domain.dx(),
        }
    }

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
    fn fft_3d_real_forward(input: &[f64], shape: [usize; 3]) -> Vec<Complex<f64>> {
        let _span = tracing::info_span!("fft_forward").entered();
        let [nx, ny, nz] = shape;
        let nz_c = nz / 2 + 1; // Hermitian-symmetric complex output length
        let n_total_c = nx * ny * nz_c;

        // --- z-axis: R2C on contiguous rows ---
        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c = real_planner.plan_fft_forward(nz);

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
        let mut planner = FftPlanner::new();
        let fft_y = planner.plan_fft_forward(ny);

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
        let fft_x = planner.plan_fft_forward(nx);

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
    fn fft_3d_real_inverse(buf: &mut [Complex<f64>], shape: [usize; 3]) -> Vec<f64> {
        let _span = tracing::info_span!("fft_inverse").entered();
        let [nx, ny, nz] = shape;
        let nz_c = nz / 2 + 1;

        let mut planner = FftPlanner::new();

        // --- x-axis: C2C inverse ---
        let ifft_x = planner.plan_fft_inverse(nx);

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
        let ifft_y = planner.plan_fft_inverse(ny);

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

        let mut real_planner = RealFftPlanner::<f64>::new();
        let c2r = real_planner.plan_fft_inverse(nz);
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

    /// Full complex 3D FFT (needed for spectral differentiation in compute_acceleration).
    fn fft_3d(buf: &mut [Complex<f64>], shape: [usize; 3], inverse: bool) {
        let [nx, ny, nz] = shape;
        let mut planner = FftPlanner::new();

        // FFT along z (axis 2, contiguous in memory)
        let fft_z: Arc<dyn rustfft::Fft<f64>> = if inverse {
            planner.plan_fft_inverse(nz)
        } else {
            planner.plan_fft_forward(nz)
        };

        let z_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..nx * ny)
            .into_par_iter()
            .map(|row| {
                let ix = row / ny;
                let iy = row % ny;
                let start = ix * ny * nz + iy * nz;
                let mut line = buf[start..start + nz].to_vec();
                fft_z.process(&mut line);
                (ix, iy, line)
            })
            .collect();

        for (ix, iy, line) in z_results {
            let start = ix * ny * nz + iy * nz;
            buf[start..start + nz].copy_from_slice(&line);
        }

        // FFT along y (axis 1)
        let fft_y: Arc<dyn rustfft::Fft<f64>> = if inverse {
            planner.plan_fft_inverse(ny)
        } else {
            planner.plan_fft_forward(ny)
        };

        let y_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..nx * nz)
            .into_par_iter()
            .map(|idx| {
                let ix = idx / nz;
                let iz = idx % nz;
                let mut line: Vec<Complex<f64>> =
                    (0..ny).map(|iy| buf[ix * ny * nz + iy * nz + iz]).collect();
                fft_y.process(&mut line);
                (ix, iz, line)
            })
            .collect();

        for (ix, iz, line) in y_results {
            for iy in 0..ny {
                buf[ix * ny * nz + iy * nz + iz] = line[iy];
            }
        }

        // FFT along x (axis 0)
        let fft_x: Arc<dyn rustfft::Fft<f64>> = if inverse {
            planner.plan_fft_inverse(nx)
        } else {
            planner.plan_fft_forward(nx)
        };

        let x_results: Vec<(usize, usize, Vec<Complex<f64>>)> = (0..ny * nz)
            .into_par_iter()
            .map(|idx| {
                let iy = idx / nz;
                let iz = idx % nz;
                let mut line: Vec<Complex<f64>> =
                    (0..nx).map(|ix| buf[ix * ny * nz + iy * nz + iz]).collect();
                fft_x.process(&mut line);
                (iy, iz, line)
            })
            .collect();

        for (iy, iz, line) in x_results {
            for ix in 0..nx {
                buf[ix * ny * nz + iy * nz + iz] = line[ix];
            }
        }
    }
}

impl PoissonSolver for FftPoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let _span = tracing::info_span!("fft_poisson_solve").entered();
        use std::f64::consts::PI;
        let [nx, ny, nz] = self.shape;
        let nz_c = nz / 2 + 1;

        // Forward R2C FFT
        let mut rho_hat = Self::fft_3d_real_forward(&density.data, self.shape);

        // Multiply by Green's function: Φ̂ = −4πG ρ̂ / k²
        {
            let _s = tracing::info_span!("green_multiply").entered();
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz_c {
                        let idx = ix * ny * nz_c + iy * nz_c + iz;
                        let kx = Self::wavenumber(ix, nx, self.dx[0]);
                        let ky = Self::wavenumber(iy, ny, self.dx[1]);
                        // For half-complex: iz maps to wavenumber iz directly (0..nz/2+1)
                        let kz = {
                            use std::f64::consts::PI;
                            2.0 * PI * iz as f64 / (nz as f64 * self.dx[2])
                        };
                        let k2 = kx * kx + ky * ky + kz * kz;
                        if k2 == 0.0 {
                            rho_hat[idx] = Complex::new(0.0, 0.0);
                        } else {
                            rho_hat[idx] *= -4.0 * PI * g / k2;
                        }
                    }
                }
            }
        }

        // Inverse C2R FFT
        let phi_raw = Self::fft_3d_real_inverse(&mut rho_hat, self.shape);
        let n_total = (nx * ny * nz) as f64;

        PotentialField {
            data: phi_raw.iter().map(|&v| v / n_total).collect(),
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
        Self::fft_3d(&mut phi_hat, self.shape, false);

        let mut gx_hat = phi_hat.clone();
        let mut gy_hat = phi_hat.clone();
        let mut gz_hat = phi_hat.clone();
        let i_unit = Complex::new(0.0f64, 1.0);

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = ix * ny * nz + iy * nz + iz;
                    let kx = Self::wavenumber(ix, nx, self.dx[0]);
                    let ky = Self::wavenumber(iy, ny, self.dx[1]);
                    let kz = Self::wavenumber(iz, nz, self.dx[2]);
                    // g = −∇Φ → ĝ_x = −ikx Φ̂  (note: acceleration = −grad Phi)
                    gx_hat[idx] = -i_unit * kx * phi_hat[idx];
                    gy_hat[idx] = -i_unit * ky * phi_hat[idx];
                    gz_hat[idx] = -i_unit * kz * phi_hat[idx];
                }
            }
        }

        Self::fft_3d(&mut gx_hat, self.shape, true);
        Self::fft_3d(&mut gy_hat, self.shape, true);
        Self::fft_3d(&mut gz_hat, self.shape, true);

        let norm = n_total as f64;
        AccelerationField {
            gx: gx_hat.iter().map(|c| c.re / norm).collect(),
            gy: gy_hat.iter().map(|c| c.re / norm).collect(),
            gz: gz_hat.iter().map(|c| c.re / norm).collect(),
            shape: potential.shape,
        }
    }
}

/// Isolated-BC Poisson solver (Hockney-Eastwood zero-padding). Correct vacuum BC.
/// Pads density into (2N)³ box, convolves with precomputed Green's function, extracts N³ solution.
pub struct FftIsolated {
    pub shape: [usize; 3],
    pub dx: [f64; 3],
    /// Precomputed FFT of the free-space Green's function on the (2N)³ grid.
    green_hat: Vec<Complex<f64>>,
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

        // Build Green's function G(r) = -1/(4πr) on the (2N)³ grid
        // with minimum-image wrapping for correct periodicity on the padded grid
        let mut green = vec![Complex::new(0.0, 0.0); n2_total];
        for ix in 0..nx2 {
            for iy in 0..ny2 {
                for iz in 0..nz2 {
                    let idx = ix * ny2 * nz2 + iy * nz2 + iz;
                    // Minimum-image coordinates (wrapped distance on (2N)³ grid)
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
                    let rz = if iz <= nz {
                        iz as f64
                    } else {
                        iz as f64 - nz2 as f64
                    } * dx[2];
                    let r = (rx * rx + ry * ry + rz * rz).sqrt();
                    if r > 1e-30 {
                        green[idx] = Complex::new(-1.0 / (4.0 * PI * r), 0.0);
                    } else {
                        // Regularized self-potential: G(0) ≈ -2.38·dx/(4π)
                        let dx_avg = (dx[0] + dx[1] + dx[2]) / 3.0;
                        green[idx] = Complex::new(-2.38 * dx_avg / (4.0 * PI), 0.0);
                    }
                }
            }
        }

        // Forward FFT of Green's function
        FftPoisson::fft_3d(&mut green, [nx2, ny2, nz2], false);

        Self {
            shape,
            dx,
            green_hat: green,
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

        // 1. Zero-pad ρ from N³ into (2N)³
        let mut rho_pad = vec![Complex::new(0.0, 0.0); n2_total];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let src_idx = ix * ny * nz + iy * nz + iz;
                    let dst_idx = ix * ny2 * nz2 + iy * nz2 + iz;
                    rho_pad[dst_idx] = Complex::new(density.data[src_idx], 0.0);
                }
            }
        }

        // 2. Forward FFT
        FftPoisson::fft_3d(&mut rho_pad, [nx2, ny2, nz2], false);

        // 3. Pointwise multiply: Φ̂ = 4πG · Ĝ · ρ̂
        // The Green's function already contains -1/(4πr), so multiply by 4πG to get
        // the factor -G/r for the convolution, and account for the cell volume dx³.
        let dx3 = self.dx[0] * self.dx[1] * self.dx[2];
        let factor = 4.0 * PI * g * dx3;
        for (rho, green) in rho_pad.iter_mut().zip(self.green_hat.iter()) {
            *rho = factor * green * *rho;
        }

        // 4. Inverse FFT, normalize
        FftPoisson::fft_3d(&mut rho_pad, [nx2, ny2, nz2], true);
        let norm = n2_total as f64;

        // 5. Extract N³ sub-grid
        let mut phi = vec![0.0f64; nx * ny * nz];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let src_idx = ix * ny2 * nz2 + iy * nz2 + iz;
                    let dst_idx = ix * ny * nz + iy * nz + iz;
                    phi[dst_idx] = rho_pad[src_idx].re / norm;
                }
            }
        }

        PotentialField {
            data: phi,
            shape: density.shape,
        }
    }

    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        // Centered finite differences (not spectral — avoids Gibbs artifacts at
        // non-periodic boundaries).
        let [nx, ny, nz] = self.shape;
        let n_total = nx * ny * nz;
        let mut gx = vec![0.0f64; n_total];
        let mut gy = vec![0.0f64; n_total];
        let mut gz = vec![0.0f64; n_total];

        let idx = |ix: usize, iy: usize, iz: usize| ix * ny * nz + iy * nz + iz;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let i = idx(ix, iy, iz);

                    // gx = -(Φ[i+1] - Φ[i-1]) / (2·dx)
                    gx[i] = if ix == 0 {
                        -(potential.data[idx(1, iy, iz)] - potential.data[idx(0, iy, iz)])
                            / self.dx[0]
                    } else if ix == nx - 1 {
                        -(potential.data[idx(nx - 1, iy, iz)] - potential.data[idx(nx - 2, iy, iz)])
                            / self.dx[0]
                    } else {
                        -(potential.data[idx(ix + 1, iy, iz)] - potential.data[idx(ix - 1, iy, iz)])
                            / (2.0 * self.dx[0])
                    };

                    gy[i] = if iy == 0 {
                        -(potential.data[idx(ix, 1, iz)] - potential.data[idx(ix, 0, iz)])
                            / self.dx[1]
                    } else if iy == ny - 1 {
                        -(potential.data[idx(ix, ny - 1, iz)] - potential.data[idx(ix, ny - 2, iz)])
                            / self.dx[1]
                    } else {
                        -(potential.data[idx(ix, iy + 1, iz)] - potential.data[idx(ix, iy - 1, iz)])
                            / (2.0 * self.dx[1])
                    };

                    gz[i] = if iz == 0 {
                        -(potential.data[idx(ix, iy, 1)] - potential.data[idx(ix, iy, 0)])
                            / self.dx[2]
                    } else if iz == nz - 1 {
                        -(potential.data[idx(ix, iy, nz - 1)] - potential.data[idx(ix, iy, nz - 2)])
                            / self.dx[2]
                    } else {
                        -(potential.data[idx(ix, iy, iz + 1)] - potential.data[idx(ix, iy, iz - 1)])
                            / (2.0 * self.dx[2])
                    };
                }
            }
        }

        AccelerationField {
            gx,
            gy,
            gz,
            shape: potential.shape,
        }
    }
}
