//! FFT-based Poisson solvers. Periodic (FftPoisson) and isolated (FftIsolated via
//! James zero-padding method). Both O(N³ log N).

use super::super::{init::domain::Domain, solver::PoissonSolver, types::*};
use rustfft::{FftPlanner, num_complex::Complex};

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

    fn fft_3d(buf: &mut [Complex<f64>], shape: [usize; 3], inverse: bool) {
        let [nx, ny, nz] = shape;
        let mut planner = FftPlanner::new();

        // FFT along z (axis 2, contiguous in memory)
        let fft_z = if inverse {
            planner.plan_fft_inverse(nz)
        } else {
            planner.plan_fft_forward(nz)
        };
        for ix in 0..nx {
            for iy in 0..ny {
                let start = ix * ny * nz + iy * nz;
                fft_z.process(&mut buf[start..start + nz]);
            }
        }

        // FFT along y (axis 1)
        let fft_y = if inverse {
            planner.plan_fft_inverse(ny)
        } else {
            planner.plan_fft_forward(ny)
        };
        let mut line_y = vec![Complex::new(0.0f64, 0.0); ny];
        for ix in 0..nx {
            for iz in 0..nz {
                for iy in 0..ny {
                    line_y[iy] = buf[ix * ny * nz + iy * nz + iz];
                }
                fft_y.process(&mut line_y);
                for iy in 0..ny {
                    buf[ix * ny * nz + iy * nz + iz] = line_y[iy];
                }
            }
        }

        // FFT along x (axis 0)
        let fft_x = if inverse {
            planner.plan_fft_inverse(nx)
        } else {
            planner.plan_fft_forward(nx)
        };
        let mut line_x = vec![Complex::new(0.0f64, 0.0); nx];
        for iy in 0..ny {
            for iz in 0..nz {
                for ix in 0..nx {
                    line_x[ix] = buf[ix * ny * nz + iy * nz + iz];
                }
                fft_x.process(&mut line_x);
                for ix in 0..nx {
                    buf[ix * ny * nz + iy * nz + iz] = line_x[ix];
                }
            }
        }
    }
}

impl PoissonSolver for FftPoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        use std::f64::consts::PI;
        let [nx, ny, nz] = self.shape;
        let n_total = nx * ny * nz;

        let mut buf: Vec<Complex<f64>> =
            density.data.iter().map(|&r| Complex::new(r, 0.0)).collect();

        Self::fft_3d(&mut buf, self.shape, false);

        // Multiply by Green's function: Φ̂ = −4πG ρ̂ / k²
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = ix * ny * nz + iy * nz + iz;
                    let kx = Self::wavenumber(ix, nx, self.dx[0]);
                    let ky = Self::wavenumber(iy, ny, self.dx[1]);
                    let kz = Self::wavenumber(iz, nz, self.dx[2]);
                    let k2 = kx * kx + ky * ky + kz * kz;
                    if k2 == 0.0 {
                        buf[idx] = Complex::new(0.0, 0.0);
                    } else {
                        buf[idx] *= -4.0 * PI * g / k2;
                    }
                }
            }
        }

        Self::fft_3d(&mut buf, self.shape, true);

        let norm = n_total as f64;
        PotentialField {
            data: buf.iter().map(|c| c.re / norm).collect(),
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

/// Isolated-BC Poisson solver (James 1977 zero-padding). Correct vacuum BC.
pub struct FftIsolated {
    pub shape: [usize; 3],
    pub dx: [f64; 3],
}

impl FftIsolated {
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
}

impl PoissonSolver for FftIsolated {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        todo!("zero-pad rho into (2N)^3 box, FFT, multiply by Green's function, IFFT, extract Phi")
    }

    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        todo!("spectral diff on (2N)^3 padded domain")
    }
}
