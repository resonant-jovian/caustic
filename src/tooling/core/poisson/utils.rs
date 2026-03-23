//! Shared Poisson solver utilities.

use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use super::super::types::*;

/// Compute gravitational acceleration g = −∇Φ via centered finite differences.
///
/// Interior points use 2nd-order central differences.
/// Boundary points use 1st-order one-sided differences.
pub fn finite_difference_acceleration(
    potential: &PotentialField,
    dx: &[f64; 3],
) -> AccelerationField {
    let [nx, ny, nz] = potential.shape;
    let slab_size = ny * nz;
    let n_total = nx * slab_size;

    let idx = |ix: usize, iy: usize, iz: usize| ix * slab_size + iy * nz + iz;

    // Pre-allocate output buffers and write directly via par_chunks_mut.
    // Avoids the previous pattern of collecting Vec-of-tuples then flattening
    // with extend_from_slice (which caused nx×3 extra allocations + memcpys).
    let mut gx = vec![0.0f64; n_total];
    let mut gy = vec![0.0f64; n_total];
    let mut gz = vec![0.0f64; n_total];

    // Three passes — one per axis. Each par_chunks_mut gives disjoint ix-slabs.
    // Potential data stays in cache across passes for typical grid sizes.
    gx.par_chunks_mut(slab_size)
        .enumerate()
        .for_each(|(ix, slab)| {
            for iy in 0..ny {
                for iz in 0..nz {
                    slab[iy * nz + iz] = if ix == 0 {
                        -(potential.data[idx(1, iy, iz)] - potential.data[idx(0, iy, iz)]) / dx[0]
                    } else if ix == nx - 1 {
                        -(potential.data[idx(nx - 1, iy, iz)]
                            - potential.data[idx(nx - 2, iy, iz)])
                            / dx[0]
                    } else {
                        -(potential.data[idx(ix + 1, iy, iz)]
                            - potential.data[idx(ix - 1, iy, iz)])
                            / (2.0 * dx[0])
                    };
                }
            }
        });

    gy.par_chunks_mut(slab_size)
        .enumerate()
        .for_each(|(ix, slab)| {
            for iy in 0..ny {
                for iz in 0..nz {
                    slab[iy * nz + iz] = if iy == 0 {
                        -(potential.data[idx(ix, 1, iz)] - potential.data[idx(ix, 0, iz)]) / dx[1]
                    } else if iy == ny - 1 {
                        -(potential.data[idx(ix, ny - 1, iz)]
                            - potential.data[idx(ix, ny - 2, iz)])
                            / dx[1]
                    } else {
                        -(potential.data[idx(ix, iy + 1, iz)]
                            - potential.data[idx(ix, iy - 1, iz)])
                            / (2.0 * dx[1])
                    };
                }
            }
        });

    gz.par_chunks_mut(slab_size)
        .enumerate()
        .for_each(|(ix, slab)| {
            for iy in 0..ny {
                for iz in 0..nz {
                    slab[iy * nz + iz] = if iz == 0 {
                        -(potential.data[idx(ix, iy, 1)] - potential.data[idx(ix, iy, 0)]) / dx[2]
                    } else if iz == nz - 1 {
                        -(potential.data[idx(ix, iy, nz - 1)]
                            - potential.data[idx(ix, iy, nz - 2)])
                            / dx[2]
                    } else {
                        -(potential.data[idx(ix, iy, iz + 1)]
                            - potential.data[idx(ix, iy, iz - 1)])
                            / (2.0 * dx[2])
                    };
                }
            }
        });

    AccelerationField {
        gx,
        gy,
        gz,
        shape: potential.shape,
    }
}

/// Compute the spectral Laplacian ∇²ρ via FFT: FFT(ρ) → multiply by −k² → IFFT.
/// Assumes periodic boundary conditions on the input density field.
pub fn spectral_laplacian(density: &DensityField, dx: &[f64; 3]) -> DensityField {
    let [nx, ny, nz] = density.shape;
    let n_total = nx * ny * nz;

    // Forward 3D FFT
    let mut planner = FftPlanner::new();
    let mut buf: Vec<Complex64> = density
        .data
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();

    // FFT along axis 2
    let fft2 = planner.plan_fft_forward(nz);
    buf.par_chunks_mut(nz).for_each(|row| {
        fft2.process(row);
    });

    // FFT along axis 1
    let fft1 = planner.plan_fft_forward(ny);
    let results_1: Vec<(usize, usize, Vec<Complex64>)> = (0..nx * nz)
        .into_par_iter()
        .map(|idx| {
            let i0 = idx / nz;
            let i2 = idx % nz;
            let mut line: Vec<Complex64> =
                (0..ny).map(|i1| buf[i0 * ny * nz + i1 * nz + i2]).collect();
            fft1.process(&mut line);
            (i0, i2, line)
        })
        .collect();
    for (i0, i2, line) in results_1 {
        for i1 in 0..ny {
            buf[i0 * ny * nz + i1 * nz + i2] = line[i1];
        }
    }

    // FFT along axis 0
    let fft0 = planner.plan_fft_forward(nx);
    let results_0: Vec<(usize, usize, Vec<Complex64>)> = (0..ny * nz)
        .into_par_iter()
        .map(|idx| {
            let i1 = idx / nz;
            let i2 = idx % nz;
            let mut line: Vec<Complex64> =
                (0..nx).map(|i0| buf[i0 * ny * nz + i1 * nz + i2]).collect();
            fft0.process(&mut line);
            (i1, i2, line)
        })
        .collect();
    for (i1, i2, line) in results_0 {
        for i0 in 0..nx {
            buf[i0 * ny * nz + i1 * nz + i2] = line[i0];
        }
    }

    // Multiply by −k²
    let lx = nx as f64 * dx[0];
    let ly = ny as f64 * dx[1];
    let lz = nz as f64 * dx[2];

    for i0 in 0..nx {
        let kx = wavenumber(i0, nx, lx);
        for i1 in 0..ny {
            let ky = wavenumber(i1, ny, ly);
            for i2 in 0..nz {
                let kz = wavenumber(i2, nz, lz);
                let k2 = kx * kx + ky * ky + kz * kz;
                buf[i0 * ny * nz + i1 * nz + i2] *= -k2;
            }
        }
    }

    // Inverse 3D FFT
    let scale = 1.0 / n_total as f64;

    let ifft2 = planner.plan_fft_inverse(nz);
    buf.par_chunks_mut(nz).for_each(|row| {
        ifft2.process(row);
    });

    let ifft1 = planner.plan_fft_inverse(ny);
    let results_1: Vec<(usize, usize, Vec<Complex64>)> = (0..nx * nz)
        .into_par_iter()
        .map(|idx| {
            let i0 = idx / nz;
            let i2 = idx % nz;
            let mut line: Vec<Complex64> =
                (0..ny).map(|i1| buf[i0 * ny * nz + i1 * nz + i2]).collect();
            ifft1.process(&mut line);
            (i0, i2, line)
        })
        .collect();
    for (i0, i2, line) in results_1 {
        for i1 in 0..ny {
            buf[i0 * ny * nz + i1 * nz + i2] = line[i1];
        }
    }

    let ifft0 = planner.plan_fft_inverse(nx);
    let results_0: Vec<(usize, usize, Vec<Complex64>)> = (0..ny * nz)
        .into_par_iter()
        .map(|idx| {
            let i1 = idx / nz;
            let i2 = idx % nz;
            let mut line: Vec<Complex64> =
                (0..nx).map(|i0| buf[i0 * ny * nz + i1 * nz + i2]).collect();
            ifft0.process(&mut line);
            (i1, i2, line)
        })
        .collect();
    for (i1, i2, line) in results_0 {
        for i0 in 0..nx {
            buf[i0 * ny * nz + i1 * nz + i2] = line[i0];
        }
    }

    let data: Vec<f64> = buf.par_iter().map(|c| c.re * scale).collect();

    DensityField {
        data,
        shape: density.shape,
    }
}

fn wavenumber(i: usize, n: usize, l: f64) -> f64 {
    use std::f64::consts::PI;
    let j = if i <= n / 2 {
        i as i64
    } else {
        i as i64 - n as i64
    };
    2.0 * PI * j as f64 / l
}
