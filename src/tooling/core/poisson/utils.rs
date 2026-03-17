//! Shared Poisson solver utilities.

use rayon::prelude::*;

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

    let idx = |ix: usize, iy: usize, iz: usize| ix * slab_size + iy * nz + iz;

    // Parallel over ix-slabs: each ix produces one slab of gx, gy, gz.
    let slabs: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..nx)
        .into_par_iter()
        .map(|ix| {
            let mut gx_slab = vec![0.0f64; slab_size];
            let mut gy_slab = vec![0.0f64; slab_size];
            let mut gz_slab = vec![0.0f64; slab_size];

            for iy in 0..ny {
                for iz in 0..nz {
                    let local = iy * nz + iz;

                    gx_slab[local] = if ix == 0 {
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

                    gy_slab[local] = if iy == 0 {
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

                    gz_slab[local] = if iz == 0 {
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

            (gx_slab, gy_slab, gz_slab)
        })
        .collect();

    // Flatten slabs into contiguous arrays
    let n_total = nx * slab_size;
    let mut gx = Vec::with_capacity(n_total);
    let mut gy = Vec::with_capacity(n_total);
    let mut gz = Vec::with_capacity(n_total);

    for (gx_slab, gy_slab, gz_slab) in slabs {
        gx.extend_from_slice(&gx_slab);
        gy.extend_from_slice(&gy_slab);
        gz.extend_from_slice(&gz_slab);
    }

    AccelerationField {
        gx,
        gy,
        gz,
        shape: potential.shape,
    }
}
