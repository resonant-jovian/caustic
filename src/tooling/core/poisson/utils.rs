//! Shared Poisson solver utilities.

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
    let n_total = nx * ny * nz;
    let mut gx = vec![0.0f64; n_total];
    let mut gy = vec![0.0f64; n_total];
    let mut gz = vec![0.0f64; n_total];

    let idx = |ix: usize, iy: usize, iz: usize| ix * ny * nz + iy * nz + iz;

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let i = idx(ix, iy, iz);

                gx[i] = if ix == 0 {
                    -(potential.data[idx(1, iy, iz)] - potential.data[idx(0, iy, iz)]) / dx[0]
                } else if ix == nx - 1 {
                    -(potential.data[idx(nx - 1, iy, iz)] - potential.data[idx(nx - 2, iy, iz)])
                        / dx[0]
                } else {
                    -(potential.data[idx(ix + 1, iy, iz)] - potential.data[idx(ix - 1, iy, iz)])
                        / (2.0 * dx[0])
                };

                gy[i] = if iy == 0 {
                    -(potential.data[idx(ix, 1, iz)] - potential.data[idx(ix, 0, iz)]) / dx[1]
                } else if iy == ny - 1 {
                    -(potential.data[idx(ix, ny - 1, iz)] - potential.data[idx(ix, ny - 2, iz)])
                        / dx[1]
                } else {
                    -(potential.data[idx(ix, iy + 1, iz)] - potential.data[idx(ix, iy - 1, iz)])
                        / (2.0 * dx[1])
                };

                gz[i] = if iz == 0 {
                    -(potential.data[idx(ix, iy, 1)] - potential.data[idx(ix, iy, 0)]) / dx[2]
                } else if iz == nz - 1 {
                    -(potential.data[idx(ix, iy, nz - 1)] - potential.data[idx(ix, iy, nz - 2)])
                        / dx[2]
                } else {
                    -(potential.data[idx(ix, iy, iz + 1)] - potential.data[idx(ix, iy, iz - 1)])
                        / (2.0 * dx[2])
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
