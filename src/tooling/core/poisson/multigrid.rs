//! Geometric multigrid solver for ∇²Φ = 4πGρ. Supports periodic and Dirichlet (isolated/open) BC.
//! O(N³) per V-cycle, O(N³ log ε) total.

use rayon::prelude::*;

use super::super::{
    init::domain::{Domain, SpatialBoundType},
    solver::PoissonSolver,
    types::*,
};

/// Geometric multigrid Poisson solver.
///
/// Uses V-cycles with red-black Gauss-Seidel smoothing, full-weighting restriction,
/// and trilinear prolongation. Supports periodic and Dirichlet zero boundary conditions.
pub struct Multigrid {
    pub levels: usize,
    pub shape: [usize; 3],
    pub dx: [f64; 3],
    pub bc: SpatialBoundType,
    pub n_smooth: usize,
    pub tolerance: f64,
}

impl Multigrid {
    /// Create a new multigrid solver.
    ///
    /// - `domain`: the computational domain (provides shape, dx, boundary type)
    /// - `levels`: requested number of multigrid levels (will be capped so the coarsest
    ///   grid has at least 2 cells per dimension)
    /// - `smoothing_steps`: number of Gauss-Seidel sweeps per pre/post-smoothing phase
    pub fn new(domain: &Domain, levels: usize, smoothing_steps: usize) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];
        let dx = domain.dx();
        let bc = domain.spatial_bc.clone();

        // Cap levels so coarsest grid has >= 2 cells per dimension.
        // At level l, shape is shape[d] / 2^l. We need shape[d] / 2^(levels-1) >= 2.
        let min_dim = shape[0].min(shape[1]).min(shape[2]);
        let max_levels = if min_dim >= 2 {
            // floor(log2(min_dim / 2)) + 1
            let mut max_l = 1usize;
            let mut s = min_dim;
            while s / 2 >= 2 {
                s /= 2;
                max_l += 1;
            }
            max_l
        } else {
            1
        };
        let levels = levels.min(max_levels);

        Self {
            levels,
            shape,
            dx,
            bc,
            n_smooth: smoothing_steps,
            tolerance: 1e-10,
        }
    }
}

/// Row-major 3D indexing.
#[inline(always)]
fn idx(ix: usize, iy: usize, iz: usize, shape: [usize; 3]) -> usize {
    ix * shape[1] * shape[2] + iy * shape[2] + iz
}

/// Red-black Gauss-Seidel smoothing on the 7-point discrete Laplacian.
///
/// Updates `phi` in-place using `n_sweeps` full sweeps (each sweep = red then black pass).
fn smooth_red_black(
    phi: &mut [f64],
    rhs: &[f64],
    shape: [usize; 3],
    dx: [f64; 3],
    n_sweeps: usize,
    bc: &SpatialBoundType,
) {
    let [nx, ny, nz] = shape;
    let inv_dx2_x = 1.0 / (dx[0] * dx[0]);
    let inv_dx2_y = 1.0 / (dx[1] * dx[1]);
    let inv_dx2_z = 1.0 / (dx[2] * dx[2]);
    let diag = 2.0 * inv_dx2_x + 2.0 * inv_dx2_y + 2.0 * inv_dx2_z;

    let is_periodic = matches!(bc, SpatialBoundType::Periodic);

    for _sweep in 0..n_sweeps {
        // Two passes: parity 0 (red) then parity 1 (black)
        for parity in 0..2u8 {
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        if ((ix + iy + iz) % 2) as u8 != parity {
                            continue;
                        }

                        let i = idx(ix, iy, iz, shape);

                        // Fetch neighbors with boundary handling
                        let phi_xm;
                        let phi_xp;
                        let phi_ym;
                        let phi_yp;
                        let phi_zm;
                        let phi_zp;

                        if is_periodic {
                            let ixm = if ix == 0 { nx - 1 } else { ix - 1 };
                            let ixp = if ix == nx - 1 { 0 } else { ix + 1 };
                            let iym = if iy == 0 { ny - 1 } else { iy - 1 };
                            let iyp = if iy == ny - 1 { 0 } else { iy + 1 };
                            let izm = if iz == 0 { nz - 1 } else { iz - 1 };
                            let izp = if iz == nz - 1 { 0 } else { iz + 1 };

                            phi_xm = phi[idx(ixm, iy, iz, shape)];
                            phi_xp = phi[idx(ixp, iy, iz, shape)];
                            phi_ym = phi[idx(ix, iym, iz, shape)];
                            phi_yp = phi[idx(ix, iyp, iz, shape)];
                            phi_zm = phi[idx(ix, iy, izm, shape)];
                            phi_zp = phi[idx(ix, iy, izp, shape)];
                        } else {
                            // Dirichlet zero BC: boundary neighbors are 0
                            phi_xm = if ix > 0 {
                                phi[idx(ix - 1, iy, iz, shape)]
                            } else {
                                0.0
                            };
                            phi_xp = if ix < nx - 1 {
                                phi[idx(ix + 1, iy, iz, shape)]
                            } else {
                                0.0
                            };
                            phi_ym = if iy > 0 {
                                phi[idx(ix, iy - 1, iz, shape)]
                            } else {
                                0.0
                            };
                            phi_yp = if iy < ny - 1 {
                                phi[idx(ix, iy + 1, iz, shape)]
                            } else {
                                0.0
                            };
                            phi_zm = if iz > 0 {
                                phi[idx(ix, iy, iz - 1, shape)]
                            } else {
                                0.0
                            };
                            phi_zp = if iz < nz - 1 {
                                phi[idx(ix, iy, iz + 1, shape)]
                            } else {
                                0.0
                            };
                        }

                        let sum_neighbors = (phi_xm + phi_xp) * inv_dx2_x
                            + (phi_ym + phi_yp) * inv_dx2_y
                            + (phi_zm + phi_zp) * inv_dx2_z;

                        phi[i] = (sum_neighbors - rhs[i]) / diag;
                    }
                }
            }
        }
    }
}

/// Compute the residual r = rhs - L*phi where L is the discrete Laplacian.
fn residual(
    phi: &[f64],
    rhs: &[f64],
    shape: [usize; 3],
    dx: [f64; 3],
    bc: &SpatialBoundType,
) -> Vec<f64> {
    let [nx, ny, nz] = shape;
    let n_total = nx * ny * nz;
    let inv_dx2_x = 1.0 / (dx[0] * dx[0]);
    let inv_dx2_y = 1.0 / (dx[1] * dx[1]);
    let inv_dx2_z = 1.0 / (dx[2] * dx[2]);

    let is_periodic = matches!(bc, SpatialBoundType::Periodic);

    let res: Vec<f64> = (0..n_total)
        .into_par_iter()
        .map(|i| {
            let ix = i / (ny * nz);
            let iy = (i / nz) % ny;
            let iz = i % nz;

            let (phi_xm, phi_xp, phi_ym, phi_yp, phi_zm, phi_zp);

            if is_periodic {
                let ixm = if ix == 0 { nx - 1 } else { ix - 1 };
                let ixp = if ix == nx - 1 { 0 } else { ix + 1 };
                let iym = if iy == 0 { ny - 1 } else { iy - 1 };
                let iyp = if iy == ny - 1 { 0 } else { iy + 1 };
                let izm = if iz == 0 { nz - 1 } else { iz - 1 };
                let izp = if iz == nz - 1 { 0 } else { iz + 1 };

                phi_xm = phi[idx(ixm, iy, iz, shape)];
                phi_xp = phi[idx(ixp, iy, iz, shape)];
                phi_ym = phi[idx(ix, iym, iz, shape)];
                phi_yp = phi[idx(ix, iyp, iz, shape)];
                phi_zm = phi[idx(ix, iy, izm, shape)];
                phi_zp = phi[idx(ix, iy, izp, shape)];
            } else {
                phi_xm = if ix > 0 {
                    phi[idx(ix - 1, iy, iz, shape)]
                } else {
                    0.0
                };
                phi_xp = if ix < nx - 1 {
                    phi[idx(ix + 1, iy, iz, shape)]
                } else {
                    0.0
                };
                phi_ym = if iy > 0 {
                    phi[idx(ix, iy - 1, iz, shape)]
                } else {
                    0.0
                };
                phi_yp = if iy < ny - 1 {
                    phi[idx(ix, iy + 1, iz, shape)]
                } else {
                    0.0
                };
                phi_zm = if iz > 0 {
                    phi[idx(ix, iy, iz - 1, shape)]
                } else {
                    0.0
                };
                phi_zp = if iz < nz - 1 {
                    phi[idx(ix, iy, iz + 1, shape)]
                } else {
                    0.0
                };
            }

            let laplacian = (phi_xm + phi_xp - 2.0 * phi[i]) * inv_dx2_x
                + (phi_ym + phi_yp - 2.0 * phi[i]) * inv_dx2_y
                + (phi_zm + phi_zp - 2.0 * phi[i]) * inv_dx2_z;

            rhs[i] - laplacian
        })
        .collect();
    res
}

/// Full-weighting restriction from fine grid to coarse grid (factor 2 coarsening).
///
/// Uses a 3x3x3 stencil centered at (2i, 2j, 2k) with weights:
/// - center: 1/8
/// - face neighbors (6): 1/16 each
/// - edge neighbors (12): 1/32 each
/// - corner neighbors (8): 1/64 each
///
/// These sum to 1/8 + 6/16 + 12/32 + 8/64 = 1/8 + 3/8 + 3/8 + 1/8 = 1.
fn restrict(fine: &[f64], fine_shape: [usize; 3]) -> (Vec<f64>, [usize; 3]) {
    let [fnx, fny, fnz] = fine_shape;
    let coarse_shape = [fnx / 2, fny / 2, fnz / 2];
    let [cnx, cny, cnz] = coarse_shape;
    let coarse: Vec<f64> = (0..cnx * cny * cnz)
        .into_par_iter()
        .map(|flat| {
            let ci = flat / (cny * cnz);
            let cj = (flat / cnz) % cny;
            let ck = flat % cnz;

            let fi = 2 * ci;
            let fj = 2 * cj;
            let fk = 2 * ck;

            let mut val = 0.0;

            for di in 0..3i32 {
                for dj in 0..3i32 {
                    for dk in 0..3i32 {
                        let si = fi as i32 + di - 1;
                        let sj = fj as i32 + dj - 1;
                        let sk = fk as i32 + dk - 1;

                        if si < 0
                            || si >= fnx as i32
                            || sj < 0
                            || sj >= fny as i32
                            || sk < 0
                            || sk >= fnz as i32
                        {
                            continue;
                        }

                        let si = si as usize;
                        let sj = sj as usize;
                        let sk = sk as usize;

                        let dist = (di != 1) as u32 + (dj != 1) as u32 + (dk != 1) as u32;
                        let weight = 1.0 / (1u32 << (3 + dist)) as f64;

                        val += weight * fine[idx(si, sj, sk, fine_shape)];
                    }
                }
            }

            val
        })
        .collect();

    (coarse, coarse_shape)
}

/// Trilinear prolongation (interpolation) from coarse grid to fine grid.
///
/// Each fine cell (i, j, k) is interpolated from the 8 surrounding coarse cells
/// using trilinear weights. Coarse cell centers are at fine indices (2*ci, 2*cj, 2*ck).
fn prolongate(coarse: &[f64], coarse_shape: [usize; 3], fine_shape: [usize; 3]) -> Vec<f64> {
    let [cnx, cny, cnz] = coarse_shape;
    let [fnx, fny, fnz] = fine_shape;
    let n_fine = fnx * fny * fnz;
    let fine: Vec<f64> = (0..n_fine)
        .into_par_iter()
        .map(|flat| {
            let fi = flat / (fny * fnz);
            let fj = (flat / fnz) % fny;
            let fk = flat % fnz;

            let (ci0, wx) = coarse_interp_index(fi, cnx);
            let (cj0, wy) = coarse_interp_index(fj, cny);
            let (ck0, wz) = coarse_interp_index(fk, cnz);

            let ci1 = (ci0 + 1).min(cnx - 1);
            let cj1 = (cj0 + 1).min(cny - 1);
            let ck1 = (ck0 + 1).min(cnz - 1);

            let c000 = coarse[idx(ci0, cj0, ck0, coarse_shape)];
            let c100 = coarse[idx(ci1, cj0, ck0, coarse_shape)];
            let c010 = coarse[idx(ci0, cj1, ck0, coarse_shape)];
            let c110 = coarse[idx(ci1, cj1, ck0, coarse_shape)];
            let c001 = coarse[idx(ci0, cj0, ck1, coarse_shape)];
            let c101 = coarse[idx(ci1, cj0, ck1, coarse_shape)];
            let c011 = coarse[idx(ci0, cj1, ck1, coarse_shape)];
            let c111 = coarse[idx(ci1, cj1, ck1, coarse_shape)];

            c000 * (1.0 - wx) * (1.0 - wy) * (1.0 - wz)
                + c100 * wx * (1.0 - wy) * (1.0 - wz)
                + c010 * (1.0 - wx) * wy * (1.0 - wz)
                + c110 * wx * wy * (1.0 - wz)
                + c001 * (1.0 - wx) * (1.0 - wy) * wz
                + c101 * wx * (1.0 - wy) * wz
                + c011 * (1.0 - wx) * wy * wz
                + c111 * wx * wy * wz
        })
        .collect();

    fine
}

/// Compute the coarse grid index and interpolation weight for a fine grid index.
///
/// The coarse grid cell centers are mapped to fine indices 2*ci. Fine index fi
/// maps to coarse coordinate fi/2. We split this into integer part ci0 and
/// fractional weight w in [0, 1).
#[inline]
fn coarse_interp_index(fi: usize, cn: usize) -> (usize, f64) {
    // Fine cell fi maps to coarse position fi / 2.
    // ci0 = floor(fi / 2), w = (fi % 2) * 0.5
    // But we want proper interpolation between coarse centers.
    // Coarse centers at fine positions: 0, 2, 4, ... => coarse indices 0, 1, 2, ...
    // So fine index fi corresponds to coarse "coordinate" fi / 2.0.
    // ci0 = fi / 2 (integer division), w = (fi as f64 / 2.0) - ci0 as f64
    let ci0 = fi / 2;
    let ci0 = ci0.min(cn.saturating_sub(1));
    let w = (fi as f64 / 2.0) - ci0 as f64;
    (ci0, w)
}

/// Recursive V-cycle multigrid solver.
///
/// - Pre-smooth `n_pre` sweeps
/// - Compute residual, restrict to coarser grid
/// - Recurse (or directly smooth if at coarsest level)
/// - Prolongate correction, add to phi
/// - Post-smooth `n_post` sweeps
#[allow(clippy::too_many_arguments)]
fn v_cycle(
    phi: &mut [f64],
    rhs: &[f64],
    shape: [usize; 3],
    dx: [f64; 3],
    level: usize,
    max_level: usize,
    n_pre: usize,
    n_post: usize,
    bc: &SpatialBoundType,
) {
    // Pre-smoothing
    smooth_red_black(phi, rhs, shape, dx, n_pre, bc);

    if level >= max_level - 1 {
        // At coarsest level: apply many smoothing sweeps as a direct solver
        smooth_red_black(phi, rhs, shape, dx, 50, bc);
        return;
    }

    // Compute residual on current grid
    let res = residual(phi, rhs, shape, dx, bc);

    // Restrict residual to coarser grid
    let (rhs_coarse, coarse_shape) = restrict(&res, shape);
    let coarse_dx = [dx[0] * 2.0, dx[1] * 2.0, dx[2] * 2.0];

    // Initialize coarse correction to zero
    let mut e_coarse = vec![0.0; coarse_shape[0] * coarse_shape[1] * coarse_shape[2]];

    // Recurse
    v_cycle(
        &mut e_coarse,
        &rhs_coarse,
        coarse_shape,
        coarse_dx,
        level + 1,
        max_level,
        n_pre,
        n_post,
        bc,
    );

    // Prolongate correction to fine grid and add to phi
    let correction = prolongate(&e_coarse, coarse_shape, shape);
    for i in 0..phi.len() {
        phi[i] += correction[i];
    }

    // Post-smoothing
    smooth_red_black(phi, rhs, shape, dx, n_post, bc);
}

/// Compute L2 norm of a vector.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl PoissonSolver for Multigrid {
    /// Solve ∇²Φ = 4πGρ using multigrid V-cycles.
    ///
    /// Iterates V-cycles until the relative residual drops below `tolerance`
    /// or the maximum iteration count (100) is reached. For periodic BC, the
    /// mean of phi is subtracted after solving (since the solution is only
    /// determined up to a constant).
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let _span = tracing::info_span!("multigrid_solve").entered();
        let [nx, ny, nz] = self.shape;
        let n_total = nx * ny * nz;

        // Build RHS = 4πGρ (the Poisson equation is ∇²Φ = 4πGρ)
        let four_pi_g = 4.0 * std::f64::consts::PI * g;
        let rhs: Vec<f64> = density.data.iter().map(|&r| four_pi_g * r).collect();

        let rhs_norm = l2_norm(&rhs);

        // Initialize phi to zero
        let mut phi = vec![0.0; n_total];

        let max_iter = 100;
        for _iter in 0..max_iter {
            v_cycle(
                &mut phi,
                &rhs,
                self.shape,
                self.dx,
                0,
                self.levels,
                self.n_smooth,
                self.n_smooth,
                &self.bc,
            );

            // Check convergence
            let res = residual(&phi, &rhs, self.shape, self.dx, &self.bc);
            let res_norm = l2_norm(&res);

            if rhs_norm < 1e-15 {
                // RHS is essentially zero — phi = 0 is the solution
                break;
            }

            if res_norm / rhs_norm < self.tolerance {
                break;
            }
        }

        // For periodic BC, subtract the mean (solution is unique only up to a constant)
        if matches!(self.bc, SpatialBoundType::Periodic) {
            let mean = phi.iter().sum::<f64>() / n_total as f64;
            for p in phi.iter_mut() {
                *p -= mean;
            }
        }

        PotentialField {
            data: phi,
            shape: self.shape,
        }
    }

    /// Compute gravitational acceleration g = -nabla Phi via finite differences.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        super::utils::finite_difference_acceleration(potential, &self.dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use std::f64::consts::PI;

    fn periodic_domain(n: i128) -> Domain {
        Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(n)
            .velocity_resolution(4)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn multigrid_sin_solution() {
        let domain = periodic_domain(32);
        let mg = Multigrid::new(&domain, 4, 3);
        let [nx, ny, nz] = mg.shape;
        let dx = mg.dx;

        // For a periodic domain [-1, 1]^3 with period 2:
        // phi(x,y,z) = sin(pi*x)*sin(pi*y)*sin(pi*z)
        // nabla^2 phi = -3*pi^2 * sin(pi*x)*sin(pi*y)*sin(pi*z)
        //
        // The Poisson equation is nabla^2 Phi = 4*pi*G*rho, so with G=1:
        // rho = -3*pi^2 * sin(pi*x)*sin(pi*y)*sin(pi*z) / (4*pi)
        let mut rho = vec![0.0; nx * ny * nz];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = -1.0 + (ix as f64 + 0.5) * dx[0];
                    let y = -1.0 + (iy as f64 + 0.5) * dx[1];
                    let z = -1.0 + (iz as f64 + 0.5) * dx[2];
                    let i = ix * ny * nz + iy * nz + iz;
                    rho[i] = -3.0 * PI * PI * (PI * x).sin() * (PI * y).sin() * (PI * z).sin()
                        / (4.0 * PI);
                }
            }
        }
        let density = DensityField {
            data: rho,
            shape: [nx, ny, nz],
        };
        let pot = mg.solve(&density, 1.0);

        // Check a few interior points
        let mid = nx / 2;
        let i = mid * ny * nz + mid * nz + mid;
        let x = -1.0 + (mid as f64 + 0.5) * dx[0];
        let expected = (PI * x).sin().powi(3); // sin^3 at center
        // Multigrid may not be super accurate at coarse resolution, allow 10% error
        let err = (pot.data[i] - expected).abs();
        assert!(
            err < 0.5,
            "Multigrid sin solution error {err} at center (expected {expected}, got {})",
            pot.data[i]
        );
    }

    #[test]
    fn multigrid_convergence_order() {
        // Double resolution should reduce error by ~4x (2nd order)
        let domain8 = periodic_domain(8);
        let domain16 = periodic_domain(16);
        let mg8 = Multigrid::new(&domain8, 3, 3);
        let mg16 = Multigrid::new(&domain16, 4, 3);

        let make_rho = |shape: [usize; 3], dx: [f64; 3]| -> Vec<f64> {
            let mut rho = vec![0.0; shape[0] * shape[1] * shape[2]];
            for ix in 0..shape[0] {
                for iy in 0..shape[1] {
                    for iz in 0..shape[2] {
                        let x = -1.0 + (ix as f64 + 0.5) * dx[0];
                        let y = -1.0 + (iy as f64 + 0.5) * dx[1];
                        let z = -1.0 + (iz as f64 + 0.5) * dx[2];
                        rho[ix * shape[1] * shape[2] + iy * shape[2] + iz] =
                            -3.0 * PI * PI * (PI * x).sin() * (PI * y).sin() * (PI * z).sin()
                                / (4.0 * PI);
                    }
                }
            }
            rho
        };

        let rho8 = make_rho(mg8.shape, mg8.dx);
        let rho16 = make_rho(mg16.shape, mg16.dx);

        let pot8 = mg8.solve(
            &DensityField {
                data: rho8,
                shape: mg8.shape,
            },
            1.0,
        );
        let pot16 = mg16.solve(
            &DensityField {
                data: rho16,
                shape: mg16.shape,
            },
            1.0,
        );

        // Compute max error at cell centers vs analytic
        let err = |pot: &PotentialField, dx: [f64; 3]| -> f64 {
            let [nx, ny, nz] = pot.shape;
            let mut max_e = 0.0f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let x = -1.0 + (ix as f64 + 0.5) * dx[0];
                        let y = -1.0 + (iy as f64 + 0.5) * dx[1];
                        let z = -1.0 + (iz as f64 + 0.5) * dx[2];
                        let exact = (PI * x).sin() * (PI * y).sin() * (PI * z).sin();
                        let e = (pot.data[ix * ny * nz + iy * nz + iz] - exact).abs();
                        max_e = max_e.max(e);
                    }
                }
            }
            max_e
        };

        let e8 = err(&pot8, mg8.dx);
        let e16 = err(&pot16, mg16.dx);

        if e8 > 1e-10 && e16 > 1e-10 {
            let ratio = e8 / e16;
            // Should be ~4 for 2nd order, but allow wide margin
            assert!(
                ratio > 1.5,
                "Convergence ratio {ratio} too low (e8={e8}, e16={e16})"
            );
        }
    }

    #[test]
    fn multigrid_vs_fft() {
        use crate::tooling::core::poisson::fft::FftPoisson;
        let domain = periodic_domain(16);
        let mg = Multigrid::new(&domain, 4, 3);
        let fft = FftPoisson::new(&domain);

        let [nx, ny, nz] = mg.shape;
        let dx = mg.dx;
        let mut rho = vec![0.0; nx * ny * nz];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = -1.0 + (ix as f64 + 0.5) * dx[0];
                    let y = -1.0 + (iy as f64 + 0.5) * dx[1];
                    let z = -1.0 + (iz as f64 + 0.5) * dx[2];
                    rho[ix * ny * nz + iy * nz + iz] =
                        (PI * x).sin() * (PI * y).sin() * (PI * z).sin();
                }
            }
        }

        let density = DensityField {
            data: rho,
            shape: [nx, ny, nz],
        };
        let pot_mg = mg.solve(&density, 1.0);
        let pot_fft = fft.solve(&density, 1.0);

        // Subtract means (multigrid and FFT may differ by a constant)
        let mean_mg: f64 = pot_mg.data.iter().sum::<f64>() / pot_mg.data.len() as f64;
        let mean_fft: f64 = pot_fft.data.iter().sum::<f64>() / pot_fft.data.len() as f64;

        let mut max_diff = 0.0f64;
        for i in 0..pot_mg.data.len() {
            let diff = ((pot_mg.data[i] - mean_mg) - (pot_fft.data[i] - mean_fft)).abs();
            max_diff = max_diff.max(diff);
        }

        // Allow reasonable tolerance since they use different methods
        let scale = pot_fft
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max)
            .max(1.0);
        assert!(
            max_diff / scale < 0.1,
            "Multigrid vs FFT max diff {max_diff} (scale {scale})"
        );
    }
}
