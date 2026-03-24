//! Velocity moment fields derived from the distribution function f(x,v,t).
//!
//! Computes the first few velocity moments on the spatial grid: zeroth (density),
//! first (mean bulk velocity), second (velocity dispersion tensor and kinetic
//! energy density), and third (heat flux, currently zeroed). These are the
//! standard fluid-like observables of a collisionless system and are used for
//! radial profile construction, anisotropy measurement, and dark-matter
//! observables such as the J-factor and projected surface density.

use super::super::{init::domain::Domain, phasespace::PhaseSpaceRepr, types::*};
use rayon::prelude::*;

/// All velocity moments of the distribution function on the spatial grid.
pub struct VelocityMoments {
    /// Zeroth moment: mass density rho(x) = integral f dv^3.
    pub density: DensityField,
    /// First moment: mean bulk velocity u_i(x) per spatial cell, indexed [component][cell].
    pub mean_velocity: [Vec<f64>; 3],
    /// Second central moment: velocity dispersion sigma_{ij}(x), stored as a flat
    /// 3x3 tensor per cell, indexed [i*3+j][cell].
    pub dispersion_tensor: [Vec<f64>; 9],
    /// Kinetic energy density 0.5 * rho * (Tr(sigma) + |u|^2) per cell.
    pub kinetic_energy_density: Vec<f64>,
    /// Third moment: heat flux q_i(x) per cell (currently zero -- requires third-order moment).
    pub heat_flux: [Vec<f64>; 3],
}

impl VelocityMoments {
    /// Compute all velocity moments from the distribution function.
    ///
    /// Parallelized over spatial cells with rayon; each cell independently
    /// queries `repr.moment()` for first and second velocity moments.
    pub fn compute(repr: &dyn PhaseSpaceRepr, domain: &Domain) -> Self {
        let density = repr.compute_density();
        let [nx1, nx2, nx3] = density.shape;
        let n_spatial = nx1 * nx2 * nx3;
        let dx = domain.dx();
        let lx = domain.lx();

        // Per-cell output: (mean_vel[3], disp[9], ke_density)
        // Compute in parallel over flat spatial indices.
        let per_cell: Vec<([f64; 3], [f64; 9], f64)> = (0..n_spatial)
            .into_par_iter()
            .map(|idx| {
                let ix1 = idx / (nx2 * nx3);
                let ix2 = (idx / nx3) % nx2;
                let ix3 = idx % nx3;

                let rho = density.data[idx];
                if rho <= 0.0 {
                    return ([0.0; 3], [0.0; 9], 0.0);
                }

                let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
                let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];
                let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                let pos = [x1, x2, x3];

                // Order 1: mean velocity
                let m1 = repr.moment(&pos, 1);
                let u = [m1.data[0], m1.data[1], m1.data[2]];

                // Order 2: dispersion tensor
                let m2 = repr.moment(&pos, 2);
                let mut d = [0.0; 9];
                for i in 0..3 {
                    for j in 0..3 {
                        let raw = m2.data[i * 3 + j];
                        d[i * 3 + j] = raw / rho - u[i] * u[j];
                    }
                }

                // KE density
                let trace = d[0] + d[4] + d[8];
                let u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                let ke = 0.5 * rho * (trace + u2);

                (u, d, ke)
            })
            .collect();

        // Scatter per-cell results into output arrays
        let mut mean_vel = [
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
        ];
        let mut disp = std::array::from_fn::<Vec<f64>, 9, _>(|_| vec![0.0; n_spatial]);
        let mut ke_density = vec![0.0; n_spatial];

        for (idx, (u, d, ke)) in per_cell.into_iter().enumerate() {
            mean_vel[0][idx] = u[0];
            mean_vel[1][idx] = u[1];
            mean_vel[2][idx] = u[2];
            for k in 0..9 {
                disp[k][idx] = d[k];
            }
            ke_density[idx] = ke;
        }

        let heat_flux = [
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
        ];

        Self {
            density,
            mean_velocity: mean_vel,
            dispersion_tensor: disp,
            kinetic_energy_density: ke_density,
            heat_flux, // zeros — requires third moment not available
        }
    }

    /// Surface density Σ = ∫ρ dl along projection axis. Gravitational lensing observable.
    ///
    /// `axis`: 0=x1, 1=x2, 2=x3. Result is in grid units (caller multiplies by dx).
    pub fn surface_density(&self, axis: usize) -> Vec<f64> {
        let [nx1, nx2, nx3] = self.density.shape;
        match axis {
            0 => {
                let mut out = vec![0.0; nx2 * nx3];
                for ix1 in 0..nx1 {
                    for ix2 in 0..nx2 {
                        for ix3 in 0..nx3 {
                            out[ix2 * nx3 + ix3] +=
                                self.density.data[ix1 * nx2 * nx3 + ix2 * nx3 + ix3];
                        }
                    }
                }
                out
            }
            1 => {
                let mut out = vec![0.0; nx1 * nx3];
                for ix1 in 0..nx1 {
                    for ix2 in 0..nx2 {
                        for ix3 in 0..nx3 {
                            out[ix1 * nx3 + ix3] +=
                                self.density.data[ix1 * nx2 * nx3 + ix2 * nx3 + ix3];
                        }
                    }
                }
                out
            }
            2 => {
                let mut out = vec![0.0; nx1 * nx2];
                for ix1 in 0..nx1 {
                    for ix2 in 0..nx2 {
                        for ix3 in 0..nx3 {
                            out[ix1 * nx2 + ix2] +=
                                self.density.data[ix1 * nx2 * nx3 + ix2 * nx3 + ix3];
                        }
                    }
                }
                out
            }
            _ => {
                debug_assert!(false, "axis must be 0, 1, or 2");
                vec![]
            }
        }
    }

    /// J-factor J = ∫ρ² dV. Proportional to dark matter annihilation signal.
    ///
    /// Returns sum(ρ²) in grid units — caller multiplies by dx³.
    pub fn j_factor(&self) -> f64 {
        self.density.data.iter().map(|&rho| rho * rho).sum()
    }
}
