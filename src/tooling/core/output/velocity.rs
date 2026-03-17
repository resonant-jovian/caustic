//! Velocity moment fields derived from f: density, mean velocity, dispersion tensor,
//! kinetic energy density, heat flux.

use super::super::{init::domain::Domain, phasespace::PhaseSpaceRepr, types::*};

/// All velocity moments of the distribution function.
pub struct VelocityMoments {
    pub density: DensityField,
    pub mean_velocity: [Vec<f64>; 3],
    pub dispersion_tensor: [Vec<f64>; 9],
    pub kinetic_energy_density: Vec<f64>,
    pub heat_flux: [Vec<f64>; 3],
}

impl VelocityMoments {
    /// Compute all velocity moments from the distribution function.
    pub fn compute(repr: &dyn PhaseSpaceRepr, domain: &Domain) -> Self {
        let density = repr.compute_density();
        let [nx1, nx2, nx3] = density.shape;
        let n_spatial = nx1 * nx2 * nx3;
        let dx = domain.dx();
        let lx = domain.lx();

        let mut mean_vel = [
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
        ];
        let mut disp = std::array::from_fn::<Vec<f64>, 9, _>(|_| vec![0.0; n_spatial]);
        let mut ke_density = vec![0.0; n_spatial];
        let heat_flux = [
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
            vec![0.0; n_spatial],
        ];

        for ix1 in 0..nx1 {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            for ix2 in 0..nx2 {
                let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];
                for ix3 in 0..nx3 {
                    let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                    let idx = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                    let pos = [x1, x2, x3];
                    let rho = density.data[idx];

                    if rho <= 0.0 {
                        continue;
                    }

                    // Order 1: mean velocity (already normalized by repr)
                    let m1 = repr.moment(&pos, 1);
                    let u = [m1.data[0], m1.data[1], m1.data[2]];
                    mean_vel[0][idx] = u[0];
                    mean_vel[1][idx] = u[1];
                    mean_vel[2][idx] = u[2];

                    // Order 2: raw second moment <f*v_i*v_j>*dv³ (unnormalized)
                    let m2 = repr.moment(&pos, 2);
                    // Dispersion: sigma_ij = m2_ij/rho - u_i*u_j
                    for i in 0..3 {
                        for j in 0..3 {
                            let raw = m2.data[i * 3 + j];
                            disp[i * 3 + j][idx] = raw / rho - u[i] * u[j];
                        }
                    }

                    // KE density: 0.5 * rho * (trace(sigma) + |u|²)
                    let trace = disp[0][idx] + disp[4][idx] + disp[8][idx];
                    let u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
                    ke_density[idx] = 0.5 * rho * (trace + u2);
                }
            }
        }

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
            _ => panic!("axis must be 0, 1, or 2"),
        }
    }

    /// J-factor J = ∫ρ² dV. Proportional to dark matter annihilation signal.
    ///
    /// Returns sum(ρ²) in grid units — caller multiplies by dx³.
    pub fn j_factor(&self) -> f64 {
        self.density.data.iter().map(|&rho| rho * rho).sum()
    }
}
