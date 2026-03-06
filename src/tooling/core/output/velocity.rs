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
        todo!(
            "iterate spatial grid; at each x: rho=int f dv, u=int f*v dv/rho, sigma^2=int f*(v-u)^2 dv/rho ..."
        )
    }

    /// Surface density Σ = ∫ρ dl along projection axis. Gravitational lensing observable.
    pub fn surface_density(&self, axis: usize) -> Vec<f64> {
        todo!("integrate rho along projection axis")
    }

    /// J-factor J = ∫ρ² dV. Proportional to dark matter annihilation signal.
    pub fn j_factor(&self) -> f64 {
        todo!("integral of rho^2 over the volume")
    }
}
