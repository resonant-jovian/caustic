//! Orbital time constraint: Δt << T_orbit / steps_per_orbit.
//! Ensures orbital motion is resolved with ~100 steps per orbit.

use super::super::super::types::*;

/// Orbital time CFL constraint.
pub struct OrbitalTimeConstraint {
    /// Target number of timesteps per orbit.
    pub steps_per_orbit: f64,
}

impl OrbitalTimeConstraint {
    /// Maximum stable timestep: T_orbit ~ 2π/√(Gρ̄); Δt = T_orbit / steps_per_orbit.
    pub fn max_dt(&self, density: &DensityField, g: f64) -> f64 {
        todo!("T_orbit ~ 2*pi / sqrt(G * rho_mean); dt = T_orbit / steps_per_orbit")
    }
}
