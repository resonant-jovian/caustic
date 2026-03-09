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
        let n = density.data.len();
        if n == 0 {
            return 1e10;
        }
        let rho_mean: f64 = density.data.iter().sum::<f64>() / n as f64;
        if rho_mean <= 0.0 || g <= 0.0 {
            return 1e10;
        }
        let t_orbit = 2.0 * std::f64::consts::PI / (g * rho_mean).sqrt();
        t_orbit / self.steps_per_orbit
    }
}
