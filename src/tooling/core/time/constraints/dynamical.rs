//! Dynamical time constraint: Δt << t_dyn = 1/√(Gρ_max).
//! Always required for correct gravitational coupling.

use super::super::super::types::*;

/// Dynamical time CFL constraint.
pub struct DynamicalTimeConstraint {
    pub safety_factor: f64,
}

impl DynamicalTimeConstraint {
    /// Maximum stable timestep: Δt = safety_factor / √(G × ρ_max).
    pub fn max_dt(&self, density: &DensityField, g: f64) -> f64 {
        todo!("t_dyn = safety_factor / sqrt(G * rho_max)")
    }
}
