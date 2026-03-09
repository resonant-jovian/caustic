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
        let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);
        if rho_max <= 0.0 || g <= 0.0 {
            return 1e10;
        }
        self.safety_factor / (g * rho_max).sqrt()
    }
}
