//! Two-body merger / interaction initial conditions.
//! Superposition of two isolated equilibria displaced and boosted.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use super::isolated::IsolatedEquilibrium;

/// f(x,v,t₀) = f₁(x−x₁, v−v₁) + f₂(x−x₂, v−v₂).
/// Exact for collisionless systems before interaction begins.
pub struct MergerIC {
    pub body1: Box<dyn IsolatedEquilibrium>,
    pub mass1: f64,
    pub body2: Box<dyn IsolatedEquilibrium>,
    pub mass2: f64,
    pub separation: [f64; 3],
    pub relative_velocity: [f64; 3],
    pub impact_parameter: f64,
}

impl MergerIC {
    pub fn new(
        body1: Box<dyn IsolatedEquilibrium>,
        mass1: f64,
        body2: Box<dyn IsolatedEquilibrium>,
        mass2: f64,
        separation: [f64; 3],
        relative_velocity: [f64; 3],
        impact_parameter: f64,
    ) -> Self {
        todo!()
    }

    /// Sample both components on the grid and sum.
    pub fn sample_on_grid(&self, domain: &Domain) -> PhaseSpaceSnapshot {
        todo!("evaluate f1(x-x1,v-v1) + f2(x-x2,v-v2) at every grid point")
    }

    /// Check that both components fit within the domain.
    pub fn validate(&self, domain: &Domain) -> anyhow::Result<()> {
        todo!()
    }
}
