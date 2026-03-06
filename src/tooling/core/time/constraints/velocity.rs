//! Velocity CFL constraint: Δt < Δv / |g|_max. Applies to Eulerian velocity advection only.

use super::super::super::{types::*, init::domain::Domain};

/// Velocity CFL timestep constraint.
pub struct VelocityCfl {
    pub cfl_factor: f64,
}

impl VelocityCfl {
    /// Maximum stable velocity timestep: Δt = cfl_factor × min(Δv) / max|g|.
    pub fn max_dt(&self, domain: &Domain, accel: &AccelerationField) -> f64 {
        todo!("dt = cfl_factor * min(dv) / max|g|")
    }
}
