//! Spatial CFL constraint: Δt < Δx / v_max. Applies to Eulerian advection only;
//! not needed for semi-Lagrangian methods.

use super::super::super::{init::domain::Domain, phasespace::PhaseSpaceRepr, types::*};

/// Spatial CFL timestep constraint.
pub struct SpatialCfl {
    pub cfl_factor: f64,
}

impl SpatialCfl {
    /// Maximum stable spatial timestep: Δt = cfl_factor × min(Δx) / v_max.
    pub fn max_dt(&self, domain: &Domain, repr: &dyn PhaseSpaceRepr) -> f64 {
        todo!("dt = cfl_factor * min(dx) / v_max")
    }

    /// Find the maximum occupied velocity magnitude in the distribution.
    pub fn v_max(repr: &dyn PhaseSpaceRepr) -> f64 {
        todo!("scan velocity grid for maximum occupied velocity")
    }
}
