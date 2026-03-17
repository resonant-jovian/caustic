//! Spatial CFL constraint: Δt < Δx / v_max. Applies to Eulerian advection only;
//! not needed for semi-Lagrangian methods.

use super::super::super::init::domain::Domain;

/// Spatial CFL timestep constraint.
pub struct SpatialCfl {
    pub cfl_factor: f64,
}

impl SpatialCfl {
    /// Maximum stable spatial timestep: Δt = cfl_factor × min(Δx) / v_max.
    pub fn max_dt(&self, domain: &Domain) -> f64 {
        let dx = domain.dx();
        let min_dx = dx[0].min(dx[1]).min(dx[2]);
        let v_max = Self::v_max(domain);
        if v_max <= 0.0 {
            return 1e10;
        }
        self.cfl_factor * min_dx / v_max
    }

    /// Maximum velocity magnitude from the domain velocity extents.
    pub fn v_max(domain: &Domain) -> f64 {
        let lv = domain.lv();
        lv[0].max(lv[1]).max(lv[2])
    }
}
