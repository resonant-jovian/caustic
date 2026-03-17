//! Velocity CFL constraint: Δt < Δv / |g|_max. Applies to Eulerian velocity advection only.

use super::super::super::{init::domain::Domain, types::*};
use rayon::prelude::*;

/// Velocity CFL timestep constraint.
pub struct VelocityCfl {
    pub cfl_factor: f64,
}

impl VelocityCfl {
    /// Maximum stable velocity timestep: Δt = cfl_factor × min(Δv) / max|g|.
    pub fn max_dt(&self, domain: &Domain, accel: &AccelerationField) -> f64 {
        let dv = domain.dv();
        let min_dv = dv[0].min(dv[1]).min(dv[2]);

        let g_max = accel
            .gx
            .par_iter()
            .zip(accel.gy.par_iter())
            .zip(accel.gz.par_iter())
            .map(|((&gx, &gy), &gz)| (gx * gx + gy * gy + gz * gz).sqrt())
            .reduce(|| 0.0_f64, f64::max);

        if g_max <= 0.0 {
            return 1e10;
        }
        self.cfl_factor * min_dv / g_max
    }
}
