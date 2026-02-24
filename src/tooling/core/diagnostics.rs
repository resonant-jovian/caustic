//! `Diagnostics` — computes and stores all conserved quantities and monitoring outputs
//! at each timestep.

use super::types::{DensityField, PotentialField};
use super::phasespace::PhaseSpaceRepr;

/// One row of the global time-series output.
pub struct GlobalDiagnostics {
    pub time: f64,
    pub total_energy: f64,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub virial_ratio: f64,
    pub total_momentum: [f64; 3],
    pub total_angular_momentum: [f64; 3],
    pub casimir_c2: f64,
    pub entropy: f64,
    pub mass_in_box: f64,
}

/// Accumulates time series of `GlobalDiagnostics`.
pub struct Diagnostics {
    pub history: Vec<GlobalDiagnostics>,
}

impl Diagnostics {
    /// Compute all global diagnostics from the current representation and potential.
    pub fn compute(
        &mut self,
        repr: &dyn PhaseSpaceRepr,
        potential: &PotentialField,
        time: f64,
    ) -> GlobalDiagnostics {
        todo!("integrate E=T+W, P, L, C2, S, M over 6D grid")
    }

    /// Total kinetic energy T = ½∫fv² dx³dv³.
    pub fn kinetic_energy(repr: &dyn PhaseSpaceRepr) -> f64 {
        todo!()
    }

    /// Total potential energy W = ½∫ρΦ dx³.
    pub fn potential_energy(density: &DensityField, potential: &PotentialField) -> f64 {
        todo!()
    }

    /// Virial ratio 2T/|W|. Equals 1.0 at equilibrium.
    pub fn virial_ratio(t: f64, w: f64) -> f64 {
        todo!()
    }

    /// Spherically averaged density profile ρ(r) at current timestep.
    pub fn density_profile(density: &DensityField) -> Vec<(f64, f64)> {
        todo!("bin by radius, average")
    }
}
