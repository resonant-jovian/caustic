//! `Diagnostics` — computes and stores all conserved quantities and monitoring outputs
//! at each timestep.

use super::types::{DensityField, PotentialField};
use super::phasespace::PhaseSpaceRepr;

/// One row of the global time-series output.
#[derive(Debug, Clone, Copy)]
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
    /// `dx3` is the spatial cell volume dx1*dx2*dx3.
    pub fn compute(
        &mut self,
        repr: &dyn PhaseSpaceRepr,
        potential: &PotentialField,
        time: f64,
        dx3: f64,
    ) -> GlobalDiagnostics {
        let density = repr.compute_density();
        let t = Self::kinetic_energy(repr);
        let w = Self::potential_energy(&density, potential, dx3);
        let e = t + w;
        let c2 = repr.casimir_c2();
        let s = repr.entropy();
        let m = repr.total_mass();
        let vir = if w.abs() > 1e-30 { 2.0 * t / w.abs() } else { 0.0 };
        let diag = GlobalDiagnostics {
            time,
            total_energy: e,
            kinetic_energy: t,
            potential_energy: w,
            virial_ratio: vir,
            total_momentum: [0.0; 3],
            total_angular_momentum: [0.0; 3],
            casimir_c2: c2,
            entropy: s,
            mass_in_box: m,
        };
        self.history.push(diag);
        diag
    }

    /// Total kinetic energy T = ½∫fv² dx³dv³.
    pub fn kinetic_energy(repr: &dyn PhaseSpaceRepr) -> f64 {
        repr.total_kinetic_energy()
    }

    /// Total potential energy W = ½∫ρΦ dx³.
    /// `dx3` is the spatial cell volume.
    pub fn potential_energy(density: &DensityField, potential: &PotentialField, dx3: f64) -> f64 {
        0.5 * density.data.iter().zip(potential.data.iter())
            .map(|(&rho, &phi)| rho * phi)
            .sum::<f64>() * dx3
    }

    /// Virial ratio 2T/|W|. Equals 1.0 at equilibrium.
    pub fn virial_ratio(t: f64, w: f64) -> f64 {
        2.0 * t / w.abs()
    }

    /// Spherically averaged density profile ρ(r) at current timestep.
    pub fn density_profile(density: &DensityField) -> Vec<(f64, f64)> {
        todo!("bin by radius, average")
    }
}
