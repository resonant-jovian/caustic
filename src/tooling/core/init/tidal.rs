//! Tidal stream initial conditions: progenitor cluster orbiting in a fixed host potential.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use super::isolated::IsolatedEquilibrium;

/// Tidal stream IC: progenitor cluster on an orbit in an external host potential.
pub struct TidalIC {
    /// Fixed external host potential Φ_host(x). Does not evolve self-consistently.
    pub host_potential: Box<dyn Fn([f64; 3]) -> f64 + Send + Sync>,
    pub progenitor: Box<dyn IsolatedEquilibrium>,
    pub progenitor_position: [f64; 3],
    pub progenitor_velocity: [f64; 3],
}

impl TidalIC {
    pub fn new(
        host_potential: Box<dyn Fn([f64; 3]) -> f64 + Send + Sync>,
        progenitor: Box<dyn IsolatedEquilibrium>,
        progenitor_position: [f64; 3],
        progenitor_velocity: [f64; 3],
    ) -> Self {
        todo!()
    }

    /// Sample progenitor f centred on (progenitor_position, progenitor_velocity).
    pub fn sample_on_grid(&self, domain: &Domain) -> PhaseSpaceSnapshot {
        todo!()
    }

    /// Compute escape velocity from progenitor at given galactocentric radius r.
    /// Particles outside this radius are tidal debris.
    pub fn escape_velocity(&self, r: f64) -> f64 {
        todo!()
    }
}
