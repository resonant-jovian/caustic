//! Disk stability initial conditions: equilibrium disk + bulge + halo with a seeded
//! perturbation mode for studying spiral arm / bar instabilities.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use super::isolated::IsolatedEquilibrium;

/// Disk stability IC: f(E, Lz) for axisymmetric disk plus an optional perturbation.
pub struct DiskStabilityIC {
    /// Disk surface density Σ(R) as function of cylindrical radius.
    pub disk_surface_density: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Radial velocity dispersion σ_R(R).
    pub disk_velocity_dispersion: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Optional central bulge component.
    pub bulge: Option<Box<dyn IsolatedEquilibrium>>,
    /// Optional fixed dark matter halo potential Φ_halo(x).
    pub halo_potential: Option<Box<dyn Fn([f64; 3]) -> f64 + Send + Sync>>,
    /// Azimuthal mode number m (m=2 = bar, m=3 = triangle).
    pub perturbation_mode_m: u32,
    /// Pattern speed Ω_p in rad/time.
    pub perturbation_pattern_speed: f64,
    /// Relative amplitude δΣ/Σ.
    pub perturbation_amplitude: f64,
}

impl DiskStabilityIC {
    pub fn new(
        disk_surface_density: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        disk_velocity_dispersion: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        perturbation_mode_m: u32,
        perturbation_pattern_speed: f64,
        perturbation_amplitude: f64,
    ) -> Self {
        todo!()
    }

    /// Toomre Q(R) = σ_R κ / (3.36 G Σ). Q > 1 means locally stable.
    pub fn toomre_q(&self, radius: f64) -> f64 {
        todo!("requires epicyclic frequency kappa(R)")
    }

    /// Sample onto 6D grid: construct f(E, Lz) for disk, superpose azimuthal perturbation.
    pub fn sample_on_grid(&self, domain: &Domain) -> PhaseSpaceSnapshot {
        todo!("construct f(E,Lz) for disk + superpose azimuthal mode perturbation")
    }
}
