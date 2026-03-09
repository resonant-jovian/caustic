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
        Self {
            disk_surface_density,
            disk_velocity_dispersion,
            bulge: None,
            halo_potential: None,
            perturbation_mode_m,
            perturbation_pattern_speed,
            perturbation_amplitude,
        }
    }

    /// Toomre Q(R) = σ_R κ / (3.36 G Σ). Q > 1 means locally stable.
    ///
    /// Requires the epicyclic frequency κ(R) = √(R dΩ²/dR + 4Ω²),
    /// which needs the full circular velocity curve from the combined
    /// disk+bulge+halo potential. Not yet implemented.
    pub fn toomre_q(&self, _radius: f64) -> f64 {
        todo!("requires epicyclic frequency κ(R) from combined disk+bulge+halo potential")
    }

    /// Sample onto 6D grid: construct f(E, Lz) for disk, superpose azimuthal perturbation.
    ///
    /// Constructing the disk DF f(E, Lz) requires:
    /// 1. Computing the total potential Φ(R, z) from disk + bulge + halo
    /// 2. Inverting Σ(R), σ_R(R) to obtain f(E, Lz) via Shu (1969) or Dehnen (1999)
    /// 3. Superposing the (m, Ω_p) perturbation mode.
    ///    This is highly specialized and not yet implemented.
    pub fn sample_on_grid(&self, _domain: &Domain) -> PhaseSpaceSnapshot {
        todo!(
            "construct f(E,Lz) for disk via Shu/Dehnen inversion + superpose azimuthal mode perturbation"
        )
    }
}
