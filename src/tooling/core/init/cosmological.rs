//! Zel'dovich approximation initial conditions for cosmological structure formation.
//! f is a 3D sheet in 6D phase space (cold dark matter).

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;

/// Cosmological parameters for Friedmann background.
pub struct CosmologyParams {
    pub h0: f64,
    pub omega_m: f64,
    pub omega_lambda: f64,
    pub a_init: f64,
}

/// 1D matter power spectrum P(k) for seeding perturbations.
pub struct PowerSpectrum {
    /// (k, P(k)) pairs.
    pub values: Vec<(f64, f64)>,
}

/// Zel'dovich pancake IC: cold dark matter sheet.
/// f(x,v,t₀) = ρ̄·δ³(v − v₀(x)) where v₀ is the Zel'dovich velocity field.
pub struct ZeldovichIC {
    pub mean_density: f64,
    pub h0: f64,
    pub omega_m: f64,
    pub omega_lambda: f64,
    pub scale_factor_init: f64,
    pub random_seed: u64,
}

impl ZeldovichIC {
    pub fn new(mean_density: f64, cosmology: CosmologyParams, seed: u64) -> Self {
        todo!()
    }

    /// Generate the Zel'dovich displacement field s(q) from P(k) via FFT.
    pub fn displacement_field(&self, domain: &Domain) -> [Vec<f64>; 3] {
        todo!("FFT of Gaussian random field weighted by P(k)^(1/2)")
    }

    /// Zel'dovich velocity v₀(q) = H·f·s (growing mode).
    pub fn velocity_field(&self, domain: &Domain) -> [Vec<f64>; 3] {
        todo!()
    }

    /// Sample onto 6D grid as a thin Gaussian in velocity centred on v₀(x).
    pub fn sample_on_grid(&self, domain: &Domain) -> PhaseSpaceSnapshot {
        todo!("Gaussian sigma = velocity dispersion of cold DM sheet")
    }
}

/// Growth factor D(a) via ODE integration: D'' + H(a)D' = (3/2) Ω_m H₀² D/a³.
pub fn growth_factor(a: f64, params: &CosmologyParams) -> f64 {
    todo!("D(a) via ODE: D'' + H(a)*D' = 3/2 * Omega_m * H0^2 * D / a^3")
}
