//! Phase-space structure diagnostics: stream count, caustic surfaces, power spectrum,
//! growth rates.

use super::super::{phasespace::PhaseSpaceRepr, types::*};

/// Phase-space structure diagnostic outputs.
pub struct PhaseSpaceDiagnostics {
    pub stream_count: StreamCountField,
    pub local_entropy: Vec<f64>,
    pub caustic_cells: Vec<[usize; 3]>,
}

impl PhaseSpaceDiagnostics {
    pub fn compute(repr: &dyn PhaseSpaceRepr) -> Self {
        todo!()
    }

    /// Extract f(v|x) at a given physical position for dark matter detection predictions.
    pub fn velocity_distribution_at(repr: &dyn PhaseSpaceRepr, x: [f64; 3]) -> Vec<f64> {
        todo!()
    }

    /// Power spectrum P(k) = |ρ̂(k)|². FFT of density field.
    pub fn power_spectrum(density: &DensityField) -> Vec<(f64, f64)> {
        todo!("FFT density, bin by k magnitude")
    }

    /// Stability analysis: fit exponential growth rate to each k-mode from density history.
    pub fn growth_rates(density_history: &[DensityField], dt: f64) -> Vec<(f64, f64)> {
        todo!("FFT of rho(k,t) time series; fit exponential")
    }
}
