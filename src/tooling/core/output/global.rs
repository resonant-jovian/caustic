//! Global time-series diagnostics output. Provides formatted output functions.

use super::super::diagnostics::GlobalDiagnostics;

/// Summary of maximum conservation drifts over the full simulation.
pub struct ConservationSummary {
    pub max_energy_drift: f64,
    pub max_momentum_drift: f64,
    pub max_angular_momentum_drift: f64,
    pub max_casimir_drift: f64,
}

/// Write time-series diagnostics to CSV.
/// Columns: t, E, T, W, 2T/|W|, Px, Py, Pz, Lx, Ly, Lz, C2, S, M.
pub fn write_csv(history: &[GlobalDiagnostics], path: &str) -> anyhow::Result<()> {
    todo!("write CSV with columns: t, E, T, W, 2T/|W|, P, L, C2, S, M")
}

/// Compute maximum relative drifts of all conserved quantities.
pub fn conservation_summary(history: &[GlobalDiagnostics]) -> ConservationSummary {
    todo!("compute max drift of E, P, L, C2 relative to initial values")
}
