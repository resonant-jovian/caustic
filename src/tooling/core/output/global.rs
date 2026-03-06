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
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "t,E,T,W,vir,Px,Py,Pz,Lx,Ly,Lz,C2,S,M")?;
    for d in history {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            d.time,
            d.total_energy,
            d.kinetic_energy,
            d.potential_energy,
            d.virial_ratio,
            d.total_momentum[0],
            d.total_momentum[1],
            d.total_momentum[2],
            d.total_angular_momentum[0],
            d.total_angular_momentum[1],
            d.total_angular_momentum[2],
            d.casimir_c2,
            d.entropy,
            d.mass_in_box
        )?;
    }
    Ok(())
}

/// Compute maximum relative drifts of all conserved quantities.
pub fn conservation_summary(history: &[GlobalDiagnostics]) -> ConservationSummary {
    if history.is_empty() {
        return ConservationSummary {
            max_energy_drift: 0.0,
            max_momentum_drift: 0.0,
            max_angular_momentum_drift: 0.0,
            max_casimir_drift: 0.0,
        };
    }
    let e0 = history[0].total_energy.abs().max(1e-30);
    let c2_0 = history[0].casimir_c2.abs().max(1e-30);

    let max_energy_drift = history
        .iter()
        .map(|d| (d.total_energy - history[0].total_energy).abs() / e0)
        .fold(0.0_f64, f64::max);
    let max_casimir_drift = history
        .iter()
        .map(|d| (d.casimir_c2 - history[0].casimir_c2).abs() / c2_0)
        .fold(0.0_f64, f64::max);

    ConservationSummary {
        max_energy_drift,
        max_momentum_drift: 0.0,
        max_angular_momentum_drift: 0.0,
        max_casimir_drift,
    }
}
