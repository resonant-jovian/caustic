//! Standardised exit output package produced at simulation termination.

use super::super::super::{
    types::*,
    conditions::ExitReason,
    diagnostics::GlobalDiagnostics,
    io::IOManager,
    output::global::{ConservationSummary, conservation_summary},
};

/// Complete output produced upon simulation exit, as specified in Section 4.3 of the spec.
pub struct ExitPackage {
    pub final_snapshot: PhaseSpaceSnapshot,
    pub diagnostics_history: Vec<GlobalDiagnostics>,
    pub exit_reason: ExitReason,
    pub exit_message: String,
    pub wall_clock_seconds: f64,
    pub total_steps: u64,
    pub peak_memory_bytes: usize,
    pub conservation_summary: ConservationSummary,
}

impl ExitPackage {
    /// Assemble the exit package from all components at simulation end.
    pub fn assemble(
        snapshot: PhaseSpaceSnapshot,
        history: Vec<GlobalDiagnostics>,
        reason: ExitReason,
        message: String,
        wall_clock_seconds: f64,
        total_steps: u64,
        peak_memory_bytes: usize,
    ) -> Self {
        let summary = conservation_summary(&history);
        Self {
            final_snapshot: snapshot,
            diagnostics_history: history,
            exit_reason: reason,
            exit_message: message,
            wall_clock_seconds,
            total_steps,
            peak_memory_bytes,
            conservation_summary: summary,
        }
    }

    /// Write all fields: snapshot HDF5, diagnostics CSV, exit metadata JSON.
    pub fn save(&self, io: &IOManager) -> anyhow::Result<()> {
        todo!("write snapshot HDF5, diagnostics CSV, exit metadata JSON")
    }

    /// Print human-readable conservation errors and performance statistics.
    pub fn print_summary(&self) {
        println!("Exit: {:?}", self.exit_reason);
        println!("Steps: {}, Wall clock: {:.2}s", self.total_steps, self.wall_clock_seconds);
        println!("Max energy drift: {:.2e}", self.conservation_summary.max_energy_drift);
        println!("Max Casimir drift: {:.2e}", self.conservation_summary.max_casimir_drift);
    }
}
