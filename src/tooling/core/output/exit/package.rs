//! Standardised exit output package produced at simulation termination.

use super::super::super::{
    conditions::ExitReason,
    diagnostics::GlobalDiagnostics,
    io::IOManager,
    output::global::{ConservationSummary, conservation_summary},
    types::*,
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

    /// Write all fields: snapshot binary, diagnostics CSV, exit metadata JSON.
    pub fn save(&self, io: &IOManager) -> anyhow::Result<()> {
        use std::io::Write;

        // 1. Save final snapshot
        io.save_snapshot(&self.final_snapshot, "final_snapshot.bin")?;

        // 2. Write diagnostics CSV
        for row in &self.diagnostics_history {
            io.append_diagnostics(row)?;
        }

        // 3. Write exit metadata JSON
        let metadata = serde_json::json!({
            "exit_reason": format!("{:?}", self.exit_reason),
            "exit_message": self.exit_message,
            "total_steps": self.total_steps,
            "wall_clock_seconds": self.wall_clock_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "conservation": {
                "max_energy_drift": self.conservation_summary.max_energy_drift,
                "max_momentum_drift": self.conservation_summary.max_momentum_drift,
                "max_angular_momentum_drift": self.conservation_summary.max_angular_momentum_drift,
                "max_casimir_drift": self.conservation_summary.max_casimir_drift,
            }
        });

        std::fs::create_dir_all(&io.output_dir)?;
        let path = io.output_dir.join("exit_metadata.json");
        let mut file = std::fs::File::create(path)?;
        file.write_all(serde_json::to_string_pretty(&metadata)?.as_bytes())?;

        Ok(())
    }

    /// Print human-readable conservation errors and performance statistics.
    pub fn print_summary(&self) {
        println!("Exit: {:?}", self.exit_reason);
        println!(
            "Steps: {}, Wall clock: {:.2}s",
            self.total_steps, self.wall_clock_seconds
        );
        println!(
            "Max energy drift: {:.2e}",
            self.conservation_summary.max_energy_drift
        );
        println!(
            "Max Casimir drift: {:.2e}",
            self.conservation_summary.max_casimir_drift
        );
    }
}
