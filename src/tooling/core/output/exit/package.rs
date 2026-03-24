//! Standardised exit output package produced at simulation termination.
//!
//! When [`Simulation::run()`](crate::sim::Simulation::run) completes, it returns an [`ExitPackage`] that
//! bundles the final phase-space snapshot, the full diagnostics history,
//! conservation error summaries, and performance statistics. The package can
//! be persisted to disk via [`ExitPackage::save()`] or printed as a
//! human-readable summary via [`ExitPackage::print_summary()`].

use super::super::super::{
    conditions::ExitReason,
    diagnostics::GlobalDiagnostics,
    io::IOManager,
    output::global::{ConservationSummary, conservation_summary},
    types::*,
};

/// Complete output produced upon simulation exit, as specified in Section 4.3 of the spec.
///
/// Returned by [`Simulation::run()`](crate::sim::Simulation::run) and also serialisable via [`save()`](Self::save).
pub struct ExitPackage {
    /// The last phase-space snapshot at the moment of termination.
    pub final_snapshot: PhaseSpaceSnapshot,
    /// Full per-step diagnostics history (energy, momentum, Casimirs, etc.).
    pub diagnostics_history: Vec<GlobalDiagnostics>,
    /// Why the simulation stopped (time limit, drift exceeded, etc.).
    pub exit_reason: ExitReason,
    /// Human-readable description of the exit condition.
    pub exit_message: String,
    /// Total wall-clock time spent in the solver loop.
    pub wall_clock_seconds: f64,
    /// Number of time steps completed.
    pub total_steps: u64,
    /// Peak resident memory observed during the run (bytes).
    pub peak_memory_bytes: usize,
    /// Worst-case conservation errors computed over the diagnostics history.
    pub conservation_summary: ConservationSummary,
}

impl ExitPackage {
    /// Assemble the exit package from all components at simulation end.
    ///
    /// Computes `ConservationSummary` from the diagnostics history automatically.
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

    /// Print human-readable conservation errors and performance statistics to stdout.
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
