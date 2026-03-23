//! The `TimeIntegrator` trait. Orchestrates the full timestep: compute density → solve
//! Poisson → compute force → advect. Different splitting orders implement this differently.

use std::sync::Arc;

use super::advecator::Advector;
use super::phasespace::PhaseSpaceRepr;
use super::progress::StepProgress;
use super::solver::PoissonSolver;
use super::types::{AccelerationField, DensityField, PotentialField};

/// End-of-step products from `TimeIntegrator::advance()`.
///
/// Contains the density, potential, and acceleration fields computed at the
/// end of the time step. By returning these explicitly, the caller can reuse
/// them for diagnostics and conservation projections without redundant
/// Poisson solves.
pub struct StepProducts {
    pub density: DensityField,
    pub potential: PotentialField,
    pub acceleration: AccelerationField,
}

/// Complete simulation state at one instant.
pub struct SimState {
    pub time: f64,
    pub step: u64,
    /// Per-step rank diagnostics from `InstrumentedStrangSplitting`.
    /// `None` when using a non-instrumented integrator or non-HT representation.
    pub step_rank_diagnostics: Option<super::time::rank_monitor::StepRankDiagnostics>,
}

/// Per-step phase timing breakdown in milliseconds.
/// Populated by instrumented time integrators and `Simulation::step()`.
#[derive(Clone, Debug, Default)]
pub struct StepTimings {
    /// Total time in spatial drift sub-steps (advect_x).
    pub drift_ms: f64,
    /// Total time in Poisson solve (density → potential → acceleration).
    pub poisson_ms: f64,
    /// Total time in velocity kick sub-steps (advect_v).
    pub kick_ms: f64,
    /// Time in post-advance density computation (for diagnostics).
    pub density_ms: f64,
    /// Time in diagnostics computation (conservation quantities).
    pub diagnostics_ms: f64,
    /// Time in I/O (checkpoints, snapshots).
    pub io_ms: f64,
    /// Remaining time not attributed to the above phases.
    pub other_ms: f64,
}

impl StepTimings {
    /// Convert to the 7-element array format used by phasma SimState.
    pub fn to_array(&self) -> [f64; 7] {
        [
            self.drift_ms,
            self.poisson_ms,
            self.kick_ms,
            self.density_ms,
            self.diagnostics_ms,
            self.io_ms,
            self.other_ms,
        ]
    }
}

/// Trait for all time integration / operator splitting strategies.
pub trait TimeIntegrator {
    /// Advance the simulation by one timestep Δt.
    ///
    /// Calls advector drift/kick sub-steps in the correct order for this splitting
    /// scheme, then computes and returns the end-of-step density, potential, and
    /// acceleration. The caller uses these products for diagnostics and
    /// conservation projections, avoiding redundant Poisson solves.
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> StepProducts;

    /// Compute the maximum stable Δt given current state and CFL constraints.
    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64;

    /// Return timing breakdown from the most recent `advance()` call.
    /// Default returns `None`; instrumented integrators override this.
    fn last_step_timings(&self) -> Option<&StepTimings> {
        None
    }

    /// Attach shared progress state for intra-step TUI visibility.
    /// Default is a no-op; integrators that support progress override this.
    fn set_progress(&mut self, _progress: Arc<StepProgress>) {}

    /// Return a suggested Δt from the adaptive controller (if any).
    /// Default returns `None`; adaptive integrators override this.
    fn suggested_dt(&self) -> Option<f64> {
        None
    }
}
