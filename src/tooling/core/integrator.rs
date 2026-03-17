//! The `TimeIntegrator` trait. Orchestrates the full timestep: compute density → solve
//! Poisson → compute force → advect. Different splitting orders implement this differently.

use super::advecator::Advector;
use super::phasespace::PhaseSpaceRepr;
use super::solver::PoissonSolver;

/// Complete simulation state at one instant.
pub struct SimState {
    pub time: f64,
    pub step: u64,
    /// Per-step rank diagnostics from `InstrumentedStrangSplitting`.
    /// `None` when using a non-instrumented integrator or non-HT representation.
    pub step_rank_diagnostics: Option<super::time::rank_monitor::StepRankDiagnostics>,
}

/// Trait for all time integration / operator splitting strategies.
pub trait TimeIntegrator {
    /// Advance the simulation by one timestep Δt.
    /// Calls advector drift/kick sub-steps in the correct order for this splitting scheme.
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    );

    /// Compute the maximum stable Δt given current state and CFL constraints.
    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64;
}
