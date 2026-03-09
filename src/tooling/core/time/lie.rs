//! Lie (first-order) splitting: drift(Δt) → kick(Δt). Only 1st-order accurate.
//! Use only for testing and comparison.

use super::super::{
    advecator::Advector, integrator::TimeIntegrator, phasespace::PhaseSpaceRepr,
    solver::PoissonSolver, types::*,
};

/// Lie (1st-order) operator splitting: drift(Δt) → kick(Δt).
pub struct LieSplitting {
    pub g: f64,
}

impl LieSplitting {
    pub fn new(g: f64) -> Self {
        Self { g }
    }
}

impl TimeIntegrator for LieSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        let _span = tracing::info_span!("lie_advance").entered();

        // 1. Drift full step
        advector.drift(repr, dt);

        // 2. Compute density → Poisson → acceleration → kick full step
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.g);
        let accel = solver.compute_acceleration(&potential);
        advector.kick(repr, &accel, dt);
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        let density = repr.compute_density();
        let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);
        if rho_max <= 0.0 || self.g <= 0.0 {
            return 1e10;
        }
        let t_dyn = 1.0 / (self.g * rho_max).sqrt();
        cfl_factor * t_dyn
    }
}
