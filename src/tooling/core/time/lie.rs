//! Lie (first-order) splitting: drift(Δt) → kick(Δt). Only 1st-order accurate.
//! Use only for testing and comparison.

use super::super::{
    types::*,
    integrator::TimeIntegrator,
    phasespace::PhaseSpaceRepr,
    solver::PoissonSolver,
    advecator::Advector,
};

/// Lie (1st-order) operator splitting: drift(Δt) → kick(Δt).
pub struct LieSplitting;

impl LieSplitting {
    pub fn new() -> Self {
        todo!()
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
        todo!("1. drift(dt); 2. compute density; 3. solve Poisson; 4. kick(dt)")
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        todo!()
    }
}
