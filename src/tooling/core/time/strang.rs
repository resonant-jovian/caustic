//! Strang splitting time integrator. 2nd-order symmetric:
//! drift(Δt/2) → kick(Δt) → drift(Δt/2). Naturally symplectic.

use super::super::{
    types::*,
    integrator::TimeIntegrator,
    phasespace::PhaseSpaceRepr,
    solver::PoissonSolver,
    advecator::Advector,
};

/// Strang splitting: drift(Δt/2) → kick(Δt) → drift(Δt/2).
pub struct StrangSplitting;

impl StrangSplitting {
    pub fn new() -> Self {
        todo!()
    }
}

impl TimeIntegrator for StrangSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        todo!("1. drift(dt/2); 2. compute density; 3. solve Poisson; 4. compute accel; 5. kick(dt); 6. drift(dt/2)")
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        todo!("t_dyn = 1/sqrt(G*rho_max); return cfl_factor * t_dyn")
    }
}
