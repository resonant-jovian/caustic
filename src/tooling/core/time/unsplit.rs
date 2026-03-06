//! Method of lines (unsplit) integrator. Treats Vlasov as a 6D PDE without operator
//! splitting. Re-solves Poisson at each Runge-Kutta stage.

use super::super::{
    advecator::Advector, integrator::TimeIntegrator, phasespace::PhaseSpaceRepr,
    solver::PoissonSolver, types::*,
};

/// Method-of-lines Runge-Kutta integrator for the full 6D Vlasov PDE.
pub struct UnsplitIntegrator {
    /// Number of Runge-Kutta stages (2, 3, or 4).
    pub rk_stages: usize,
}

impl UnsplitIntegrator {
    pub fn new(rk_stages: usize) -> Self {
        todo!()
    }
}

impl TimeIntegrator for UnsplitIntegrator {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        todo!(
            "RK on full 6D Vlasov RHS: df/dt = -v*grad_x f + grad Phi * grad_v f. Re-solve Poisson at each stage."
        )
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        todo!()
    }
}
