//! Yoshida 4th-order symplectic integrator. Uses 3 Strang-like sub-steps with
//! specific coefficients w0, w1.

use super::super::{
    advecator::Advector, integrator::TimeIntegrator, phasespace::PhaseSpaceRepr,
    solver::PoissonSolver, types::*,
};

/// Yoshida coefficient w1 = 1 / (2 − 2^(1/3)).
const YOSHIDA_W1: f64 = 1.3512071919596578;

/// Yoshida coefficient w0 = 1 − 2·w1.
const YOSHIDA_W0: f64 = -1.7024143839193153;

/// Yoshida 4th-order symplectic integrator.
pub struct YoshidaSplitting;

impl YoshidaSplitting {
    pub fn new() -> Self {
        todo!()
    }
}

impl TimeIntegrator for YoshidaSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        todo!("3-stage Strang with weights [w1, w0, w1] for drift sub-steps, [w1, w1] for kicks")
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        todo!("same as Strang but Yoshida has larger stable region")
    }
}
