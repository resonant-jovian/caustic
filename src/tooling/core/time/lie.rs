//! Lie (first-order) operator splitting: drift(Δt) → kick(Δt).
//!
//! Applies spatial drift and velocity kick sequentially without the symmetric
//! half-step structure of Strang splitting. This yields only 1st-order accuracy
//! and does not preserve symplecticity, so it is not suitable for production runs.
//!
//! Primary use: as an error estimator in adaptive splitting schemes, where the
//! difference between a Lie step and a Strang step provides a local truncation
//! error estimate. Also useful as a baseline for convergence comparisons.

use std::sync::Arc;

use super::super::{
    advecator::Advector,
    integrator::{StepProducts, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};
use crate::CausticError;

/// First-order Lie operator splitting: drift(Δt) followed by kick(Δt).
///
/// Not symplectic. Mainly used as a cheap error estimator for adaptive methods
/// or as a convergence baseline.
pub struct LieSplitting {
    /// Gravitational constant G used in the Poisson solve.
    pub g: f64,
    /// Optional lock-free progress reporter for TUI sub-step tracking.
    progress: Option<Arc<StepProgress>>,
}

impl LieSplitting {
    /// Creates a Lie splitting integrator with the given gravitational constant.
    pub fn new(g: f64) -> Self {
        Self { g, progress: None }
    }
}

impl TimeIntegrator for LieSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("lie_advance").entered();

        if let Some(ref p) = self.progress {
            p.start_step();
            p.set_phase(StepPhase::DriftHalf1);
            p.set_sub_step(0, 2);
        }

        // 1. Drift full step
        advector.drift(repr, dt);

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(1, 2);
        }

        // 2. Compute density → Poisson → acceleration → kick full step
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.g);
        let accel = solver.compute_acceleration(&potential);
        advector.kick(repr, &accel, dt);

        // Apply hypercollision damping if the representation is SpectralV
        if let Some(spectral) = repr
            .as_any_mut()
            .downcast_mut::<super::super::algos::spectral::SpectralV>()
        {
            spectral.apply_hypercollision(dt);
        }

        // Compute end-of-step products for caller reuse
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.g);
        let acceleration = solver.compute_acceleration(&potential);

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
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

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}
