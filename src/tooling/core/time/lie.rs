//! Lie (first-order) operator splitting: drift(Δt) → kick(Δt).
//!
//! Applies spatial drift and velocity kick sequentially without the symmetric
//! half-step structure of Strang splitting. This yields only 1st-order accuracy
//! and does not preserve symplecticity, so it is not suitable for production runs.
//!
//! Primary use: as an error estimator in adaptive splitting schemes, where the
//! difference between a Lie step and a Strang step provides a local truncation
//! error estimate. Also useful as a baseline for convergence comparisons.

use super::super::{
    context::SimContext,
    integrator::{StepProducts, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::StepPhase,
    types::*,
};
use super::helpers;
use crate::CausticError;

/// First-order Lie operator splitting: drift(Δt) followed by kick(Δt).
///
/// Not symplectic. Mainly used as a cheap error estimator for adaptive methods
/// or as a convergence baseline.
pub struct LieSplitting;

impl LieSplitting {
    /// Creates a Lie splitting integrator.
    pub fn new() -> Self {
        Self
    }
}

impl TimeIntegrator for LieSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        ctx: &SimContext,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("lie_advance").entered();
        let dt = ctx.dt;

        ctx.progress.start_step();
        ctx.progress.set_phase(StepPhase::DriftHalf1);
        ctx.progress.set_sub_step(0, 2);

        // 1. Drift full step
        ctx.advector.drift(repr, &ctx.with_dt(dt));

        ctx.progress.set_phase(StepPhase::Kick);
        ctx.progress.set_sub_step(1, 2);

        // 2. Compute density → Poisson → acceleration → kick full step
        let density = repr.compute_density();
        let potential = ctx.solver.solve(&density, ctx);
        let accel = ctx.solver.compute_acceleration(&potential);
        ctx.advector.kick(repr, &accel, &ctx.with_dt(dt));

        // Apply hypercollision damping if the representation is SpectralV
        helpers::apply_hypercollision_if_spectral(repr, dt);

        // Compute end-of-step products for caller reuse
        let density = repr.compute_density();
        let potential = ctx.solver.solve(&density, ctx);
        let acceleration = ctx.solver.compute_acceleration(&potential);

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        helpers::dynamical_timestep(repr, 1.0, cfl_factor)
    }
}
