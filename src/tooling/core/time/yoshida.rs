//! Yoshida fourth-order symplectic integrator for the Vlasov--Poisson system.
//!
//! Composes three Strang-like sub-steps with carefully chosen coefficients from
//! Yoshida (1990) to cancel the leading third-order error term, yielding fourth-order
//! global accuracy. The full step consists of 7 sub-steps (4 drifts + 3 kicks):
//!   drift(w1*dt/2) -> kick(w1*dt) -> drift((w1+w0)*dt/2) -> kick(w0*dt)
//!   -> drift((w0+w1)*dt/2) -> kick(w1*dt) -> drift(w1*dt/2)
//! where w1 = 1/(2 - 2^(1/3)) and w0 = 1 - 2*w1.
//!
//! More expensive per step than [`StrangSplitting`](super::strang::StrangSplitting)
//! (3 Poisson solves vs 1), but allows much larger time steps for the same error budget.
//! When the representation is `SpectralV`, hypercollision damping is applied after each kick.

use super::super::{
    context::SimContext,
    events::SimEvent,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::StepPhase,
    types::*,
};
use super::helpers;
use crate::CausticError;

/// Yoshida coefficient w1 = 1 / (2 - 2^(1/3)), the positive sub-step weight.
const YOSHIDA_W1: f64 = 1.3512071919596578;

/// Yoshida coefficient w0 = 1 - 2*w1, the negative sub-step weight (note: w0 < 0).
const YOSHIDA_W0: f64 = -1.7024143839193153;

/// Fourth-order symplectic time integrator via Yoshida splitting.
///
/// Composes 7 sub-steps (4 drifts, 3 kicks with Poisson solves) using the Yoshida (1990)
/// triple-jump coefficients to achieve O(dt^4) accuracy while preserving symplecticity.
#[derive(Default)]
pub struct YoshidaSplitting {
    /// Timing breakdown from the most recent step (drift, kick, Poisson, density).
    last_timings: StepTimings,
}

impl YoshidaSplitting {
    /// Create a new Yoshida splitting integrator.
    pub fn new() -> Self {
        Self {
            last_timings: StepTimings::default(),
        }
    }
}

impl TimeIntegrator for YoshidaSplitting {
    /// Advance the phase-space representation by one time step dt using Yoshida splitting.
    ///
    /// Executes the 7-sub-step sequence (4 drifts + 3 kick-with-Poisson-solve) and applies
    /// SpectralV hypercollision damping after each kick. Returns end-of-step products.
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        ctx: &SimContext,
    ) -> Result<StepProducts, CausticError> {
        let mut timings = StepTimings::default();
        let dt = ctx.dt;

        ctx.progress.start_step();

        // Optimized 3-substep form with 4 drifts and 3 kicks:
        //   drift(w1·dt/2) → kick(w1·dt) → drift((w1+w0)·dt/2) → kick(w0·dt)
        //   → drift((w0+w1)·dt/2) → kick(w1·dt) → drift(w1·dt/2)

        // Substep 1: drift w1·dt/2
        helpers::report_phase!(ctx, StepPhase::YoshidaDrift1, 0, 7);
        ctx.emitter.emit(SimEvent::PhaseEntered { phase: StepPhase::YoshidaDrift1, step: ctx.step });
        helpers::time_ms!(
            timings,
            drift_ms,
            ctx.advector.drift(repr, &ctx.with_dt(YOSHIDA_W1 * dt / 2.0))
        );

        // Substep 2: kick w1·dt
        helpers::report_phase!(ctx, StepPhase::YoshidaKick1, 1, 7);
        ctx.emitter.emit(SimEvent::PhaseEntered { phase: StepPhase::YoshidaKick1, step: ctx.step });
        let (_density, _potential, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(repr, ctx)
        );
        helpers::time_ms!(
            timings,
            kick_ms,
            ctx.advector.kick(repr, &accel, &ctx.with_dt(YOSHIDA_W1 * dt))
        );

        // Apply hypercollision damping after kick 1
        helpers::apply_hypercollision_if_spectral(repr, YOSHIDA_W1 * dt);

        // Substep 3: drift (w1+w0)·dt/2
        helpers::report_phase!(ctx, StepPhase::YoshidaDrift2, 2, 7);
        ctx.emitter.emit(SimEvent::PhaseEntered { phase: StepPhase::YoshidaDrift2, step: ctx.step });
        helpers::time_ms!(
            timings,
            drift_ms,
            ctx.advector.drift(repr, &ctx.with_dt((YOSHIDA_W1 + YOSHIDA_W0) * dt / 2.0))
        );

        // Substep 4: kick w0·dt
        helpers::report_phase!(ctx, StepPhase::YoshidaKick2, 3, 7);
        ctx.emitter.emit(SimEvent::PhaseEntered { phase: StepPhase::YoshidaKick2, step: ctx.step });
        let (_density, _potential, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(repr, ctx)
        );
        helpers::time_ms!(
            timings,
            kick_ms,
            ctx.advector.kick(repr, &accel, &ctx.with_dt(YOSHIDA_W0 * dt))
        );

        // Apply hypercollision damping after kick 2
        helpers::apply_hypercollision_if_spectral(repr, YOSHIDA_W0 * dt);

        // Substep 5: drift (w0+w1)·dt/2
        helpers::report_phase!(ctx, StepPhase::YoshidaDrift3, 4, 7);
        ctx.emitter.emit(SimEvent::PhaseEntered { phase: StepPhase::YoshidaDrift3, step: ctx.step });
        helpers::time_ms!(
            timings,
            drift_ms,
            ctx.advector.drift(repr, &ctx.with_dt((YOSHIDA_W0 + YOSHIDA_W1) * dt / 2.0))
        );

        // Substep 6: kick w1·dt
        helpers::report_phase!(ctx, StepPhase::YoshidaKick3, 5, 7);
        ctx.emitter.emit(SimEvent::PhaseEntered { phase: StepPhase::YoshidaKick3, step: ctx.step });
        let (_density, _potential, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(repr, ctx)
        );
        helpers::time_ms!(
            timings,
            kick_ms,
            ctx.advector.kick(repr, &accel, &ctx.with_dt(YOSHIDA_W1 * dt))
        );

        // Apply hypercollision damping after kick 3
        helpers::apply_hypercollision_if_spectral(repr, YOSHIDA_W1 * dt);

        // Substep 7: drift w1·dt/2
        helpers::report_phase!(ctx, StepPhase::YoshidaDrift4, 6, 7);
        ctx.emitter.emit(SimEvent::PhaseEntered { phase: StepPhase::YoshidaDrift4, step: ctx.step });
        helpers::time_ms!(
            timings,
            drift_ms,
            ctx.advector.drift(repr, &ctx.with_dt(YOSHIDA_W1 * dt / 2.0))
        );

        // Compute end-of-step products for caller reuse
        let (density, potential, acceleration) = helpers::time_ms!(
            timings,
            density_ms,
            helpers::solve_poisson(repr, ctx)
        );

        self.last_timings = timings;

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    /// Estimate the maximum stable time step from the dynamical time t_dyn = 1/sqrt(G*rho_max).
    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        helpers::dynamical_timestep(repr, 1.0, cfl_factor)
    }

    /// Return the timing breakdown from the most recent step.
    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }
}
