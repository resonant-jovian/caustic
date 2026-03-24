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

use std::sync::Arc;

use super::super::{
    advecator::Advector,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
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
pub struct YoshidaSplitting {
    /// Gravitational constant G used when solving the Poisson equation.
    pub g: f64,
    /// Timing breakdown from the most recent step (drift, kick, Poisson, density).
    last_timings: StepTimings,
    /// Optional progress reporter for intra-step phase tracking by the TUI.
    progress: Option<Arc<StepProgress>>,
}

impl YoshidaSplitting {
    /// Create a new Yoshida splitting integrator with the given gravitational constant.
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
            progress: None,
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
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("yoshida_advance").entered();
        let mut timings = StepTimings::default();

        if let Some(ref p) = self.progress {
            p.start_step();
        }

        // Optimized 3-substep form with 4 drifts and 3 kicks:
        //   drift(w1·dt/2) → kick(w1·dt) → drift((w1+w0)·dt/2) → kick(w0·dt)
        //   → drift((w0+w1)·dt/2) → kick(w1·dt) → drift(w1·dt/2)

        // Substep 1: drift w1·dt/2
        helpers::report_phase!(self.progress, StepPhase::YoshidaDrift1, 0, 7);
        {
            let _s = tracing::info_span!("yoshida_drift_1").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, YOSHIDA_W1 * dt / 2.0));
        }

        // Substep 2: kick w1·dt
        helpers::report_phase!(self.progress, StepPhase::YoshidaKick1, 1, 7);
        {
            let _s = tracing::info_span!("yoshida_kick_1").entered();
            let (_density, _potential, accel) =
                helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, solver, self.g));
            helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, YOSHIDA_W1 * dt));
        }

        // Apply hypercollision damping after kick 1
        helpers::apply_hypercollision_if_spectral(repr, YOSHIDA_W1 * dt);

        // Substep 3: drift (w1+w0)·dt/2
        helpers::report_phase!(self.progress, StepPhase::YoshidaDrift2, 2, 7);
        {
            let _s = tracing::info_span!("yoshida_drift_2").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, (YOSHIDA_W1 + YOSHIDA_W0) * dt / 2.0));
        }

        // Substep 4: kick w0·dt
        helpers::report_phase!(self.progress, StepPhase::YoshidaKick2, 3, 7);
        {
            let _s = tracing::info_span!("yoshida_kick_2").entered();
            let (_density, _potential, accel) =
                helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, solver, self.g));
            helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, YOSHIDA_W0 * dt));
        }

        // Apply hypercollision damping after kick 2
        helpers::apply_hypercollision_if_spectral(repr, YOSHIDA_W0 * dt);

        // Substep 5: drift (w0+w1)·dt/2
        helpers::report_phase!(self.progress, StepPhase::YoshidaDrift3, 4, 7);
        {
            let _s = tracing::info_span!("yoshida_drift_3").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, (YOSHIDA_W0 + YOSHIDA_W1) * dt / 2.0));
        }

        // Substep 6: kick w1·dt
        helpers::report_phase!(self.progress, StepPhase::YoshidaKick3, 5, 7);
        {
            let _s = tracing::info_span!("yoshida_kick_3").entered();
            let (_density, _potential, accel) =
                helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, solver, self.g));
            helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, YOSHIDA_W1 * dt));
        }

        // Apply hypercollision damping after kick 3
        helpers::apply_hypercollision_if_spectral(repr, YOSHIDA_W1 * dt);

        // Substep 7: drift w1·dt/2
        helpers::report_phase!(self.progress, StepPhase::YoshidaDrift4, 6, 7);
        {
            let _s = tracing::info_span!("yoshida_drift_4").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, YOSHIDA_W1 * dt / 2.0));
        }

        // Compute end-of-step products for caller reuse
        let (density, potential, acceleration) =
            helpers::time_ms!(timings, density_ms, helpers::solve_poisson(repr, solver, self.g));

        self.last_timings = timings;

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    /// Estimate the maximum stable time step from the dynamical time t_dyn = 1/sqrt(G*rho_max).
    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        helpers::dynamical_timestep(repr, self.g, cfl_factor)
    }

    /// Return the timing breakdown from the most recent step.
    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }

    /// Attach a shared progress reporter for intra-step phase tracking.
    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}
