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
use std::time::Instant;

use super::super::{
    advecator::Advector,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};
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
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::YoshidaDrift1);
            p.set_sub_step(0, 7);
        }
        {
            let _s = tracing::info_span!("yoshida_drift_1").entered();
            let t0 = Instant::now();
            advector.drift(repr, YOSHIDA_W1 * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Substep 2: kick w1·dt
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::YoshidaKick1);
            p.set_sub_step(1, 7);
        }
        {
            let _s = tracing::info_span!("yoshida_kick_1").entered();
            let t0 = Instant::now();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            advector.kick(repr, &accel, YOSHIDA_W1 * dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Apply hypercollision damping after kick 1
        if let Some(spectral) = repr
            .as_any_mut()
            .downcast_mut::<super::super::algos::spectral::SpectralV>()
        {
            spectral.apply_hypercollision(YOSHIDA_W1 * dt);
        }

        // Substep 3: drift (w1+w0)·dt/2
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::YoshidaDrift2);
            p.set_sub_step(2, 7);
        }
        {
            let _s = tracing::info_span!("yoshida_drift_2").entered();
            let t0 = Instant::now();
            advector.drift(repr, (YOSHIDA_W1 + YOSHIDA_W0) * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Substep 4: kick w0·dt
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::YoshidaKick2);
            p.set_sub_step(3, 7);
        }
        {
            let _s = tracing::info_span!("yoshida_kick_2").entered();
            let t0 = Instant::now();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            advector.kick(repr, &accel, YOSHIDA_W0 * dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Apply hypercollision damping after kick 2
        if let Some(spectral) = repr
            .as_any_mut()
            .downcast_mut::<super::super::algos::spectral::SpectralV>()
        {
            spectral.apply_hypercollision(YOSHIDA_W0 * dt);
        }

        // Substep 5: drift (w0+w1)·dt/2
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::YoshidaDrift3);
            p.set_sub_step(4, 7);
        }
        {
            let _s = tracing::info_span!("yoshida_drift_3").entered();
            let t0 = Instant::now();
            advector.drift(repr, (YOSHIDA_W0 + YOSHIDA_W1) * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Substep 6: kick w1·dt
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::YoshidaKick3);
            p.set_sub_step(5, 7);
        }
        {
            let _s = tracing::info_span!("yoshida_kick_3").entered();
            let t0 = Instant::now();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            advector.kick(repr, &accel, YOSHIDA_W1 * dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Apply hypercollision damping after kick 3
        if let Some(spectral) = repr
            .as_any_mut()
            .downcast_mut::<super::super::algos::spectral::SpectralV>()
        {
            spectral.apply_hypercollision(YOSHIDA_W1 * dt);
        }

        // Substep 7: drift w1·dt/2
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::YoshidaDrift4);
            p.set_sub_step(6, 7);
        }
        {
            let _s = tracing::info_span!("yoshida_drift_4").entered();
            let t0 = Instant::now();
            advector.drift(repr, YOSHIDA_W1 * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Compute end-of-step products for caller reuse
        let t0 = Instant::now();
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.g);
        let acceleration = solver.compute_acceleration(&potential);
        timings.density_ms += t0.elapsed().as_secs_f64() * 1000.0;

        self.last_timings = timings;

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    /// Estimate the maximum stable time step from the dynamical time t_dyn = 1/sqrt(G*rho_max).
    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        let density = repr.compute_density();
        let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);
        if rho_max <= 0.0 || self.g <= 0.0 {
            return 1e10;
        }
        let t_dyn = 1.0 / (self.g * rho_max).sqrt();
        cfl_factor * t_dyn
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
