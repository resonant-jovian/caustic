//! Yoshida 4th-order symplectic integrator. Uses 3 Strang-like sub-steps with
//! specific coefficients w0, w1.

use std::sync::Arc;
use std::time::Instant;

use super::super::{
    advecator::Advector,
    integrator::{StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};

/// Yoshida coefficient w1 = 1 / (2 − 2^(1/3)).
const YOSHIDA_W1: f64 = 1.3512071919596578;

/// Yoshida coefficient w0 = 1 − 2·w1.
const YOSHIDA_W0: f64 = -1.7024143839193153;

/// Yoshida 4th-order symplectic integrator.
pub struct YoshidaSplitting {
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl YoshidaSplitting {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
            progress: None,
        }
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

        self.last_timings = timings;
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

    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}
