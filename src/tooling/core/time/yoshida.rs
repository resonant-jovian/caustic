//! Yoshida 4th-order symplectic integrator. Uses 3 Strang-like sub-steps with
//! specific coefficients w0, w1.

use std::time::Instant;

use super::super::{
    advecator::Advector,
    integrator::{StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
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
}

impl YoshidaSplitting {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
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

        // Optimized 3-substep form with 4 drifts and 3 kicks:
        //   drift(w1·dt/2) → kick(w1·dt) → drift((w1+w0)·dt/2) → kick(w0·dt)
        //   → drift((w0+w1)·dt/2) → kick(w1·dt) → drift(w1·dt/2)

        // Substep 1: drift w1·dt/2
        {
            let _s = tracing::info_span!("yoshida_drift_1").entered();
            let t0 = Instant::now();
            advector.drift(repr, YOSHIDA_W1 * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Substep 2: kick w1·dt
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

        // Substep 3: drift (w1+w0)·dt/2
        {
            let _s = tracing::info_span!("yoshida_drift_2").entered();
            let t0 = Instant::now();
            advector.drift(repr, (YOSHIDA_W1 + YOSHIDA_W0) * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Substep 4: kick w0·dt
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

        // Substep 5: drift (w0+w1)·dt/2
        {
            let _s = tracing::info_span!("yoshida_drift_3").entered();
            let t0 = Instant::now();
            advector.drift(repr, (YOSHIDA_W0 + YOSHIDA_W1) * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Substep 6: kick w1·dt
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

        // Substep 7: drift w1·dt/2
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
}
