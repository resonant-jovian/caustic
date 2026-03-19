//! Strang splitting time integrator. 2nd-order symmetric:
//! drift(Δt/2) → kick(Δt) → drift(Δt/2). Naturally symplectic.

use std::time::Instant;

use super::super::{
    advecator::Advector,
    integrator::{StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    solver::PoissonSolver,
    types::*,
};

/// Strang splitting: drift(Δt/2) → kick(Δt) → drift(Δt/2).
pub struct StrangSplitting {
    pub g: f64,
    last_timings: StepTimings,
}

impl StrangSplitting {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
        }
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
        let _span = tracing::info_span!("strang_advance").entered();
        let mut timings = StepTimings::default();

        {
            let _s = tracing::info_span!("drift_half").entered();
            let t0 = Instant::now();
            advector.drift(repr, dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        let accel = {
            let _s = tracing::info_span!("poisson_solve").entered();
            let t0 = Instant::now();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            accel
        };

        {
            let _s = tracing::info_span!("kick").entered();
            let t0 = Instant::now();
            advector.kick(repr, &accel, dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        {
            let _s = tracing::info_span!("drift_half").entered();
            let t0 = Instant::now();
            advector.drift(repr, dt / 2.0);
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
