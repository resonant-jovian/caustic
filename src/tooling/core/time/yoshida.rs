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
pub struct YoshidaSplitting {
    pub g: f64,
}

impl YoshidaSplitting {
    pub fn new(g: f64) -> Self {
        Self { g }
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

        // Optimized 3-substep form with 4 drifts and 3 kicks:
        //   drift(w1·dt/2) → kick(w1·dt) → drift((w1+w0)·dt/2) → kick(w0·dt)
        //   → drift((w0+w1)·dt/2) → kick(w1·dt) → drift(w1·dt/2)

        // Substep 1: drift w1·dt/2
        {
            let _s = tracing::info_span!("yoshida_drift_1").entered();
            advector.drift(repr, YOSHIDA_W1 * dt / 2.0);
        }

        // Substep 2: kick w1·dt
        {
            let _s = tracing::info_span!("yoshida_kick_1").entered();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            advector.kick(repr, &accel, YOSHIDA_W1 * dt);
        }

        // Substep 3: drift (w1+w0)·dt/2
        {
            let _s = tracing::info_span!("yoshida_drift_2").entered();
            advector.drift(repr, (YOSHIDA_W1 + YOSHIDA_W0) * dt / 2.0);
        }

        // Substep 4: kick w0·dt
        {
            let _s = tracing::info_span!("yoshida_kick_2").entered();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            advector.kick(repr, &accel, YOSHIDA_W0 * dt);
        }

        // Substep 5: drift (w0+w1)·dt/2
        {
            let _s = tracing::info_span!("yoshida_drift_3").entered();
            advector.drift(repr, (YOSHIDA_W0 + YOSHIDA_W1) * dt / 2.0);
        }

        // Substep 6: kick w1·dt
        {
            let _s = tracing::info_span!("yoshida_kick_3").entered();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            advector.kick(repr, &accel, YOSHIDA_W1 * dt);
        }

        // Substep 7: drift w1·dt/2
        {
            let _s = tracing::info_span!("yoshida_drift_4").entered();
            advector.drift(repr, YOSHIDA_W1 * dt / 2.0);
        }
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
}
