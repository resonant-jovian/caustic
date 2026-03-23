//! Cosmological comoving-coordinate Strang splitting.
//!
//! Wraps StrangSplitting with scale-factor-dependent coefficients for
//! cosmological simulations in an expanding universe:
//! - Drift: v*dt/(a*m) instead of v*dt
//! - Kick: acceleration *= a*m
//! - Poisson: 4*pi*G*a^2*rho_bar*delta

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

/// Cosmological Strang splitting with scale-factor-dependent coefficients.
pub struct CosmologicalStrangSplitting {
    pub g: f64,
    /// Current scale factor a(t).
    pub scale_factor: f64,
    /// Hubble parameter H(t).
    pub hubble: f64,
    /// Matter density parameter Omega_m.
    pub omega_m: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl CosmologicalStrangSplitting {
    pub fn new(g: f64, scale_factor: f64, hubble: f64, omega_m: f64) -> Self {
        Self {
            g,
            scale_factor,
            hubble,
            omega_m,
            last_timings: StepTimings::default(),
            progress: None,
        }
    }

    /// Update the scale factor and Hubble parameter for the current time.
    pub fn set_cosmology(&mut self, scale_factor: f64, hubble: f64) {
        self.scale_factor = scale_factor;
        self.hubble = hubble;
    }
}

impl TimeIntegrator for CosmologicalStrangSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("cosmo_strang_advance").entered();
        let mut timings = StepTimings::default();
        let a = self.scale_factor;

        if let Some(ref p) = self.progress {
            p.start_step();
            p.set_phase(StepPhase::DriftHalf1);
            p.set_sub_step(0, 5);
        }

        // Comoving drift: displacement scaled by 1/a
        {
            let t0 = Instant::now();
            advector.drift(repr, dt / (2.0 * a));
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::PoissonSolve);
            p.set_sub_step(1, 5);
        }

        // Poisson solve with cosmological factor
        let accel = {
            let t0 = Instant::now();
            let density = repr.compute_density();
            // Solve with effective G_eff = G * a^2 for comoving Poisson equation
            let g_eff = self.g * a * a;
            let potential = solver.solve(&density, g_eff);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            accel
        };

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(2, 5);
        }

        // Kick with cosmological factor
        {
            let t0 = Instant::now();
            advector.kick(repr, &accel, dt * a);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::DriftHalf2);
            p.set_sub_step(3, 5);
        }

        {
            let t0 = Instant::now();
            advector.drift(repr, dt / (2.0 * a));
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Update scale factor via simple Euler step: a' = a * H * dt
        self.scale_factor += self.scale_factor * self.hubble * dt;

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
            p.set_sub_step(4, 5);
        }

        // Compute end-of-step products for caller reuse (use updated scale factor)
        let t0 = Instant::now();
        let density = repr.compute_density();
        let g_eff = self.g * self.scale_factor * self.scale_factor;
        let potential = solver.solve(&density, g_eff);
        let acceleration = solver.compute_acceleration(&potential);
        timings.density_ms += t0.elapsed().as_secs_f64() * 1000.0;

        self.last_timings = timings;

        Ok(StepProducts { density, potential, acceleration })
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        let density = repr.compute_density();
        let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);
        if rho_max <= 0.0 || self.g <= 0.0 {
            return 1e10;
        }
        let t_dyn = 1.0 / (self.g * rho_max).sqrt();
        // Also limit by Hubble time
        let t_hubble = if self.hubble.abs() > 1e-30 {
            0.1 / self.hubble.abs()
        } else {
            1e10
        };
        cfl_factor * t_dyn.min(t_hubble)
    }

    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}
