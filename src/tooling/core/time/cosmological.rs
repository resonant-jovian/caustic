//! Cosmological comoving-coordinate Strang splitting.
//!
//! Adapts second-order Strang splitting for an expanding universe by
//! incorporating scale-factor-dependent coefficients and Hubble drag.
//! The scale factor a(t) is evolved via Euler integration each step.
//!
//! Coordinate modifications relative to standard Strang splitting:
//! - Drift: displacement scaled by 1/a (comoving coordinates)
//! - Kick: acceleration scaled by a (momentum in comoving frame)
//! - Poisson: effective G_eff = G * a^2 for the comoving Poisson equation
//! - CFL: time step also limited by Hubble time 0.1/H(t)

use super::super::{
    context::SimContext,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::StepPhase,
    types::*,
};
use super::helpers;
use crate::CausticError;

/// Cosmological Strang splitting with scale-factor-dependent coefficients.
pub struct CosmologicalStrangSplitting {
    /// Current scale factor a(t).
    pub scale_factor: f64,
    /// Hubble parameter H(t).
    pub hubble: f64,
    /// Matter density parameter Omega_m.
    pub omega_m: f64,
    last_timings: StepTimings,
}

impl CosmologicalStrangSplitting {
    pub fn new(scale_factor: f64, hubble: f64, omega_m: f64) -> Self {
        Self {
            scale_factor,
            hubble,
            omega_m,
            last_timings: StepTimings::default(),
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
        ctx: &SimContext,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("cosmo_strang_advance").entered();
        let mut timings = StepTimings::default();
        let a = self.scale_factor;
        let dt = ctx.dt;

        ctx.progress.start_step();
        helpers::report_phase!(ctx, StepPhase::DriftHalf1, 0, 5);

        // Comoving drift: displacement scaled by 1/a
        helpers::time_ms!(timings, drift_ms, ctx.advector.drift(repr, &ctx.with_dt(dt / (2.0 * a))));

        helpers::report_phase!(ctx, StepPhase::PoissonSolve, 1, 5);

        // Poisson solve with cosmological factor (G_eff = G * a^2)
        let accel = helpers::time_ms!(timings, poisson_ms, {
            let g_eff = ctx.g * a * a;
            let cosmo_ctx = ctx.with_g(g_eff);
            let density = repr.compute_density();
            let potential = ctx.solver.solve(&density, &cosmo_ctx);
            ctx.solver.compute_acceleration(&potential)
        });

        helpers::report_phase!(ctx, StepPhase::Kick, 2, 5);

        // Kick with cosmological factor
        helpers::time_ms!(timings, kick_ms, ctx.advector.kick(repr, &accel, &ctx.with_dt(dt * a)));

        helpers::report_phase!(ctx, StepPhase::DriftHalf2, 3, 5);

        helpers::time_ms!(timings, drift_ms, ctx.advector.drift(repr, &ctx.with_dt(dt / (2.0 * a))));

        // Update scale factor via simple Euler step: a' = a * H * dt
        self.scale_factor += self.scale_factor * self.hubble * dt;

        helpers::report_phase!(ctx, StepPhase::StepComplete, 4, 5);

        // Compute end-of-step products for caller reuse (use updated scale factor)
        let (density, potential, acceleration) = helpers::time_ms!(timings, density_ms, {
            let g_eff = ctx.g * self.scale_factor * self.scale_factor;
            let cosmo_ctx = ctx.with_g(g_eff);
            let density = repr.compute_density();
            let potential = ctx.solver.solve(&density, &cosmo_ctx);
            let acceleration = ctx.solver.compute_acceleration(&potential);
            (density, potential, acceleration)
        });

        self.last_timings = timings;

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        let density = repr.compute_density();
        let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);
        if rho_max <= 0.0 {
            return 1e10;
        }
        // Use a default G=1.0 for the CFL estimate since we no longer store g
        let t_dyn = 1.0 / rho_max.sqrt();
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
}
