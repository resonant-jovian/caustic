//! Strang splitting time integrator for the Vlasov--Poisson system.
//!
//! Applies second-order symmetric operator splitting:
//!   drift(dt/2) -> kick(dt) -> drift(dt/2)
//! where "drift" is spatial advection (v * grad_x f = 0) and "kick" is velocity
//! advection (grad_Phi * grad_v f = 0) after solving the Poisson equation for Phi.
//!
//! The splitting is time-reversible and symplectic, making it the default integrator
//! for most simulations. For higher accuracy use [`YoshidaSplitting`](super::yoshida::YoshidaSplitting).
//! When the representation is `SpectralV`, hypercollision damping is applied after the kick.

use super::super::{
    context::SimContext,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::StepPhase,
    types::*,
};
use super::helpers;
use crate::CausticError;

/// Second-order symplectic time integrator via Strang splitting.
///
/// Executes drift(dt/2) -> Poisson solve -> kick(dt) -> drift(dt/2) each step.
/// After the kick, applies hypercollision damping when the representation is `SpectralV`.
pub struct StrangSplitting {
    /// Timing breakdown from the most recent step (drift, kick, Poisson, density).
    last_timings: StepTimings,
}

impl StrangSplitting {
    /// Create a new Strang splitting integrator.
    pub fn new() -> Self {
        Self {
            last_timings: StepTimings::default(),
        }
    }
}

impl TimeIntegrator for StrangSplitting {
    /// Advance the phase-space representation by one time step dt using Strang splitting.
    ///
    /// Sequence: drift(dt/2) -> Poisson solve -> kick(dt) -> \[hypercollision\] -> drift(dt/2).
    /// Returns the end-of-step density, potential, and acceleration for reuse by diagnostics.
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        ctx: &SimContext,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("strang_advance").entered();
        let mut timings = StepTimings::default();
        let dt = ctx.dt;

        ctx.progress.start_step();
        helpers::report_phase!(ctx, StepPhase::DriftHalf1, 0, 5);

        {
            let _s = tracing::info_span!("drift_half").entered();
            helpers::time_ms!(timings, drift_ms, ctx.advector.drift(repr, &ctx.with_dt(dt / 2.0)));
        }

        helpers::report_phase!(ctx, StepPhase::PoissonSolve, 1, 5);

        let accel = {
            let _s = tracing::info_span!("poisson_solve").entered();
            let (density, potential, accel) = helpers::time_ms!(
                timings,
                poisson_ms,
                helpers::solve_poisson(repr, ctx)
            );
            let _ = (density, potential);
            accel
        };

        helpers::report_phase!(ctx, StepPhase::Kick, 2, 5);

        {
            let _s = tracing::info_span!("kick").entered();
            helpers::time_ms!(timings, kick_ms, ctx.advector.kick(repr, &accel, &ctx.with_dt(dt)));
        }

        // Apply hypercollision damping if the representation is SpectralV
        helpers::apply_hypercollision_if_spectral(repr, dt);

        helpers::report_phase!(ctx, StepPhase::DriftHalf2, 3, 5);

        {
            let _s = tracing::info_span!("drift_half").entered();
            helpers::time_ms!(timings, drift_ms, ctx.advector.drift(repr, &ctx.with_dt(dt / 2.0)));
        }

        helpers::report_phase!(ctx, StepPhase::StepComplete, 4, 5);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::spectral::SpectralV;
    use crate::tooling::core::events::EventEmitter;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::progress::StepProgress;

    #[test]
    fn test_hypercollision_applied_for_spectral_v() {
        // Create a SpectralV with high-mode energy and nonzero hypercollision_nu,
        // run one Strang step, and verify that high modes are damped.
        let domain = Domain::builder()
            .spatial_extent(2.0)
            .velocity_extent(3.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let n_modes = 4;
        let n_modes3 = n_modes * n_modes * n_modes;
        let mut spec = SpectralV::new(domain.clone(), n_modes);

        // Set a zeroth mode for nonzero density and high modes with energy
        for si in 0..(4 * 4 * 4) {
            let base = si * n_modes3;
            spec.coefficients[base] = 1.0; // a_{0,0,0}
            // Set high mode: mode (3,3,3)
            let high = 3 * n_modes * n_modes + 3 * n_modes + 3;
            spec.coefficients[base + high] = 0.5;
        }

        // Enable hypercollision
        spec.hypercollision_nu = 1.0;
        spec.hypercollision_order = 2;

        // Record the high-mode energy before stepping
        let high_idx = 3 * n_modes * n_modes + 3 * n_modes + 3;
        let high_mode_before: f64 = (0..(4 * 4 * 4))
            .map(|si| spec.coefficients[si * n_modes3 + high_idx].abs())
            .sum::<f64>();

        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let mut integrator = StrangSplitting::new();

        let ctx = SimContext {
            solver: &poisson,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.1,
            g: 1.0,
        };

        integrator
            .advance(&mut spec, &ctx)
            .unwrap();

        // The high mode (3,3,3) should be significantly damped.
        // Damping factor for mode (3,3,3) with nu=1.0, order=2, dt=0.1:
        //   exp(-1.0 * 0.1 * (3^4 + 3^4 + 3^4)) = exp(-24.3) which is essentially 0.
        let high_mode_after: f64 = (0..(4 * 4 * 4))
            .map(|si| spec.coefficients[si * n_modes3 + high_idx].abs())
            .sum::<f64>();

        assert!(
            high_mode_after < high_mode_before * 0.01,
            "High modes should be strongly damped by hypercollision: before={}, after={}",
            high_mode_before,
            high_mode_after
        );

        // The zeroth mode should be mostly preserved (mode 0 has n^4 = 0, so
        // the damping factor is exp(0) = 1).
        let zeroth_mode_after: f64 = (0..(4 * 4 * 4))
            .map(|si| spec.coefficients[si * n_modes3].abs())
            .sum::<f64>();
        let zeroth_mode_before = 64.0; // 4^3 cells * 1.0

        assert!(
            zeroth_mode_after > zeroth_mode_before * 0.5,
            "Zeroth mode should be approximately preserved: before={}, after={}",
            zeroth_mode_before,
            zeroth_mode_after
        );
    }
}
