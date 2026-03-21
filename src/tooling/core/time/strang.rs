//! Strang splitting time integrator. 2nd-order symmetric:
//! drift(Δt/2) → kick(Δt) → drift(Δt/2). Naturally symplectic.

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

/// Strang splitting: drift(Δt/2) → kick(Δt) → drift(Δt/2).
pub struct StrangSplitting {
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl StrangSplitting {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
            progress: None,
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

        if let Some(ref p) = self.progress {
            p.start_step();
            p.set_phase(StepPhase::DriftHalf1);
            p.set_sub_step(0, 5);
        }

        {
            let _s = tracing::info_span!("drift_half").entered();
            let t0 = Instant::now();
            advector.drift(repr, dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::PoissonSolve);
            p.set_sub_step(1, 5);
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

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(2, 5);
        }

        {
            let _s = tracing::info_span!("kick").entered();
            let t0 = Instant::now();
            advector.kick(repr, &accel, dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Apply hypercollision damping if the representation is SpectralV
        if let Some(spectral) = repr
            .as_any_mut()
            .downcast_mut::<super::super::algos::spectral::SpectralV>()
        {
            spectral.apply_hypercollision(dt);
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::DriftHalf2);
            p.set_sub_step(3, 5);
        }

        {
            let _s = tracing::info_span!("drift_half").entered();
            let t0 = Instant::now();
            advector.drift(repr, dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
            p.set_sub_step(4, 5);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::spectral::SpectralV;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::poisson::fft::FftPoisson;

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
        let mut integrator = StrangSplitting::new(1.0);

        integrator.advance(&mut spec, &poisson, &advector, 0.1);

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
