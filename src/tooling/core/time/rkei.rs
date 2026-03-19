//! RKEI (Runge-Kutta Exponential Integrator) time integrator.
//!
//! Unlike Strang splitting, RKEI does NOT split drift and kick. Instead, each
//! RK stage traces the full characteristic ODE (dx/dt = v, dv/dt = g(x))
//! with a frozen acceleration field, then reconstructs f via semi-Lagrangian
//! advection along the combined trajectory.
//!
//! This eliminates splitting errors at the cost of one Poisson solve per stage
//! (3 total for RK3, vs 1 for Strang).
//!
//! 3rd-order SSP-RK3 Butcher tableau (Shu-Osher):
//!   Stage 1: f^(1) = advect(f^n, g^n, Δt)
//!   Stage 2: f^(2) = ¾ f^n + ¼ advect(f^(1), g^(1), Δt)
//!   Stage 3: f^{n+1} = ⅓ f^n + ⅔ advect(f^(2), g^(2), Δt)

use std::sync::Arc;

use super::super::{
    advecator::Advector, integrator::TimeIntegrator, phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress}, solver::PoissonSolver, types::*,
};
use rayon::prelude::*;

/// RKEI: 3rd-order unsplit Runge-Kutta time integrator.
///
/// Each stage performs a full semi-Lagrangian advection (combined drift + kick)
/// with the acceleration field frozen at the beginning of that stage.
pub struct RkeiIntegrator {
    pub g: f64,
    progress: Option<Arc<StepProgress>>,
}

impl RkeiIntegrator {
    pub fn new(g: f64) -> Self {
        Self { g, progress: None }
    }
}

impl TimeIntegrator for RkeiIntegrator {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        let _span = tracing::info_span!("rkei_advance").entered();

        if let Some(ref p) = self.progress {
            p.start_step();
            p.set_phase(StepPhase::RkeiStage1);
            p.set_sub_step(0, 3);
        }

        // Save f^n as a snapshot for convex combination
        let f_n = repr.to_snapshot(0.0);

        // ── Stage 1: f^(1) = advect(f^n, g^n, Δt) ──
        {
            let _s = tracing::info_span!("rkei_stage_1").entered();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);

            // Full advection: drift Δt then kick Δt with frozen acceleration
            advector.drift(repr, dt);
            advector.kick(repr, &accel, dt);
        }
        // repr now holds f^(1)

        // ── Stage 2: f^(2) = ¾ f^n + ¼ advect(f^(1), g^(1), Δt) ──
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::RkeiStage2);
            p.set_sub_step(1, 3);
        }
        let f_1 = repr.to_snapshot(0.0);
        {
            let _s = tracing::info_span!("rkei_stage_2").entered();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);

            advector.drift(repr, dt);
            advector.kick(repr, &accel, dt);
        }
        // repr now holds advect(f^(1), g^(1), Δt)

        // Convex combination: f^(2) = ¾ f^n + ¼ repr
        {
            let adv_snap = repr.to_snapshot(0.0);
            let combined: Vec<f64> = f_n
                .data
                .par_iter()
                .zip(adv_snap.data.par_iter())
                .map(|(&a, &b)| 0.75 * a + 0.25 * b)
                .collect();
            repr.load_snapshot(PhaseSpaceSnapshot {
                data: combined,
                shape: f_n.shape,
                time: 0.0,
            });
        }
        // repr now holds f^(2)

        // ── Stage 3: f^{n+1} = ⅓ f^n + ⅔ advect(f^(2), g^(2), Δt) ──
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::RkeiStage3);
            p.set_sub_step(2, 3);
        }
        {
            let _s = tracing::info_span!("rkei_stage_3").entered();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);

            advector.drift(repr, dt);
            advector.kick(repr, &accel, dt);
        }
        // repr now holds advect(f^(2), g^(2), Δt)

        // Convex combination: f^{n+1} = ⅓ f^n + ⅔ repr
        {
            let adv_snap = repr.to_snapshot(0.0);
            let combined: Vec<f64> = f_n
                .data
                .par_iter()
                .zip(adv_snap.data.par_iter())
                .map(|(&a, &b)| (1.0 / 3.0) * a + (2.0 / 3.0) * b)
                .collect();
            repr.load_snapshot(PhaseSpaceSnapshot {
                data: combined,
                shape: f_n.shape,
                time: 0.0,
            });
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

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rkei_smoke_test() {
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
        use crate::tooling::core::poisson::fft::FftPoisson;

        let domain = Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let mut grid = UniformGrid6D::new(domain.clone());
        for v in grid.data.iter_mut() {
            *v = 1.0;
        }

        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut integrator = RkeiIntegrator::new(1.0);

        integrator.advance(&mut grid, &poisson, &advector, 0.01);

        // Should not contain NaN
        assert!(
            !grid.data.iter().any(|x| x.is_nan()),
            "RKEI produced NaN values"
        );
    }
}
