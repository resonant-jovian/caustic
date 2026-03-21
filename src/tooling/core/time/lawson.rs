//! Lawson-RK integrator: exact free-streaming + RK4 gravity kick.
//!
//! Removes the linear free-streaming CFL constraint by solving
//! ∂f/∂t + v·∇ₓf = 0 via semi-Lagrangian advection (stable at any
//! Courant number), then applies 4th-order Runge-Kutta sub-stepping
//! to the nonlinear gravity term with 4 Poisson solves per step.
//!
//! The effective CFL constraint is determined solely by the velocity-space
//! advection: a_max · dt / dv, typically less restrictive than the spatial
//! CFL v_max · dt / dx that limits standard Strang splitting.
//!
//! Best suited for `UniformGrid6D` and `SpectralV` representations.
//! Not recommended for HT (use BUG-based integrators instead).
//!
//! Structure (Strang-Lawson):
//!   1. Exact drift dt/2
//!   2. RK4 kick (4 Poisson solves)
//!   3. Exact drift dt/2

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

/// Lawson-RK integrator: exact drift + RK4 gravity kick.
pub struct LawsonRkIntegrator {
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl LawsonRkIntegrator {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
            progress: None,
        }
    }

    /// Compute acceleration field from the current distribution.
    fn compute_accel(
        repr: &dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        g: f64,
    ) -> AccelerationField {
        let density = repr.compute_density();
        let potential = solver.solve(&density, g);
        solver.compute_acceleration(&potential)
    }
}

impl TimeIntegrator for LawsonRkIntegrator {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        let _span = tracing::info_span!("lawson_rk_advance").entered();
        let mut timings = StepTimings::default();

        if let Some(ref p) = self.progress {
            p.start_step();
            p.set_phase(StepPhase::DriftHalf1);
            p.set_sub_step(0, 7);
        }

        // Step 1: Exact half-drift (free-streaming, no CFL restriction)
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Save state after half-drift for RK4 stage resets
        let snap_after_drift = repr.to_snapshot(0.0);

        // Step 2: RK4 kick with 4 Poisson solves
        // Stage 1: evaluate acceleration at drifted state
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::PoissonSolve);
            p.set_sub_step(1, 7);
        }
        let t0 = Instant::now();
        let a1 = Self::compute_accel(repr, solver, self.g);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Stage 2: half-kick with a1, evaluate acceleration
        if let Some(ref p) = self.progress {
            p.set_sub_step(2, 7);
        }
        let t0 = Instant::now();
        advector.kick(repr, &a1, dt / 2.0);
        let a2 = Self::compute_accel(repr, solver, self.g);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Stage 3: restore, half-kick with a2, evaluate acceleration
        if let Some(ref p) = self.progress {
            p.set_sub_step(3, 7);
        }
        let t0 = Instant::now();
        let PhaseSpaceSnapshot { data, shape, time } = snap_after_drift;
        repr.load_snapshot(PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time,
        });
        advector.kick(repr, &a2, dt / 2.0);
        let a3 = Self::compute_accel(repr, solver, self.g);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Stage 4: restore, full-kick with a3, evaluate acceleration
        if let Some(ref p) = self.progress {
            p.set_sub_step(4, 7);
        }
        let t0 = Instant::now();
        repr.load_snapshot(PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time,
        });
        advector.kick(repr, &a3, dt);
        let a4 = Self::compute_accel(repr, solver, self.g);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Apply RK4-weighted kick: (a1 + 2*a2 + 2*a3 + a4) / 6 * dt
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(5, 7);
        }
        let t0 = Instant::now();
        repr.load_snapshot(PhaseSpaceSnapshot { data, shape, time });
        advector.kick(repr, &a1, dt / 6.0);
        advector.kick(repr, &a2, dt / 3.0);
        advector.kick(repr, &a3, dt / 3.0);
        advector.kick(repr, &a4, dt / 6.0);
        timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Exact half-drift
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::DriftHalf2);
            p.set_sub_step(6, 7);
        }
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
        }
        self.last_timings = timings;
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        // Lawson-RK: only velocity-space CFL matters (dynamical time).
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

    #[test]
    fn lawson_rk_construction() {
        let integrator = LawsonRkIntegrator::new(1.0);
        assert_eq!(integrator.g, 1.0);
    }
}
