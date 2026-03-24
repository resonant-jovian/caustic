//! Lawson-RK integrator: exact free-streaming propagation with RK4 gravity kicks.
//!
//! The free-streaming (drift) part of the Vlasov equation is solved exactly
//! via semi-Lagrangian advection, which is unconditionally stable regardless
//! of the spatial Courant number. This eliminates the often-restrictive
//! spatial CFL constraint v_max * dt / dx that limits standard Strang
//! splitting. The remaining nonlinear gravity term is then integrated with
//! classical 4th-order Runge-Kutta, requiring 4 Poisson solves per step.
//!
//! The only CFL restriction comes from velocity-space advection:
//! a_max * dt / dv, which is typically much less restrictive.
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
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};
use super::helpers;
use crate::CausticError;

/// Lawson-RK integrator: exact drift + RK4 gravity kick.
///
/// Wraps each step in a Strang-style drift/2 -- RK4 kick -- drift/2 sequence,
/// where the drift is solved exactly and the kick uses four Poisson solves.
pub struct LawsonRkIntegrator {
    /// Gravitational constant G used in Poisson solves.
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl LawsonRkIntegrator {
    /// Create a Lawson-RK integrator with the given gravitational constant.
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
    ) -> Result<StepProducts, CausticError> {
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
        let snap_after_drift = repr.to_snapshot(0.0).ok_or_else(|| {
            CausticError::Solver("Lawson-RK integrator requires to_snapshot support".into())
        })?;

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
        })?;
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
        })?;
        advector.kick(repr, &a3, dt);
        let a4 = Self::compute_accel(repr, solver, self.g);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Apply RK4-weighted kick: (a1 + 2*a2 + 2*a3 + a4) / 6 * dt
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(5, 7);
        }
        let t0 = Instant::now();
        repr.load_snapshot(PhaseSpaceSnapshot { data, shape, time })?;
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

        let t0 = Instant::now();
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.g);
        let acceleration = solver.compute_acceleration(&potential);
        timings.density_ms += t0.elapsed().as_secs_f64() * 1000.0;

        self.last_timings = timings;

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        // Lawson-RK: only velocity-space CFL matters (dynamical time).
        helpers::dynamical_timestep(repr, self.g, cfl_factor)
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
