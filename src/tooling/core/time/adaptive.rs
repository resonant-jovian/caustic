//! Adaptive time stepping via embedded error estimation and PI control.
//!
//! Each step runs both a Strang (2nd-order) and a Lie (1st-order) split with
//! the same timestep. The relative L2 difference between the two results
//! serves as a local truncation error estimate. A PI controller then adjusts
//! the next timestep to keep this error near a user-specified tolerance.
//!
//! Steps whose error exceeds the tolerance are rejected: the state is rolled
//! back to the initial snapshot and retried with a smaller dt (up to
//! `max_retries` attempts). The accepted result is always the higher-order
//! Strang solution.

use std::sync::Arc;

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

/// PI controller for adaptive step-size selection.
pub struct PIController {
    /// Target relative error tolerance.
    pub tolerance: f64,
    /// Integral gain (typically ~0.3/p where p is the method order).
    pub k_i: f64,
    /// Proportional gain (typically ~0.4/p).
    pub k_p: f64,
    /// Minimum allowed timestep.
    pub dt_min: f64,
    /// Maximum allowed timestep.
    pub dt_max: f64,
    /// Safety factor applied to the step-size estimate (~0.9).
    pub safety_factor: f64,
    /// Error from the previous accepted step.
    prev_err: Option<f64>,
}

impl PIController {
    /// Create a PI controller for a method of the given `order` and error `tolerance`.
    ///
    /// Gains `k_i` and `k_p` are set to standard values (0.3/p and 0.4/p).
    pub fn new(tolerance: f64, order: f64) -> Self {
        Self {
            tolerance,
            k_i: 0.3 / order,
            k_p: 0.4 / order,
            dt_min: 1e-15,
            dt_max: 1e10,
            safety_factor: 0.9,
            prev_err: None,
        }
    }

    /// Compute the next timestep from the current error estimate.
    /// Returns (dt_new, accepted) where accepted=false means reject and retry.
    pub fn step(&mut self, dt: f64, err: f64) -> (f64, bool) {
        let accepted = err <= self.tolerance;

        let ratio = if err > 1e-30 {
            self.tolerance / err
        } else {
            // Error is essentially zero — allow maximum growth
            5.0
        };

        let dt_new = if let Some(prev) = self.prev_err {
            let prev_ratio = if prev > 1e-30 {
                self.tolerance / prev
            } else {
                5.0
            };
            // PI controller: dt_new = safety * dt * ratio^k_i * prev_ratio^(-k_p)
            self.safety_factor * dt * ratio.powf(self.k_i) * prev_ratio.powf(-self.k_p)
        } else {
            // First step: use I-controller only
            self.safety_factor * dt * ratio.powf(self.k_i)
        };

        let dt_new = dt_new.clamp(self.dt_min, self.dt_max);

        // Limit growth to 5x per step, limit shrinkage to 0.2x
        let dt_new = dt_new.min(5.0 * dt).max(0.2 * dt);

        if accepted {
            self.prev_err = Some(err);
        }

        (dt_new, accepted)
    }
}

/// Adaptive Strang splitting with Lie-in-Strang embedded error estimation.
///
/// Compares Strang (2nd-order) and Lie (1st-order) results each step to
/// estimate local error, then adapts dt via an internal PI controller.
pub struct AdaptiveStrangSplitting {
    /// Gravitational constant G used in Poisson solves.
    pub g: f64,
    /// PI controller that converts error estimates into timestep adjustments.
    pub controller: PIController,
    suggested_dt: Option<f64>,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
    /// Maximum number of retry attempts per step before accepting.
    pub max_retries: usize,
}

impl AdaptiveStrangSplitting {
    /// Create an adaptive integrator with the given gravitational constant and error tolerance.
    ///
    /// The PI controller is initialised for a 2nd-order method. `max_retries` defaults to 5.
    pub fn new(g: f64, tolerance: f64) -> Self {
        Self {
            g,
            controller: PIController::new(tolerance, 2.0),
            suggested_dt: None,
            last_timings: StepTimings::default(),
            progress: None,
            max_retries: 5,
        }
    }

    /// Compute relative L2 error between two snapshots.
    fn relative_error(a: &[f64], b: &[f64]) -> f64 {
        let mut diff_sq = 0.0f64;
        let mut norm_sq = 0.0f64;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let d = ai - bi;
            diff_sq += d * d;
            norm_sq += ai * ai;
        }
        if norm_sq > 1e-30 {
            (diff_sq / norm_sq).sqrt()
        } else {
            diff_sq.sqrt()
        }
    }
}

impl TimeIntegrator for AdaptiveStrangSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("adaptive_strang_advance").entered();
        let mut timings = StepTimings::default();

        if let Some(ref p) = self.progress {
            p.start_step();
        }

        // Use suggested dt from controller if available, but don't exceed input dt
        let mut dt_try = self.suggested_dt.unwrap_or(dt).min(dt);

        for _retry in 0..self.max_retries {
            // Take two snapshots of the initial state. PhaseSpaceSnapshot is not
            // Clone, so we need separate calls: one for the Lie step reload, one
            // for rollback on rejection.
            let snap_for_lie = repr.to_snapshot(0.0).ok_or_else(|| {
                CausticError::Solver("adaptive integrator requires to_snapshot support".into())
            })?;
            let snap_for_rollback = repr.to_snapshot(0.0).ok_or_else(|| {
                CausticError::Solver("adaptive integrator requires to_snapshot support".into())
            })?;

            // --- Strang step: drift(dt/2) -> kick(dt) -> drift(dt/2) ---
            helpers::report_phase!(self.progress, StepPhase::DriftHalf1, 0, 7);
            {
                let _s = tracing::info_span!("strang_drift_half").entered();
                helpers::time_ms!(timings, drift_ms, advector.drift(repr, dt_try / 2.0));
            }

            helpers::report_phase!(self.progress, StepPhase::PoissonSolve, 1, 7);
            let accel = {
                let _s = tracing::info_span!("strang_poisson").entered();
                let (_density, _potential, accel) = helpers::time_ms!(
                    timings,
                    poisson_ms,
                    helpers::solve_poisson(repr, solver, self.g)
                );
                accel
            };

            helpers::report_phase!(self.progress, StepPhase::Kick, 2, 7);
            {
                let _s = tracing::info_span!("strang_kick").entered();
                helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, dt_try));
            }

            helpers::report_phase!(self.progress, StepPhase::DriftHalf2, 3, 7);
            {
                let _s = tracing::info_span!("strang_drift_half").entered();
                helpers::time_ms!(timings, drift_ms, advector.drift(repr, dt_try / 2.0));
            }

            // Capture Strang result before overwriting with Lie step
            let strang_snap = repr.to_snapshot(0.0).ok_or_else(|| {
                CausticError::Solver("adaptive integrator requires to_snapshot support".into())
            })?;

            // --- Lie step: drift(dt) -> kick(dt), from the saved initial state ---
            repr.load_snapshot(snap_for_lie)?;

            helpers::report_phase!(self.progress, StepPhase::DriftHalf1, 4, 7);
            {
                let _s = tracing::info_span!("lie_drift").entered();
                helpers::time_ms!(timings, drift_ms, advector.drift(repr, dt_try));
            }

            helpers::report_phase!(self.progress, StepPhase::Kick, 5, 7);
            {
                let _s = tracing::info_span!("lie_kick").entered();
                let (_density, _potential, accel) = helpers::time_ms!(
                    timings,
                    poisson_ms,
                    helpers::solve_poisson(repr, solver, self.g)
                );
                helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, dt_try));
            }

            // --- Error estimation ---
            let lie_snap = repr.to_snapshot(0.0).ok_or_else(|| {
                CausticError::Solver("adaptive integrator requires to_snapshot support".into())
            })?;
            let err = Self::relative_error(&strang_snap.data, &lie_snap.data);

            helpers::report_phase!(self.progress, StepPhase::Diagnostics, 6, 7);

            let (dt_new, accepted) = self.controller.step(dt_try, err);
            self.suggested_dt = Some(dt_new);

            if accepted {
                // Accept Strang result (higher order)
                repr.load_snapshot(strang_snap)?;
                break;
            } else {
                // Reject: reload the original initial state and retry with smaller dt
                repr.load_snapshot(snap_for_rollback)?;
                dt_try = dt_new.min(dt);
            }
        }

        helpers::report_phase!(self.progress, StepPhase::StepComplete, 0, 0);

        let (density, potential, acceleration) =
            helpers::time_ms!(timings, density_ms, helpers::solve_poisson(repr, solver, self.g));

        self.last_timings = timings;

        Ok(StepProducts {
            density,
            potential,
            acceleration,
        })
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        helpers::dynamical_timestep(repr, self.g, cfl_factor)
    }

    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}
