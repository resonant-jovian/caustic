//! RK-BUG integrator: Runge-Kutta stages composed with BUG steps.
//!
//! Performs a BUG step at each RK stage, achieving arbitrary-order convergence
//! on the low-rank manifold. Currently implements SSP-RK3 (3rd-order strong
//! stability-preserving) Butcher tableau.
//!
//! Reference: Ceruti, Einkemmer, Kusch & Lubich (arXiv 2502.07040, Feb 2025),
//! "Runge-Kutta methods for dynamical low-rank approximation".

use super::super::{
    algos::ht::HtTensor,
    context::SimContext,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::StepPhase,
    types::*,
};
use super::helpers;
use crate::CausticError;

use super::bug::{BugConfig, bug_drift_substep, bug_kick_substep, conservative_correction};

/// Configuration for the RK-BUG integrator.
pub struct RkBugConfig {
    /// Truncation tolerance for BUG sub-steps and HT additions.
    pub tolerance: f64,
    /// Maximum rank per node.
    pub max_rank: usize,
    /// RK order (currently supports 3 for SSP-RK3).
    pub rk_order: usize,
    /// Number of extra basis columns per BUG K-step.
    pub rank_increase: usize,
    /// Apply conservative correction after each stage.
    pub conservative: bool,
}

impl Default for RkBugConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_rank: 50,
            rk_order: 3,
            rank_increase: 2,
            conservative: false,
        }
    }
}

/// RK-BUG integrator: classical Runge-Kutta time stepping with BUG tensor updates.
///
/// Uses SSP-RK3 for the macro time step and BUG for the HT factor updates at each
/// stage. Falls back to standard Strang splitting for non-HT representations.
pub struct RkBugIntegrator {
    /// BUG truncation, rank, and conservation settings.
    pub config: RkBugConfig,
    last_timings: StepTimings,
}

impl RkBugIntegrator {
    /// Create a new RK-BUG integrator with the given config.
    pub fn new(config: RkBugConfig) -> Self {
        Self {
            config,
            last_timings: StepTimings::default(),
        }
    }

    fn bug_config(&self) -> BugConfig {
        BugConfig {
            tolerance: self.config.tolerance,
            max_rank: self.config.max_rank,
            midpoint: false,
            conservative: false,
            rank_increase: self.config.rank_increase,
        }
    }

    /// Perform one BUG Strang step: drift(dt/2) → kick(dt) → drift(dt/2).
    ///
    /// The input HT is modified in-place.
    fn bug_strang_step(
        ht: &mut HtTensor,
        ctx: &SimContext,
        dt: f64,
        config: &BugConfig,
        timings: &mut StepTimings,
    ) {
        helpers::time_ms!(timings, drift_ms, bug_drift_substep(ht, dt / 2.0, config));

        let (_, _, accel) =
            helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(ht, ctx));

        helpers::time_ms!(timings, kick_ms, bug_kick_substep(ht, &accel, dt, config));

        helpers::time_ms!(timings, drift_ms, bug_drift_substep(ht, dt / 2.0, config));
    }

    /// SSP-RK3 (Shu-Osher form) with BUG sub-steps:
    ///
    /// Stage 1: Y^(1) = BUG(Y^(0), dt)
    /// Stage 2: Z^(2) = BUG(Y^(1), dt); Y^(2) = 3/4 · Y^(0) + 1/4 · Z^(2)
    /// Stage 3: Z^(3) = BUG(Y^(2), dt); Y^(n+1) = 1/3 · Y^(0) + 2/3 · Z^(3)
    ///
    /// The `scaled_add` on HtTensor handles weighted addition with truncation.
    fn ssp_rk3_step(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        ctx: &SimContext,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        let Some(ht) = repr.as_any_mut().downcast_mut::<HtTensor>() else {
            debug_assert!(false, "RK-BUG requires HtTensor");
            return;
        };

        let density_before = if self.config.conservative {
            Some(ht.compute_density())
        } else {
            None
        };

        let config = self.bug_config();
        let tol = self.config.tolerance;

        // Save Y^(0)
        let y0 = ht.clone();

        // Stage 1: Y^(1) = BUG(Y^(0), dt)
        helpers::report_phase!(ctx, StepPhase::BugKStep, 0, 4);
        Self::bug_strang_step(ht, ctx, dt, &config, timings);
        // ht is now Y^(1)

        // Stage 2: Z^(2) = BUG(Y^(1), dt)
        helpers::report_phase!(ctx, StepPhase::BugKStep, 1, 4);
        let mut z2 = ht.clone();
        Self::bug_strang_step(&mut z2, ctx, dt, &config, timings);
        // Y^(2) = 3/4 · Y^(0) + 1/4 · Z^(2)
        let y2 = y0.scaled_add(3.0 / 4.0, &z2, 1.0 / 4.0, tol);
        *ht = y2;

        // Stage 3: Z^(3) = BUG(Y^(2), dt)
        helpers::report_phase!(ctx, StepPhase::BugLStep, 2, 4);
        let mut z3 = ht.clone();
        Self::bug_strang_step(&mut z3, ctx, dt, &config, timings);
        // Y^(n+1) = 1/3 · Y^(0) + 2/3 · Z^(3)
        let result = y0.scaled_add(1.0 / 3.0, &z3, 2.0 / 3.0, tol);
        *ht = result;

        helpers::report_phase!(ctx, StepPhase::BugSStep, 3, 4);

        if let Some(ref dens) = density_before {
            conservative_correction(ht, dens);
        }
    }

    /// Fallback: standard Strang splitting for non-HT representations.
    fn strang_fallback(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        ctx: &SimContext,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        helpers::time_ms!(timings, drift_ms, ctx.advector.drift(repr, &ctx.with_dt(dt / 2.0)));

        let (_, _, accel) = helpers::time_ms!(
            timings,
            poisson_ms,
            helpers::solve_poisson(repr, ctx)
        );

        helpers::time_ms!(timings, kick_ms, ctx.advector.kick(repr, &accel, &ctx.with_dt(dt)));

        helpers::time_ms!(timings, drift_ms, ctx.advector.drift(repr, &ctx.with_dt(dt / 2.0)));
    }
}

impl TimeIntegrator for RkBugIntegrator {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        ctx: &SimContext,
    ) -> Result<StepProducts, CausticError> {
        let mut timings = StepTimings::default();
        let dt = ctx.dt;

        ctx.progress.start_step();

        let is_ht = repr.as_any().downcast_ref::<HtTensor>().is_some();

        if is_ht {
            self.ssp_rk3_step(repr, ctx, dt, &mut timings);
        } else {
            self.strang_fallback(repr, ctx, dt, &mut timings);
        }

        helpers::report_phase!(ctx, StepPhase::StepComplete, 4, 4);

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

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        helpers::dynamical_timestep(repr, 1.0, cfl_factor)
    }

    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk_bug_config_defaults() {
        let cfg = RkBugConfig::default();
        assert_eq!(cfg.rk_order, 3);
        assert_eq!(cfg.max_rank, 50);
        assert_eq!(cfg.rank_increase, 2);
        assert!(!cfg.conservative);
    }
}
