//! RK-BUG integrator: Runge-Kutta stages composed with BUG steps.
//!
//! Performs a BUG step at each RK stage, achieving arbitrary-order convergence
//! on the low-rank manifold. Currently implements SSP-RK3 (3rd-order strong
//! stability-preserving) Butcher tableau.
//!
//! Reference: Ceruti, Einkemmer, Kusch & Lubich (arXiv 2502.07040, Feb 2025),
//! "Runge-Kutta methods for dynamical low-rank approximation".

use std::sync::Arc;
use std::time::Instant;

use super::super::{
    advecator::Advector,
    algos::ht::HtTensor,
    integrator::{StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};

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

/// RK-BUG integrator: Runge-Kutta method with BUG sub-steps.
pub struct RkBugIntegrator {
    pub config: RkBugConfig,
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl RkBugIntegrator {
    pub fn new(g: f64, config: RkBugConfig) -> Self {
        Self {
            config,
            g,
            last_timings: StepTimings::default(),
            progress: None,
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
        solver: &dyn PoissonSolver,
        g: f64,
        dt: f64,
        config: &BugConfig,
        timings: &mut StepTimings,
    ) {
        let t0 = Instant::now();
        bug_drift_substep(ht, dt / 2.0, config);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let density = ht.compute_density();
        let potential = solver.solve(&density, g);
        let accel = solver.compute_acceleration(&potential);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        bug_kick_substep(ht, &accel, dt, config);
        timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        bug_drift_substep(ht, dt / 2.0, config);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
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
        solver: &dyn PoissonSolver,
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
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::BugKStep);
            p.set_sub_step(0, 4);
        }
        Self::bug_strang_step(ht, solver, self.g, dt, &config, timings);
        // ht is now Y^(1)

        // Stage 2: Z^(2) = BUG(Y^(1), dt)
        if let Some(ref p) = self.progress {
            p.set_sub_step(1, 4);
        }
        let mut z2 = ht.clone();
        Self::bug_strang_step(&mut z2, solver, self.g, dt, &config, timings);
        // Y^(2) = 3/4 · Y^(0) + 1/4 · Z^(2)
        let y2 = y0.scaled_add(3.0 / 4.0, &z2, 1.0 / 4.0, tol);
        *ht = y2;

        // Stage 3: Z^(3) = BUG(Y^(2), dt)
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::BugLStep);
            p.set_sub_step(2, 4);
        }
        let mut z3 = ht.clone();
        Self::bug_strang_step(&mut z3, solver, self.g, dt, &config, timings);
        // Y^(n+1) = 1/3 · Y^(0) + 2/3 · Z^(3)
        let result = y0.scaled_add(1.0 / 3.0, &z3, 2.0 / 3.0, tol);
        *ht = result;

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::BugSStep);
            p.set_sub_step(3, 4);
        }

        if let Some(ref dens) = density_before {
            conservative_correction(ht, dens);
        }
    }

    /// Fallback: standard Strang splitting for non-HT representations.
    fn strang_fallback(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.g);
        let accel = solver.compute_acceleration(&potential);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        advector.kick(repr, &accel, dt);
        timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
}

impl TimeIntegrator for RkBugIntegrator {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        let _span = tracing::info_span!("rk_bug_advance").entered();
        let mut timings = StepTimings::default();

        if let Some(ref p) = self.progress {
            p.start_step();
        }

        let is_ht = repr.as_any().downcast_ref::<HtTensor>().is_some();

        if is_ht {
            self.ssp_rk3_step(repr, solver, dt, &mut timings);
        } else {
            self.strang_fallback(repr, solver, advector, dt, &mut timings);
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
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

    #[test]
    fn rk_bug_config_defaults() {
        let cfg = RkBugConfig::default();
        assert_eq!(cfg.rk_order, 3);
        assert_eq!(cfg.max_rank, 50);
        assert_eq!(cfg.rank_increase, 2);
        assert!(!cfg.conservative);
    }
}
