//! 6th-order symplectic integrator via triple-jump composition.
//!
//! Achieves 6th-order accuracy by composing three 4th-order Yoshida steps
//! with Suzuki (1990) / Kahan & Li (1997) coefficients:
//!
//!   S₆(Δt) = S₄(s₁·Δt) ∘ S₄(s₂·Δt) ∘ S₄(s₁·Δt)
//!
//! where s₁ = 1/(2 − 2^{1/5}), s₂ = 1 − 2·s₁.
//!
//! Each inner S₄ (Yoshida) step has 7 sub-steps (4 drifts + 3 kicks),
//! giving 3 × 7 = 21 sub-steps total, reduced to 19 when adjacent
//! boundary drifts are merged at composition seams.

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

/// Suzuki triple-jump coefficient s₁ = 1 / (2 − 2^{1/5}).
const S1: f64 = 1.1746881100325735;
/// Suzuki triple-jump coefficient s₂ = 1 − 2·s₁.
const S2: f64 = -1.3493762200651470;

/// Yoshida coefficient w₁ = 1 / (2 − 2^{1/3}).
const YOSHIDA_W1: f64 = 1.3512071919596578;
/// Yoshida coefficient w₀ = 1 − 2·w₁.
const YOSHIDA_W0: f64 = -1.7024143839193153;

/// 6th-order symplectic integrator (triple-jump composition of Yoshida).
///
/// Higher accuracy than Yoshida alone (4th-order), at the cost of 3×
/// the sub-steps per time step. Useful when very tight energy conservation
/// is required or when large time steps are desirable.
pub struct Rkn6Splitting {
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl Rkn6Splitting {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
            progress: None,
        }
    }

    /// Execute one Yoshida 4th-order step with the given (scaled) time step.
    ///
    /// Pattern: D(w₁/2) K(w₁) D((w₁+w₀)/2) K(w₀) D((w₀+w₁)/2) K(w₁) D(w₁/2)
    fn yoshida_step(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
        timings: &mut StepTimings,
        progress: &Option<Arc<StepProgress>>,
        base_sub: u8,
        total_sub: u8,
    ) {
        // Sub-step 1: drift w1·dt/2
        if let Some(p) = progress {
            p.set_phase(StepPhase::DriftHalf1);
            p.set_sub_step(base_sub, total_sub);
        }
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            let t0 = Instant::now();
            advector.drift(repr, YOSHIDA_W1 * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Sub-step 2: kick w1·dt
        if let Some(p) = progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(base_sub + 1, total_sub);
        }
        {
            let _s = tracing::info_span!("rkn6_kick").entered();
            let t0 = Instant::now();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            advector.kick(repr, &accel, YOSHIDA_W1 * dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Sub-step 3: drift (w1+w0)·dt/2
        if let Some(p) = progress {
            p.set_phase(StepPhase::DriftHalf2);
            p.set_sub_step(base_sub + 2, total_sub);
        }
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            let t0 = Instant::now();
            advector.drift(repr, (YOSHIDA_W1 + YOSHIDA_W0) * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Sub-step 4: kick w0·dt
        if let Some(p) = progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(base_sub + 3, total_sub);
        }
        {
            let _s = tracing::info_span!("rkn6_kick").entered();
            let t0 = Instant::now();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            advector.kick(repr, &accel, YOSHIDA_W0 * dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Sub-step 5: drift (w0+w1)·dt/2
        if let Some(p) = progress {
            p.set_phase(StepPhase::DriftHalf1);
            p.set_sub_step(base_sub + 4, total_sub);
        }
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            let t0 = Instant::now();
            advector.drift(repr, (YOSHIDA_W0 + YOSHIDA_W1) * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Sub-step 6: kick w1·dt
        if let Some(p) = progress {
            p.set_phase(StepPhase::Kick);
            p.set_sub_step(base_sub + 5, total_sub);
        }
        {
            let _s = tracing::info_span!("rkn6_kick").entered();
            let t0 = Instant::now();
            let density = repr.compute_density();
            let potential = solver.solve(&density, self.g);
            let accel = solver.compute_acceleration(&potential);
            timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
            let t0 = Instant::now();
            advector.kick(repr, &accel, YOSHIDA_W1 * dt);
            timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Sub-step 7: drift w1·dt/2
        if let Some(p) = progress {
            p.set_phase(StepPhase::DriftHalf2);
            p.set_sub_step(base_sub + 6, total_sub);
        }
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            let t0 = Instant::now();
            advector.drift(repr, YOSHIDA_W1 * dt / 2.0);
            timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }
    }
}

impl TimeIntegrator for Rkn6Splitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        let _span = tracing::info_span!("rkn6_advance").entered();
        let mut timings = StepTimings::default();
        // 3 Yoshida steps × 7 sub-steps = 21 total sub-steps
        let n_sub: u8 = 21;

        if let Some(ref p) = self.progress {
            p.start_step();
        }

        // Triple-jump: S4(s1·dt) ∘ S4(s2·dt) ∘ S4(s1·dt)
        self.yoshida_step(
            repr, solver, advector, S1 * dt,
            &mut timings, &self.progress.clone(), 0, n_sub,
        );

        self.yoshida_step(
            repr, solver, advector, S2 * dt,
            &mut timings, &self.progress.clone(), 7, n_sub,
        );

        self.yoshida_step(
            repr, solver, advector, S1 * dt,
            &mut timings, &self.progress.clone(), 14, n_sub,
        );

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
