//! 6th-order symplectic integrator via Suzuki triple-jump composition.
//!
//! Achieves 6th-order accuracy by composing three 4th-order Yoshida (Strang)
//! blocks with Suzuki (1990) / Kahan & Li (1997) triple-jump coefficients:
//!
//!   S₆(Δt) = S₄(s₁·Δt) . S₄(s₂·Δt) . S₄(s₁·Δt)
//!
//! where s₁ = 1/(2 - 2^{1/5}), s₂ = 1 - 2*s₁. The negative middle
//! coefficient s₂ cancels the 5th-order error term from the outer blocks.
//! Each inner S₄ step is a standard Yoshida 4th-order splitting with 7
//! sub-steps (4 drifts + 3 kicks), giving 3 x 7 = 21 sub-steps total,
//! reduced to 19 when adjacent boundary drifts are merged at the two
//! composition seams. Provides much tighter energy conservation than 4th-order
//! methods at the cost of roughly 3x more Poisson solves per time step.

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

/// Suzuki triple-jump coefficient s₁ = 1 / (2 − 2^{1/5}).
const S1: f64 = 1.1746881100325735;
/// Suzuki triple-jump coefficient s₂ = 1 − 2·s₁.
const S2: f64 = -1.349_376_220_065_147;

/// Yoshida coefficient w₁ = 1 / (2 − 2^{1/3}).
const YOSHIDA_W1: f64 = 1.3512071919596578;
/// Yoshida coefficient w₀ = 1 − 2·w₁.
const YOSHIDA_W0: f64 = -1.7024143839193153;

/// 6th-order symplectic integrator (triple-jump composition of Yoshida).
///
/// Higher accuracy than Yoshida alone (4th-order), at the cost of 3x
/// the sub-steps per time step. Useful when very tight energy conservation
/// is required or when large time steps are desirable.
pub struct Rkn6Splitting {
    /// Gravitational constant G used in the Poisson solve.
    pub g: f64,
    /// Timing breakdown from the most recent `advance` call.
    last_timings: StepTimings,
    /// Optional lock-free progress reporter for TUI sub-step tracking.
    progress: Option<Arc<StepProgress>>,
}

impl Rkn6Splitting {
    /// Creates a 6th-order triple-jump integrator with the given gravitational constant.
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
            progress: None,
        }
    }

    /// Single drift sub-step with progress reporting and timing.
    #[inline]
    fn rkn6_drift(
        repr: &mut dyn PhaseSpaceRepr,
        advector: &dyn Advector,
        coeff: f64,
        timings: &mut StepTimings,
        progress: &Option<Arc<StepProgress>>,
        phase: StepPhase,
        sub: u8,
    ) {
        helpers::report_phase!(progress, phase, sub, 19);
        let _s = tracing::info_span!("rkn6_drift").entered();
        helpers::time_ms!(timings, drift_ms, advector.drift(repr, coeff));
    }

    /// Single kick sub-step (density → potential → acceleration → kick) with timing.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn rkn6_kick(
        g: f64,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        coeff: f64,
        timings: &mut StepTimings,
        progress: &Option<Arc<StepProgress>>,
        sub: u8,
    ) {
        helpers::report_phase!(progress, StepPhase::Kick, sub, 19);
        let _s = tracing::info_span!("rkn6_kick").entered();
        let (_density, _potential, accel) =
            helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, solver, g));
        helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, coeff));
    }

    /// Execute one Yoshida 4th-order step with the given (scaled) time step.
    ///
    /// Pattern: D(w₁/2) K(w₁) D((w₁+w₀)/2) K(w₀) D((w₀+w₁)/2) K(w₁) D(w₁/2)
    #[allow(clippy::too_many_arguments)]
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
        helpers::report_phase!(progress, StepPhase::DriftHalf1, base_sub, total_sub);
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, YOSHIDA_W1 * dt / 2.0));
        }

        // Sub-step 2: kick w1·dt
        helpers::report_phase!(progress, StepPhase::Kick, base_sub + 1, total_sub);
        {
            let _s = tracing::info_span!("rkn6_kick").entered();
            let (_density, _potential, accel) =
                helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, solver, self.g));
            helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, YOSHIDA_W1 * dt));
        }

        // Sub-step 3: drift (w1+w0)·dt/2
        helpers::report_phase!(progress, StepPhase::DriftHalf2, base_sub + 2, total_sub);
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, (YOSHIDA_W1 + YOSHIDA_W0) * dt / 2.0));
        }

        // Sub-step 4: kick w0·dt
        helpers::report_phase!(progress, StepPhase::Kick, base_sub + 3, total_sub);
        {
            let _s = tracing::info_span!("rkn6_kick").entered();
            let (_density, _potential, accel) =
                helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, solver, self.g));
            helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, YOSHIDA_W0 * dt));
        }

        // Sub-step 5: drift (w0+w1)·dt/2
        helpers::report_phase!(progress, StepPhase::DriftHalf1, base_sub + 4, total_sub);
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, (YOSHIDA_W0 + YOSHIDA_W1) * dt / 2.0));
        }

        // Sub-step 6: kick w1·dt
        helpers::report_phase!(progress, StepPhase::Kick, base_sub + 5, total_sub);
        {
            let _s = tracing::info_span!("rkn6_kick").entered();
            let (_density, _potential, accel) =
                helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, solver, self.g));
            helpers::time_ms!(timings, kick_ms, advector.kick(repr, &accel, YOSHIDA_W1 * dt));
        }

        // Sub-step 7: drift w1·dt/2
        helpers::report_phase!(progress, StepPhase::DriftHalf2, base_sub + 6, total_sub);
        {
            let _s = tracing::info_span!("rkn6_drift").entered();
            helpers::time_ms!(timings, drift_ms, advector.drift(repr, YOSHIDA_W1 * dt / 2.0));
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
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("rkn6_advance").entered();
        let mut timings = StepTimings::default();
        let g = self.g;
        let progress = self.progress.clone();

        if let Some(ref p) = progress {
            p.start_step();
        }

        // Inlined 19-sub-step sequence: three Yoshida S₄ steps composed via
        // Suzuki triple-jump, with adjacent boundary drifts merged at the two
        // composition seams (saves 2 drift operations vs the naive 21-step form).
        //
        // S₆(Δt) = S₄(s₁·Δt) ∘ S₄(s₂·Δt) ∘ S₄(s₁·Δt)

        let s1dt = S1 * dt;
        let s2dt = S2 * dt;

        // Drift coefficients
        let d_half_w1_s1 = YOSHIDA_W1 * s1dt / 2.0;
        let d_half_w01_s1 = (YOSHIDA_W1 + YOSHIDA_W0) * s1dt / 2.0;
        let d_half_w01_s2 = (YOSHIDA_W1 + YOSHIDA_W0) * s2dt / 2.0;
        // Merged boundary drift at composition seams: w₁·(s₁+s₂)·dt/2
        let d_seam = YOSHIDA_W1 * (S1 + S2) * dt / 2.0;

        // Kick coefficients
        let k_w1_s1 = YOSHIDA_W1 * s1dt;
        let k_w0_s1 = YOSHIDA_W0 * s1dt;
        let k_w1_s2 = YOSHIDA_W1 * s2dt;
        let k_w0_s2 = YOSHIDA_W0 * s2dt;

        // ── S₄(s₁·Δt) ─────────────────────────────────────────────────────
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w1_s1,
            &mut timings,
            &progress,
            StepPhase::DriftHalf1,
            0,
        );
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w1_s1,
            &mut timings,
            &progress,
            1,
        );
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w01_s1,
            &mut timings,
            &progress,
            StepPhase::DriftHalf2,
            2,
        );
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w0_s1,
            &mut timings,
            &progress,
            3,
        );
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w01_s1,
            &mut timings,
            &progress,
            StepPhase::DriftHalf1,
            4,
        );
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w1_s1,
            &mut timings,
            &progress,
            5,
        );

        // Merged: last drift of S₄(s₁) + first drift of S₄(s₂)
        Self::rkn6_drift(
            repr,
            advector,
            d_seam,
            &mut timings,
            &progress,
            StepPhase::DriftHalf2,
            6,
        );

        // ── S₄(s₂·Δt) interior ─────────────────────────────────────────────
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w1_s2,
            &mut timings,
            &progress,
            7,
        );
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w01_s2,
            &mut timings,
            &progress,
            StepPhase::DriftHalf1,
            8,
        );
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w0_s2,
            &mut timings,
            &progress,
            9,
        );
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w01_s2,
            &mut timings,
            &progress,
            StepPhase::DriftHalf2,
            10,
        );
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w1_s2,
            &mut timings,
            &progress,
            11,
        );

        // Merged: last drift of S₄(s₂) + first drift of S₄(s₁)
        Self::rkn6_drift(
            repr,
            advector,
            d_seam,
            &mut timings,
            &progress,
            StepPhase::DriftHalf1,
            12,
        );

        // ── S₄(s₁·Δt) ─────────────────────────────────────────────────────
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w1_s1,
            &mut timings,
            &progress,
            13,
        );
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w01_s1,
            &mut timings,
            &progress,
            StepPhase::DriftHalf2,
            14,
        );
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w0_s1,
            &mut timings,
            &progress,
            15,
        );
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w01_s1,
            &mut timings,
            &progress,
            StepPhase::DriftHalf1,
            16,
        );
        Self::rkn6_kick(
            g,
            repr,
            solver,
            advector,
            k_w1_s1,
            &mut timings,
            &progress,
            17,
        );
        Self::rkn6_drift(
            repr,
            advector,
            d_half_w1_s1,
            &mut timings,
            &progress,
            StepPhase::DriftHalf2,
            18,
        );

        helpers::report_phase!(progress, StepPhase::StepComplete, 19, 19);

        // Compute end-of-step products for caller reuse
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
