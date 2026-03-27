//! Blanes-Moan optimized 4th-order symplectic integrator (BM4, method SRKN₆b).
//!
//! A 6-stage palindromic drift-kick-drift (DKD) composition that achieves
//! 4th-order accuracy with significantly smaller leading-order error constants
//! than the classical Yoshida 4th-order splitting. The palindromic structure
//! (a₁ a₂ a₃ a₃ a₂ a₁ for drifts, b₁ b₂ b₃ b₂ b₁ for kicks) ensures
//! time-reversibility and symplecticity. Totals 11 sub-steps (6 drifts +
//! 5 kicks) versus Yoshida's 7, trading more force evaluations per step for
//! a lower error constant that allows larger stable time steps.
//!
//! Coefficients from Blanes & Moan, "Practical symplectic partitioned
//! Runge-Kutta and Runge-Kutta-Nystrom methods",
//! J. Comput. Appl. Math. 142 (2002), 313-330, method SRKN₆b.

use super::super::{
    context::SimContext,
    integrator::{StepProducts, StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::StepPhase,
    types::*,
};
use super::helpers;
use crate::CausticError;

// BM4 palindromic drift coefficients a_i (6 values, symmetric: a1 a2 a3 a3 a2 a1).
// 2*(a1 + a2 + a3) = 1.
const BM4_A: [f64; 6] = [
    0.0502627644003922,
    0.4134651224174328,
    0.0362721131821750,
    0.0362721131821750,
    0.4134651224174328,
    0.0502627644003922,
];

// BM4 palindromic kick coefficients b_i (5 values between drifts, symmetric: b1 b2 b3 b2 b1).
// 2*(b1 + b2) + b3 = 1.
const BM4_B: [f64; 5] = [
    0.1488177424796559,
    0.527_753_876_542_476,
    -0.3531432380442638,
    0.527_753_876_542_476,
    0.1488177424796559,
];

/// Blanes-Moan 4th-order symplectic integrator (6-stage, palindromic DKD).
///
/// Pattern: D(a₁) K(b₁) D(a₂) K(b₂) D(a₃) K(b₃) D(a₃) K(b₂) D(a₂) K(b₁) D(a₁)
///
/// Compared to Yoshida (7 sub-steps, 4th-order), BM4 has more sub-steps (11)
/// but achieves significantly smaller leading-order error coefficients, making
/// it preferable when drift and kick evaluations are cheap relative to the
/// accuracy gain.
#[derive(Default)]
pub struct BlanesMoanSplitting {
    /// Timing breakdown from the most recent `advance` call.
    last_timings: StepTimings,
}

impl BlanesMoanSplitting {
    /// Creates a Blanes-Moan 4th-order integrator.
    pub fn new() -> Self {
        Self {
            last_timings: StepTimings::default(),
        }
    }
}

impl TimeIntegrator for BlanesMoanSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        ctx: &SimContext,
    ) -> Result<StepProducts, CausticError> {
        let mut timings = StepTimings::default();
        let n_sub: u8 = 11;
        let dt = ctx.dt;

        ctx.progress.start_step();

        // DKD palindromic: D(a0) K(b0) D(a1) K(b1) D(a2) K(b2) D(a3) K(b3) D(a4) K(b4) D(a5)
        // = 6 drifts interleaved with 5 kicks = 11 sub-steps.
        for i in 0..6 {
            // --- Drift a_i * dt ---
            helpers::report_phase!(ctx, StepPhase::DriftHalf1, 2 * i as u8, n_sub);
            helpers::time_ms!(
                timings,
                drift_ms,
                ctx.advector.drift(repr, &ctx.with_dt(BM4_A[i] * dt))
            );

            // --- Kick b_i * dt (5 kicks between 6 drifts) ---
            if i < 5 {
                helpers::report_phase!(ctx, StepPhase::Kick, 2 * i as u8 + 1, n_sub);
                let (_density, _potential, accel) =
                    helpers::time_ms!(timings, poisson_ms, helpers::solve_poisson(repr, ctx));
                helpers::time_ms!(
                    timings,
                    kick_ms,
                    ctx.advector.kick(repr, &accel, &ctx.with_dt(BM4_B[i] * dt))
                );
            }
        }

        helpers::report_phase!(ctx, StepPhase::StepComplete, n_sub, n_sub);

        // Compute end-of-step products for caller reuse
        let (density, potential, acceleration) =
            helpers::time_ms!(timings, density_ms, helpers::solve_poisson(repr, ctx));

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
