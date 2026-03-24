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
pub struct BlanesMoanSplitting {
    /// Gravitational constant G used in the Poisson solve.
    pub g: f64,
    /// Timing breakdown from the most recent `advance` call.
    last_timings: StepTimings,
    /// Optional lock-free progress reporter for TUI sub-step tracking.
    progress: Option<Arc<StepProgress>>,
}

impl BlanesMoanSplitting {
    /// Creates a Blanes-Moan 4th-order integrator with the given gravitational constant.
    pub fn new(g: f64) -> Self {
        Self {
            g,
            last_timings: StepTimings::default(),
            progress: None,
        }
    }
}

impl TimeIntegrator for BlanesMoanSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> Result<StepProducts, CausticError> {
        let _span = tracing::info_span!("bm4_advance").entered();
        let mut timings = StepTimings::default();
        let n_sub: u8 = 11;

        if let Some(ref p) = self.progress {
            p.start_step();
        }

        // DKD palindromic: D(a0) K(b0) D(a1) K(b1) D(a2) K(b2) D(a3) K(b3) D(a4) K(b4) D(a5)
        // = 6 drifts interleaved with 5 kicks = 11 sub-steps.
        for i in 0..6 {
            // --- Drift a_i * dt ---
            if let Some(ref p) = self.progress {
                p.set_phase(StepPhase::DriftHalf1);
                p.set_sub_step(2 * i as u8, n_sub);
            }
            {
                let _s = tracing::info_span!("bm4_drift", stage = i).entered();
                let t0 = Instant::now();
                advector.drift(repr, BM4_A[i] * dt);
                timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
            }

            // --- Kick b_i * dt (5 kicks between 6 drifts) ---
            if i < 5 {
                if let Some(ref p) = self.progress {
                    p.set_phase(StepPhase::Kick);
                    p.set_sub_step(2 * i as u8 + 1, n_sub);
                }
                {
                    let _s = tracing::info_span!("bm4_kick", stage = i).entered();
                    let t0 = Instant::now();
                    let density = repr.compute_density();
                    let potential = solver.solve(&density, self.g);
                    let accel = solver.compute_acceleration(&potential);
                    timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
                    let t0 = Instant::now();
                    advector.kick(repr, &accel, BM4_B[i] * dt);
                    timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
        }

        // Compute end-of-step products for caller reuse
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
