//! Exit conditions — predicates evaluated after each timestep to determine whether
//! the simulation should terminate.

use super::context::SimContext;
use super::diagnostics::GlobalDiagnostics;
use super::events::{ExitConditionKind, SimEvent};
use std::cell::Cell;

/// Reason why the simulation terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExitReason {
    /// Reached user-specified t_final.
    TimeLimitReached,
    /// `‖∂f/∂t‖ < ε_steady`: distribution function has stopped evolving.
    SteadyState,
    /// `|E(t)−E(0)|/|E(0)| > ε_energy`: energy conservation violated.
    EnergyDrift,
    /// `M(t)/M(0) < ε_mass`: too much mass has left the domain.
    MassLoss,
    /// `|C₂(t)−C₂(0)|/C₂(0) > ε_casimir`: numerical diffusion detected.
    CasimirDrift,
    /// Adaptive Δt has dropped below Δt_min.
    CflViolation,
    /// Wall-clock runtime exceeded limit.
    WallClockLimit,
    /// First caustic formed: max stream count exceeded 1.
    #[serde(alias = "CausticFormed")]
    FirstCausticFormed,
    /// Virial ratio stabilised at 1.0 ± ε: violent relaxation complete.
    #[serde(alias = "VirialStabilized")]
    VirialRelaxed,
    /// User-defined predicate returned true.
    #[serde(alias = "UserStop")]
    UserDefined,
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimeLimitReached => write!(f, "Time limit reached"),
            Self::SteadyState => write!(f, "Steady state reached"),
            Self::EnergyDrift => write!(f, "Energy drift exceeded"),
            Self::MassLoss => write!(f, "Mass loss exceeded"),
            Self::CasimirDrift => write!(f, "Casimir drift exceeded"),
            Self::CflViolation => write!(f, "CFL violation"),
            Self::WallClockLimit => write!(f, "Wall clock limit reached"),
            Self::FirstCausticFormed => write!(f, "Caustic formed"),
            Self::VirialRelaxed => write!(f, "Virial ratio stabilized"),
            Self::UserDefined => write!(f, "User stop"),
        }
    }
}

/// Trait for simulation exit predicates. Evaluated after each timestep.
///
/// # Examples
///
/// ```
/// use caustic::{EnergyDriftCondition, ExitCondition, WallClockCondition};
///
/// // Exit if relative energy drift exceeds 1%
/// let energy_exit = EnergyDriftCondition { tolerance: 0.01 };
///
/// // Exit after 1 hour of wall-clock time
/// let wall_exit = WallClockCondition::new(3600.0);
/// ```
pub trait ExitCondition {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason>;
}

/// Exit when simulation time reaches `t_final`.
pub struct TimeLimitCondition {
    /// Maximum simulation time.
    pub t_final: f64,
}

impl ExitCondition for TimeLimitCondition {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        let fraction = if self.t_final > 0.0 {
            (diag.time / self.t_final).clamp(0.0, 1.0)
        } else {
            1.0
        };
        ctx.emitter.emit(SimEvent::ExitConditionStatus {
            condition: ExitConditionKind::TimeLimit,
            current_value: diag.time,
            threshold: self.t_final,
            fraction_to_threshold: fraction,
        });

        if diag.time >= self.t_final {
            Some(ExitReason::TimeLimitReached)
        } else {
            None
        }
    }
}

/// Returns true if |current - initial| / |initial| > tolerance.
fn exceeds_relative_drift(current: f64, initial: f64, tolerance: f64) -> bool {
    let ref_val = initial.abs();
    ref_val > 1e-30 && (current - initial).abs() / ref_val > tolerance
}

/// Exit when relative energy drift exceeds `tolerance`.
pub struct EnergyDriftCondition {
    /// Maximum allowed |E(t)-E(0)|/|E(0)|.
    pub tolerance: f64,
}

impl ExitCondition for EnergyDriftCondition {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        let ref_val = initial.total_energy.abs();
        let relative_drift = if ref_val > 1e-30 {
            (diag.total_energy - initial.total_energy).abs() / ref_val
        } else {
            0.0
        };
        let fraction = if self.tolerance > 0.0 {
            (relative_drift / self.tolerance).clamp(0.0, 1.0)
        } else {
            1.0
        };
        ctx.emitter.emit(SimEvent::ExitConditionStatus {
            condition: ExitConditionKind::EnergyDrift,
            current_value: relative_drift,
            threshold: self.tolerance,
            fraction_to_threshold: fraction,
        });

        exceeds_relative_drift(diag.total_energy, initial.total_energy, self.tolerance)
            .then_some(ExitReason::EnergyDrift)
    }
}

/// Exit when mass fraction drops below `threshold`.
pub struct MassLossCondition {
    /// Minimum mass fraction M(t)/M(0) before exit.
    pub threshold: f64,
}

impl ExitCondition for MassLossCondition {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        let m0 = initial.mass_in_box.abs();
        let mass_fraction = if m0 > 1e-30 { diag.mass_in_box / m0 } else { 1.0 };
        let relative_change = (1.0 - mass_fraction).abs();
        let loss_threshold = (1.0 - self.threshold).abs();
        let fraction = if loss_threshold > 0.0 {
            (relative_change / loss_threshold).clamp(0.0, 1.0)
        } else {
            1.0
        };
        ctx.emitter.emit(SimEvent::ExitConditionStatus {
            condition: ExitConditionKind::MassLoss,
            current_value: relative_change,
            threshold: loss_threshold,
            fraction_to_threshold: fraction,
        });

        if m0 > 1e-30 && mass_fraction < self.threshold {
            Some(ExitReason::MassLoss)
        } else {
            None
        }
    }
}

/// Exit when relative Casimir drift exceeds `tolerance`.
pub struct CasimirDriftCondition {
    /// Maximum allowed |C2(t)-C2(0)|/C2(0).
    pub tolerance: f64,
}

impl ExitCondition for CasimirDriftCondition {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        let ref_val = initial.casimir_c2.abs();
        let relative_drift = if ref_val > 1e-30 {
            (diag.casimir_c2 - initial.casimir_c2).abs() / ref_val
        } else {
            0.0
        };
        let fraction = if self.tolerance > 0.0 {
            (relative_drift / self.tolerance).clamp(0.0, 1.0)
        } else {
            1.0
        };
        ctx.emitter.emit(SimEvent::ExitConditionStatus {
            condition: ExitConditionKind::CasimirDrift,
            current_value: relative_drift,
            threshold: self.tolerance,
            fraction_to_threshold: fraction,
        });

        exceeds_relative_drift(diag.casimir_c2, initial.casimir_c2, self.tolerance)
            .then_some(ExitReason::CasimirDrift)
    }
}

/// Exit when wall-clock time exceeds `limit_secs` seconds.
pub struct WallClockCondition {
    /// Maximum wall-clock seconds.
    pub limit_secs: f64,
    /// Instant when the condition was created.
    pub start: std::time::Instant,
}

impl WallClockCondition {
    pub fn new(limit_secs: f64) -> Self {
        Self {
            limit_secs,
            start: std::time::Instant::now(),
        }
    }
}

impl ExitCondition for WallClockCondition {
    fn check(&self, _diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        let elapsed = self.start.elapsed().as_secs_f64();
        let fraction = if self.limit_secs > 0.0 {
            (elapsed / self.limit_secs).clamp(0.0, 1.0)
        } else {
            1.0
        };
        ctx.emitter.emit(SimEvent::ExitConditionStatus {
            condition: ExitConditionKind::WallClock,
            current_value: elapsed,
            threshold: self.limit_secs,
            fraction_to_threshold: fraction,
        });

        if elapsed > self.limit_secs {
            Some(ExitReason::WallClockLimit)
        } else {
            None
        }
    }
}

/// Exit when `‖∂f/∂t‖ < threshold` (steady state reached).
/// Uses interior mutability (Cell) to track previous entropy across calls,
/// since the trait takes `&self`.
pub struct SteadyStateCondition {
    /// Maximum entropy change rate before declaring steady state.
    pub threshold: f64,
    prev_entropy: Cell<Option<f64>>,
}

impl SteadyStateCondition {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            prev_entropy: Cell::new(None),
        }
    }
}

impl ExitCondition for SteadyStateCondition {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        let current = diag.entropy;
        if let Some(prev) = self.prev_entropy.get() {
            let dt = diag.time; // Approximate: use absolute time as proxy
            let rate = if dt > 1e-30 {
                (current - prev).abs() / dt
            } else {
                f64::MAX
            };
            self.prev_entropy.set(Some(current));

            // Fraction is inverted: rate < threshold means steady state,
            // so fraction_to_threshold = 1.0 when rate has dropped to threshold.
            let fraction = if rate < f64::MAX && self.threshold > 0.0 {
                (1.0 - (rate / self.threshold).min(1.0)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            ctx.emitter.emit(SimEvent::ExitConditionStatus {
                condition: ExitConditionKind::SteadyState,
                current_value: rate,
                threshold: self.threshold,
                fraction_to_threshold: fraction,
            });

            if rate < self.threshold {
                return Some(ExitReason::SteadyState);
            }
        } else {
            self.prev_entropy.set(Some(current));
        }
        None
    }
}

/// Exit when adaptive Δt drops below `dt_min`.
pub struct CflViolationCondition {
    /// Minimum adaptive timestep before declaring CFL violation.
    pub dt_min: f64,
}

impl ExitCondition for CflViolationCondition {
    fn check(&self, _diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics, _ctx: &SimContext) -> Option<ExitReason> {
        // CFL violation is checked in Simulation::step via max_dt; this condition is a fallback
        None
    }
}

/// Exit when the first caustic forms (max stream count > 1).
pub struct CausticFormationCondition;

impl ExitCondition for CausticFormationCondition {
    fn check(&self, _diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics, _ctx: &SimContext) -> Option<ExitReason> {
        // Stream count field is not stored in GlobalDiagnostics; requires repr access.
        // Deferred until max_stream_count is added to GlobalDiagnostics.
        None
    }
}

/// Exit when the virial ratio 2T/|W| stabilises within `tolerance` of 1.0.
///
/// Includes a minimum step count to avoid premature exit for systems that
/// start in or near virial equilibrium (e.g. Plummer sphere ICs).
pub struct VirialRelaxedCondition {
    /// Maximum |2T/|W| - 1| for virial equilibrium.
    pub tolerance: f64,
    /// Minimum number of evaluation calls before the condition can trigger.
    pub min_steps: u64,
    step_count: Cell<u64>,
}

impl VirialRelaxedCondition {
    pub fn new(tolerance: f64, min_steps: u64) -> Self {
        Self {
            tolerance,
            min_steps,
            step_count: Cell::new(0),
        }
    }
}

impl ExitCondition for VirialRelaxedCondition {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        let count = self.step_count.get();
        self.step_count.set(count + 1);

        let virial_deviation = (diag.virial_ratio - 1.0).abs();
        // Fraction is inverted: deviation < tolerance means virial equilibrium,
        // so fraction_to_threshold = 1.0 when deviation has dropped to tolerance.
        let fraction = if self.tolerance > 0.0 {
            (1.0 - (virial_deviation / self.tolerance).min(1.0)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        ctx.emitter.emit(SimEvent::ExitConditionStatus {
            condition: ExitConditionKind::VirialRelaxed,
            current_value: virial_deviation,
            threshold: self.tolerance,
            fraction_to_threshold: fraction,
        });

        if count < self.min_steps {
            return None;
        }
        if virial_deviation < self.tolerance {
            Some(ExitReason::VirialRelaxed)
        } else {
            None
        }
    }
}
