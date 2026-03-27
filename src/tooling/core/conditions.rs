//! Exit conditions — predicates evaluated after each timestep to determine whether
//! the simulation should terminate.

use super::diagnostics::GlobalDiagnostics;
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
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason>;
}

/// Exit when simulation time reaches `t_final`.
pub struct TimeLimitCondition {
    /// Maximum simulation time.
    pub t_final: f64,
}

impl ExitCondition for TimeLimitCondition {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
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
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason> {
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
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason> {
        let m0 = initial.mass_in_box.abs();
        if m0 > 1e-30 && diag.mass_in_box / m0 < self.threshold {
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
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason> {
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
    fn check(&self, _diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        if self.start.elapsed().as_secs_f64() > self.limit_secs {
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
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        let current = diag.entropy;
        if let Some(prev) = self.prev_entropy.get() {
            let dt = diag.time; // Approximate: use absolute time as proxy
            let rate = if dt > 1e-30 {
                (current - prev).abs() / dt
            } else {
                f64::MAX
            };
            self.prev_entropy.set(Some(current));
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
    fn check(&self, _diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        // CFL violation is checked in Simulation::step via max_dt; this condition is a fallback
        None
    }
}

/// Exit when the first caustic forms (max stream count > 1).
pub struct CausticFormationCondition;

impl ExitCondition for CausticFormationCondition {
    fn check(&self, _diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
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
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        let count = self.step_count.get();
        self.step_count.set(count + 1);
        if count < self.min_steps {
            return None;
        }
        if (diag.virial_ratio - 1.0).abs() < self.tolerance {
            Some(ExitReason::VirialRelaxed)
        } else {
            None
        }
    }
}
