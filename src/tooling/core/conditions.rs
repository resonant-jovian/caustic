//! Exit conditions — predicates evaluated after each timestep to determine whether
//! the simulation should terminate.

use super::diagnostics::GlobalDiagnostics;

/// Reason why the simulation terminated.
#[derive(Debug, Clone)]
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
    FirstCausticFormed,
    /// Virial ratio stabilised at 1.0 ± ε: violent relaxation complete.
    VirialRelaxed,
    /// User-defined predicate returned true.
    UserDefined,
}

/// Trait for simulation exit predicates. Evaluated after each timestep.
pub trait ExitCondition {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason>;
}

/// Exit when simulation time reaches `t_final`.
pub struct TimeLimitCondition {
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

/// Exit when relative energy drift exceeds `tolerance`.
pub struct EnergyDriftCondition {
    pub tolerance: f64,
}

impl ExitCondition for EnergyDriftCondition {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason> {
        let e0 = initial.total_energy.abs();
        if e0 > 1e-30 && (diag.total_energy - initial.total_energy).abs() / e0 > self.tolerance {
            Some(ExitReason::EnergyDrift)
        } else {
            None
        }
    }
}

/// Exit when mass fraction drops below `threshold`.
pub struct MassLossCondition {
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
    pub tolerance: f64,
}

impl ExitCondition for CasimirDriftCondition {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason> {
        let c0 = initial.casimir_c2.abs();
        if c0 > 1e-30 && (diag.casimir_c2 - initial.casimir_c2).abs() / c0 > self.tolerance {
            Some(ExitReason::CasimirDrift)
        } else {
            None
        }
    }
}

/// Exit when wall-clock time exceeds `limit_secs` seconds.
pub struct WallClockCondition {
    pub limit_secs: f64,
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
pub struct SteadyStateCondition {
    pub threshold: f64,
}

impl ExitCondition for SteadyStateCondition {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        // Approximated by entropy rate of change: if entropy is nearly constant, steady state
        // For now, placeholder — needs history of entropy values
        None
    }
}

/// Exit when adaptive Δt drops below `dt_min`.
pub struct CflViolationCondition {
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
        // Stream count field is not stored in GlobalDiagnostics; requires repr access
        // For now: not implemented as a standalone exit condition
        None
    }
}

/// Exit when the virial ratio 2T/|W| stabilises within `tolerance` of 1.0.
pub struct VirialRelaxedCondition {
    pub tolerance: f64,
}

impl ExitCondition for VirialRelaxedCondition {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        if (diag.virial_ratio - 1.0).abs() < self.tolerance {
            Some(ExitReason::VirialRelaxed)
        } else {
            None
        }
    }
}
