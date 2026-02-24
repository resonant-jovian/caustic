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
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason>;
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
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!("return EnergyDrift if |E-E0|/|E0| > tolerance")
    }
}

/// Exit when mass fraction drops below `threshold`.
pub struct MassLossCondition {
    pub threshold: f64,
}

impl ExitCondition for MassLossCondition {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!()
    }
}

/// Exit when relative Casimir drift exceeds `tolerance`.
pub struct CasimirDriftCondition {
    pub tolerance: f64,
}

impl ExitCondition for CasimirDriftCondition {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!()
    }
}

/// Exit when wall-clock time exceeds `limit_secs` seconds.
pub struct WallClockCondition {
    pub limit_secs: f64,
}

impl ExitCondition for WallClockCondition {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!()
    }
}

/// Exit when `‖∂f/∂t‖ < threshold` (steady state reached).
pub struct SteadyStateCondition {
    pub threshold: f64,
}

impl ExitCondition for SteadyStateCondition {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!()
    }
}

/// Exit when adaptive Δt drops below `dt_min`.
pub struct CflViolationCondition {
    pub dt_min: f64,
}

impl ExitCondition for CflViolationCondition {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!()
    }
}

/// Exit when the first caustic forms (max stream count > 1).
pub struct CausticFormationCondition;

impl ExitCondition for CausticFormationCondition {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!("check stream count field for max > 1")
    }
}

/// Exit when the virial ratio 2T/|W| stabilises within `tolerance` of 1.0.
pub struct VirialRelaxedCondition {
    pub tolerance: f64,
}

impl ExitCondition for VirialRelaxedCondition {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        initial: &GlobalDiagnostics,
    ) -> Option<ExitReason> {
        todo!()
    }
}
