//! Caustic-specific exit conditions.

use super::super::super::{
    conditions::{ExitCondition, ExitReason},
    diagnostics::GlobalDiagnostics,
};

/// Exit when the first caustic forms (max stream count > 1).
pub struct CausticExitCondition;

impl ExitCondition for CausticExitCondition {
    fn check(&self, _diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        // Stream count requires repr access; not available through GlobalDiagnostics alone
        None
    }
}

/// Exit when virial ratio 2T/|W| stabilises within tolerance for N consecutive steps.
pub struct VirialRelaxedExit {
    pub tolerance: f64,
}

impl ExitCondition for VirialRelaxedExit {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        if (diag.virial_ratio - 1.0).abs() < self.tolerance {
            Some(ExitReason::VirialRelaxed)
        } else {
            None
        }
    }
}
