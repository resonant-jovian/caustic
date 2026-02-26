//! Caustic-specific exit conditions.

use super::super::super::{
    conditions::{ExitCondition, ExitReason},
    diagnostics::GlobalDiagnostics,
};

/// Exit when the first caustic forms (max stream count > 1).
pub struct CausticExitCondition;

impl ExitCondition for CausticExitCondition {
    fn check(&self, diag: &GlobalDiagnostics, _initial: &GlobalDiagnostics) -> Option<ExitReason> {
        todo!("check stream count field for max > 1")
    }
}

/// Exit when virial ratio 2T/|W| stabilises within tolerance for N consecutive steps.
pub struct VirialRelaxedExit {
    pub tolerance: f64,
}

impl ExitCondition for VirialRelaxedExit {
    fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason> {
        todo!("exit when 2T/|W| stabilises within tolerance for N consecutive steps")
    }
}
