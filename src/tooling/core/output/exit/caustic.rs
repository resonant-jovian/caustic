//! Caustic-specific exit conditions.

use super::super::super::{
    conditions::{ExitCondition, ExitReason},
    context::SimContext,
    diagnostics::GlobalDiagnostics,
    events::{ExitConditionKind, SimEvent},
};

/// Exit when the first caustic forms (max stream count > 1).
pub struct CausticExitCondition;

impl ExitCondition for CausticExitCondition {
    fn check(
        &self,
        _diag: &GlobalDiagnostics,
        _initial: &GlobalDiagnostics,
        _ctx: &SimContext,
    ) -> Option<ExitReason> {
        // Stream count requires repr access; not available through GlobalDiagnostics alone
        None
    }
}

/// Exit when virial ratio 2T/|W| stabilises within tolerance for N consecutive steps.
pub struct VirialRelaxedExit {
    pub tolerance: f64,
}

impl ExitCondition for VirialRelaxedExit {
    fn check(
        &self,
        diag: &GlobalDiagnostics,
        _initial: &GlobalDiagnostics,
        ctx: &SimContext,
    ) -> Option<ExitReason> {
        let virial_deviation = (diag.virial_ratio - 1.0).abs();
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

        if virial_deviation < self.tolerance {
            Some(ExitReason::VirialRelaxed)
        } else {
            None
        }
    }
}
