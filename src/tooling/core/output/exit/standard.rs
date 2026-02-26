//! Standard exit condition evaluation and reporting.

use super::super::super::{
    conditions::{ExitCondition, ExitReason},
    diagnostics::GlobalDiagnostics,
};

/// Evaluates all active exit conditions each timestep.
pub struct ExitEvaluator {
    pub conditions: Vec<Box<dyn ExitCondition>>,
    pub initial: GlobalDiagnostics,
}

impl ExitEvaluator {
    pub fn new(initial: GlobalDiagnostics) -> Self {
        todo!()
    }

    pub fn add_condition(&mut self, cond: Box<dyn ExitCondition>) {
        todo!()
    }

    /// Evaluate all conditions and return the first triggered reason.
    pub fn check(&self, current: &GlobalDiagnostics) -> Option<ExitReason> {
        todo!("evaluate all conditions, return first Some(reason)")
    }
}
