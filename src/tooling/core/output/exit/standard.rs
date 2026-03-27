//! Standard exit condition evaluation and reporting.
//!
//! Provides [`ExitEvaluator`], which aggregates a set of [`ExitCondition`] trait
//! objects and checks them against the current simulation diagnostics each timestep.
//! The first condition that triggers terminates the simulation with an [`ExitReason`].
//!
//! Typical conditions include energy drift, mass loss, wall-clock limits, and
//! steady-state detection. Conditions are compared against the initial diagnostics
//! snapshot captured at simulation start.

use super::super::super::{
    conditions::{ExitCondition, ExitReason},
    context::SimContext,
    diagnostics::GlobalDiagnostics,
};

/// Aggregates and evaluates all registered exit conditions each timestep.
///
/// Holds a snapshot of the initial diagnostics so each condition can compute
/// drift relative to the simulation's starting state.
pub struct ExitEvaluator {
    /// Registered exit conditions, checked in order until one triggers.
    pub conditions: Vec<Box<dyn ExitCondition>>,
    /// Diagnostics snapshot from the first timestep, used as the reference baseline.
    pub initial: GlobalDiagnostics,
}

impl ExitEvaluator {
    /// Create a new evaluator with the given initial diagnostics and no conditions.
    pub fn new(initial: GlobalDiagnostics) -> Self {
        Self {
            conditions: vec![],
            initial,
        }
    }

    /// Register an exit condition to be checked each timestep.
    pub fn add_condition(&mut self, cond: Box<dyn ExitCondition>) {
        self.conditions.push(cond);
    }

    /// Check all registered conditions against the current diagnostics.
    ///
    /// Returns the first triggered [`ExitReason`], or `None` if the simulation should continue.
    pub fn check(&self, current: &GlobalDiagnostics, ctx: &SimContext) -> Option<ExitReason> {
        self.conditions
            .iter()
            .find_map(|c| c.check(current, &self.initial, ctx))
    }
}
