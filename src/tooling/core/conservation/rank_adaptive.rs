//! Adaptive rank control with conservation-aware thresholds.
//!
//! Manages the trade-off between compression quality (low rank) and
//! physical fidelity (conservation of mass, momentum, energy). The
//! controller monitors conservation errors after each time step and
//! adjusts the truncation tolerance and maximum rank accordingly.
//!
//! Strategy from caustic.md Section 6:
//! - When conservation error < ε_loose: decrease rank (loosen tolerance)
//! - When conservation error > ε_tight: increase rank (tighten tolerance)
//! - When rank hits budget r_max: switch to budget-driven truncation
//!
//! Reference: caustic.md Section 6, Ceruti, Kusch & Lubich arXiv:2107.08834

/// Adaptive rank controller for HT tensor truncation.
pub struct RankAdaptiveController {
    /// Current truncation tolerance.
    pub tolerance: f64,
    /// Maximum allowed rank (memory budget).
    pub rank_budget: usize,
    /// Minimum allowed tolerance (accuracy floor).
    pub min_tolerance: f64,
    /// Maximum allowed tolerance (compression ceiling).
    pub max_tolerance: f64,
    /// Conservation error threshold above which rank increases.
    pub conservation_tight: f64,
    /// Conservation error threshold below which rank can decrease.
    pub conservation_loose: f64,
    /// Factor by which to tighten tolerance when conservation fails.
    pub tighten_factor: f64,
    /// Factor by which to loosen tolerance when over-resolving.
    pub loosen_factor: f64,
    /// Current observed maximum rank across all HT nodes.
    pub current_max_rank: usize,
    /// Whether rank budget has been hit (budget-driven mode).
    pub budget_saturated: bool,
    /// History of truncation errors for monitoring.
    pub truncation_errors: Vec<f64>,
    /// History of conservation errors for monitoring.
    pub conservation_errors: Vec<(f64, f64, f64)>, // (mass, momentum, energy)
}

impl RankAdaptiveController {
    /// Create with default parameters suitable for gravitational dynamics.
    pub fn new(rank_budget: usize) -> Self {
        Self {
            tolerance: 1e-6,
            rank_budget,
            min_tolerance: 1e-12,
            max_tolerance: 1e-2,
            conservation_tight: 1e-6,
            conservation_loose: 1e-8,
            tighten_factor: 0.5,
            loosen_factor: 2.0,
            current_max_rank: 1,
            budget_saturated: false,
            truncation_errors: Vec::new(),
            conservation_errors: Vec::new(),
        }
    }

    /// Create with custom conservation thresholds.
    pub fn with_thresholds(
        rank_budget: usize,
        initial_tolerance: f64,
        conservation_tight: f64,
        conservation_loose: f64,
    ) -> Self {
        let mut ctrl = Self::new(rank_budget);
        ctrl.tolerance = initial_tolerance;
        ctrl.conservation_tight = conservation_tight;
        ctrl.conservation_loose = conservation_loose;
        ctrl
    }

    /// Update the controller after a time step, given conservation errors.
    ///
    /// Returns the new truncation tolerance to use for the next step.
    ///
    /// # Arguments
    /// * `mass_error` - |ΔM/M| relative mass conservation error
    /// * `momentum_error` - |ΔP/P| relative momentum conservation error
    /// * `energy_error` - |ΔE/E| relative energy conservation error
    /// * `max_rank` - maximum rank observed across all HT nodes after truncation
    /// * `truncation_error` - Frobenius norm error from the last truncation
    pub fn update(
        &mut self,
        mass_error: f64,
        momentum_error: f64,
        energy_error: f64,
        max_rank: usize,
        truncation_error: f64,
    ) -> f64 {
        self.current_max_rank = max_rank;
        self.truncation_errors.push(truncation_error);
        self.conservation_errors
            .push((mass_error, momentum_error, energy_error));

        let worst_conservation = mass_error.max(momentum_error).max(energy_error);

        // Check if rank budget is saturated
        self.budget_saturated = max_rank >= self.rank_budget;

        if self.budget_saturated {
            // Budget-driven mode: accept current tolerance, log warning
            // Don't tighten further since we can't add more ranks
            return self.tolerance;
        }

        if worst_conservation > self.conservation_tight {
            // Conservation failing: tighten tolerance → more ranks
            self.tolerance = (self.tolerance * self.tighten_factor).max(self.min_tolerance);
        } else if worst_conservation < self.conservation_loose {
            // Over-resolving: loosen tolerance → fewer ranks
            self.tolerance = (self.tolerance * self.loosen_factor).min(self.max_tolerance);
        }
        // else: in the acceptable range, keep current tolerance

        self.tolerance
    }

    /// Recommend whether to increase the rank budget (for checkpoint-and-refine).
    pub fn should_increase_budget(&self) -> bool {
        // Recommend budget increase if:
        // 1. Budget is saturated AND
        // 2. Conservation error is above tight threshold for the last 3 steps
        if !self.budget_saturated {
            return false;
        }
        let n = self.conservation_errors.len();
        if n < 3 {
            return false;
        }
        self.conservation_errors[n - 3..]
            .iter()
            .all(|(m, p, e)| m.max(*p).max(*e) > self.conservation_tight)
    }

    /// Get summary statistics for diagnostics.
    pub fn summary(&self) -> RankAdaptiveSummary {
        let n = self.conservation_errors.len();
        let (last_mass, last_mom, last_energy) = if n > 0 {
            self.conservation_errors[n - 1]
        } else {
            (0.0, 0.0, 0.0)
        };

        RankAdaptiveSummary {
            tolerance: self.tolerance,
            max_rank: self.current_max_rank,
            rank_budget: self.rank_budget,
            budget_saturated: self.budget_saturated,
            last_mass_error: last_mass,
            last_momentum_error: last_mom,
            last_energy_error: last_energy,
            total_steps: n,
        }
    }
}

/// Summary of rank-adaptive controller state for diagnostics.
#[derive(Clone, Debug)]
pub struct RankAdaptiveSummary {
    /// Current truncation tolerance used for SVD compression.
    pub tolerance: f64,
    /// Current observed maximum rank across all HT nodes.
    pub max_rank: usize,
    /// Maximum allowed rank (memory budget).
    pub rank_budget: usize,
    /// Whether the rank budget is currently saturated.
    pub budget_saturated: bool,
    /// Relative mass conservation error from the most recent step.
    pub last_mass_error: f64,
    /// Relative momentum conservation error from the most recent step.
    pub last_momentum_error: f64,
    /// Relative energy conservation error from the most recent step.
    pub last_energy_error: f64,
    /// Total number of time steps processed by the controller.
    pub total_steps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_adaptive_tightens_on_conservation_error() {
        let mut ctrl = RankAdaptiveController::new(100);
        ctrl.tolerance = 1e-6;
        ctrl.conservation_tight = 1e-6;

        // Report large conservation error → tolerance should decrease
        let new_tol = ctrl.update(1e-4, 1e-5, 1e-5, 10, 1e-6);
        assert!(
            new_tol < 1e-6,
            "Tolerance should decrease on conservation error: {new_tol}"
        );
    }

    #[test]
    fn rank_adaptive_loosens_when_well_conserved() {
        let mut ctrl = RankAdaptiveController::new(100);
        ctrl.tolerance = 1e-8;
        ctrl.conservation_loose = 1e-8;

        // Report tiny conservation error → tolerance should increase
        let new_tol = ctrl.update(1e-12, 1e-12, 1e-12, 10, 1e-8);
        assert!(
            new_tol > 1e-8,
            "Tolerance should increase when over-resolving: {new_tol}"
        );
    }

    #[test]
    fn rank_adaptive_respects_budget() {
        let mut ctrl = RankAdaptiveController::new(20);
        ctrl.tolerance = 1e-6;

        // Report max_rank = budget → should not tighten further
        let tol_before = ctrl.tolerance;
        let new_tol = ctrl.update(1e-4, 1e-4, 1e-4, 20, 1e-6);
        assert_eq!(
            new_tol, tol_before,
            "Should not tighten when budget-saturated"
        );
        assert!(ctrl.budget_saturated);
    }

    #[test]
    fn rank_adaptive_budget_increase_recommendation() {
        let mut ctrl = RankAdaptiveController::new(20);
        ctrl.conservation_tight = 1e-6;

        // Simulate 3 steps at budget with poor conservation
        ctrl.update(1e-4, 1e-4, 1e-4, 20, 1e-6);
        ctrl.update(1e-4, 1e-4, 1e-4, 20, 1e-6);
        ctrl.update(1e-4, 1e-4, 1e-4, 20, 1e-6);

        assert!(
            ctrl.should_increase_budget(),
            "Should recommend budget increase after sustained saturation"
        );
    }

    #[test]
    fn rank_adaptive_tolerance_bounds() {
        let mut ctrl = RankAdaptiveController::new(100);
        ctrl.min_tolerance = 1e-10;
        ctrl.max_tolerance = 1e-3;
        ctrl.tolerance = 1e-10;

        // Try to tighten below minimum
        let new_tol = ctrl.update(1.0, 1.0, 1.0, 5, 1e-10);
        assert!(
            new_tol >= 1e-10,
            "Tolerance should not go below minimum: {new_tol}"
        );

        // Reset and try to loosen above maximum
        ctrl.tolerance = 1e-3;
        let new_tol = ctrl.update(1e-15, 1e-15, 1e-15, 5, 1e-3);
        assert!(
            new_tol <= 1e-3,
            "Tolerance should not go above maximum: {new_tol}"
        );
    }
}
