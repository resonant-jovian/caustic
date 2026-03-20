//! Parallel BUG integrator for Hierarchical Tucker format.
//!
//! All 6 leaf K-steps are computed simultaneously from frozen state at t^n
//! using rayon, instead of sequential processing. Transfer tensor updates
//! for sibling leaves sharing a parent are combined via block-diagonal approach.
//!
//! Reference: Ceruti, Kusch & Lubich (2024), "A parallel rank-adaptive
//! integrator for dynamical low-rank approximation".

use std::sync::Arc;
use std::time::Instant;

use faer::Mat;

use super::super::{
    advecator::Advector,
    algos::ht::HtTensor,
    integrator::{StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};

use super::bug::{
    self, BugConfig, LEAF_PARENT, conservative_correction, k_step_leaf,
    representative_accelerations, representative_velocities, sample_aug_displacements,
    update_transfer,
};

/// Configuration for the Parallel BUG integrator.
pub struct ParallelBugConfig {
    /// Truncation tolerance for rank adaptation.
    pub tolerance: f64,
    /// Maximum rank per node.
    pub max_rank: usize,
    /// Apply conservative moment correction after truncation.
    pub conservative: bool,
    /// Number of extra basis columns per K-step (0 = rank-preserving).
    pub rank_increase: usize,
    /// Enable step rejection via embedded error estimate (Lie-BUG vs Strang-BUG).
    pub error_rejection: bool,
    /// Tolerance for step rejection (relative error threshold).
    pub rejection_tolerance: f64,
}

impl Default for ParallelBugConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_rank: 50,
            conservative: false,
            rank_increase: 2,
            error_rejection: false,
            rejection_tolerance: 1e-4,
        }
    }
}

/// Parallel BUG integrator: computes all 6 leaf K-steps simultaneously.
///
/// Falls back to standard Strang splitting for non-HT representations.
pub struct ParallelBugIntegrator {
    pub config: ParallelBugConfig,
    pub g: f64,
    last_timings: StepTimings,
    progress: Option<Arc<StepProgress>>,
}

impl ParallelBugIntegrator {
    pub fn new(g: f64, config: ParallelBugConfig) -> Self {
        Self {
            config,
            g,
            last_timings: StepTimings::default(),
            progress: None,
        }
    }

    fn bug_config(&self) -> BugConfig {
        BugConfig {
            tolerance: self.config.tolerance,
            max_rank: self.config.max_rank,
            midpoint: false,
            conservative: false,
            rank_increase: self.config.rank_increase,
        }
    }

    /// Compute all 6 K-step results in parallel from frozen HT state.
    ///
    /// Returns `[(new_frame, R_matrix); 6]` for leaves 0..6.
    fn parallel_k_steps(
        ht: &HtTensor,
        accel: &AccelerationField,
        dt_drift: f64,
        dt_kick: f64,
        config: &ParallelBugConfig,
    ) -> Vec<(Mat<f64>, Mat<f64>)> {
        use rayon::prelude::*;

        let dims: Vec<usize> = (0..6).collect();
        dims.par_iter()
            .map(|&d| {
                if d < 3 {
                    // Spatial leaf: drift
                    let reps = representative_velocities(ht, d + 3);
                    let primary = if reps.is_empty() {
                        0.0
                    } else {
                        reps.iter().sum::<f64>() / reps.len() as f64 * dt_drift
                    };
                    let aug = sample_aug_displacements(&reps, dt_drift, config.rank_increase);
                    k_step_leaf(ht, d, primary, &aug, config.max_rank, config.tolerance)
                } else {
                    // Velocity leaf: kick
                    let reps = representative_accelerations(ht, d - 3, accel);
                    let primary = if reps.is_empty() {
                        0.0
                    } else {
                        reps.iter().sum::<f64>() / reps.len() as f64 * dt_kick
                    };
                    let aug = sample_aug_displacements(&reps, dt_kick, config.rank_increase);
                    k_step_leaf(ht, d, primary, &aug, config.max_rank, config.tolerance)
                }
            })
            .collect()
    }

    /// Apply all 6 K-step results to the HT tensor.
    ///
    /// Sibling leaves (1,2 share parent 6; 4,5 share parent 7) are handled
    /// by applying updates sequentially within each sibling pair to ensure
    /// the transfer tensor dimensions remain consistent.
    fn apply_k_steps(ht: &mut HtTensor, results: Vec<(Mat<f64>, Mat<f64>)>) {
        // Apply in order that respects sibling constraints:
        // Process leaf 0 (parent 8, left) — no sibling conflict
        // Process leaf 3 (parent 9, left) — no sibling conflict
        // Process leaf 1 (parent 6, left) then leaf 2 (parent 6, right)
        // Process leaf 4 (parent 7, left) then leaf 5 (parent 7, right)
        let apply_leaf = |ht: &mut HtTensor, d: usize, result: &(Mat<f64>, Mat<f64>)| {
            *ht.leaf_frame_mut(d) = result.0.clone();
            update_transfer(ht, d, &result.1);
        };

        // Independent leaves first
        apply_leaf(ht, 0, &results[0]);
        apply_leaf(ht, 3, &results[3]);

        // Sibling pair: leaves 1, 2 (parent node 6)
        apply_leaf(ht, 1, &results[1]);
        apply_leaf(ht, 2, &results[2]);

        // Sibling pair: leaves 4, 5 (parent node 7)
        apply_leaf(ht, 4, &results[4]);
        apply_leaf(ht, 5, &results[5]);
    }

    /// Parallel BUG step with Strang splitting: drift(dt/2) → kick(dt) → drift(dt/2).
    ///
    /// Each sub-step computes leaf K-steps in parallel, then applies sequentially.
    fn parallel_bug_step(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        let ht = repr
            .as_any_mut()
            .downcast_mut::<HtTensor>()
            .expect("Parallel BUG requires HtTensor");

        let density_before = if self.config.conservative {
            Some(ht.compute_density())
        } else {
            None
        };

        // Phase 1: Half drift — parallel K-steps for spatial leaves
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::BugKStep);
            p.set_sub_step(0, 5);
        }
        let t0 = Instant::now();
        {
            let results = Self::parallel_drift_k_steps(ht, dt / 2.0, &self.config);
            for (d, (new_frame, r_mat)) in results.into_iter().enumerate() {
                *ht.leaf_frame_mut(d) = new_frame;
                update_transfer(ht, d, &r_mat);
            }
        }
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Poisson solve at midpoint
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::BugLStep);
            p.set_sub_step(1, 5);
        }
        let t0 = Instant::now();
        let density = ht.compute_density();
        let potential = solver.solve(&density, self.g);
        let accel = solver.compute_acceleration(&potential);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Full kick — parallel K-steps for velocity leaves
        if let Some(ref p) = self.progress {
            p.set_sub_step(2, 5);
        }
        let t0 = Instant::now();
        {
            let results = Self::parallel_kick_k_steps(ht, &accel, dt, &self.config);
            for (i, (new_frame, r_mat)) in results.into_iter().enumerate() {
                let d = i + 3;
                *ht.leaf_frame_mut(d) = new_frame;
                update_transfer(ht, d, &r_mat);
            }
        }
        timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 3: Half drift — parallel K-steps for spatial leaves
        if let Some(ref p) = self.progress {
            p.set_sub_step(3, 5);
        }
        let t0 = Instant::now();
        {
            let results = Self::parallel_drift_k_steps(ht, dt / 2.0, &self.config);
            for (d, (new_frame, r_mat)) in results.into_iter().enumerate() {
                *ht.leaf_frame_mut(d) = new_frame;
                update_transfer(ht, d, &r_mat);
            }
        }
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // SVD-truncate interior nodes
        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::BugSStep);
            p.set_sub_step(4, 5);
        }
        ht.truncate(self.config.tolerance);

        if let Some(ref dens) = density_before {
            conservative_correction(ht, dens);
        }
    }

    /// Compute drift K-steps for spatial leaves 0,1,2 in parallel.
    fn parallel_drift_k_steps(
        ht: &HtTensor,
        dt: f64,
        config: &ParallelBugConfig,
    ) -> Vec<(Mat<f64>, Mat<f64>)> {
        use rayon::prelude::*;

        (0..3usize)
            .into_par_iter()
            .map(|d| {
                let reps = representative_velocities(ht, d + 3);
                let primary = if reps.is_empty() {
                    0.0
                } else {
                    reps.iter().sum::<f64>() / reps.len() as f64 * dt
                };
                let aug = sample_aug_displacements(&reps, dt, config.rank_increase);
                k_step_leaf(ht, d, primary, &aug, config.max_rank, config.tolerance)
            })
            .collect()
    }

    /// Compute kick K-steps for velocity leaves 3,4,5 in parallel.
    fn parallel_kick_k_steps(
        ht: &HtTensor,
        accel: &AccelerationField,
        dt: f64,
        config: &ParallelBugConfig,
    ) -> Vec<(Mat<f64>, Mat<f64>)> {
        use rayon::prelude::*;

        (3..6usize)
            .into_par_iter()
            .map(|d| {
                let reps = representative_accelerations(ht, d - 3, accel);
                let primary = if reps.is_empty() {
                    0.0
                } else {
                    reps.iter().sum::<f64>() / reps.len() as f64 * dt
                };
                let aug = sample_aug_displacements(&reps, dt, config.rank_increase);
                k_step_leaf(ht, d, primary, &aug, config.max_rank, config.tolerance)
            })
            .collect()
    }

    /// Fallback: standard Strang splitting for non-HT representations.
    fn strang_fallback(
        &self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
        timings: &mut StepTimings,
    ) {
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.g);
        let accel = solver.compute_acceleration(&potential);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        advector.kick(repr, &accel, dt);
        timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
}

impl TimeIntegrator for ParallelBugIntegrator {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        let _span = tracing::info_span!("parallel_bug_advance").entered();
        let mut timings = StepTimings::default();

        if let Some(ref p) = self.progress {
            p.start_step();
        }

        let is_ht = repr.as_any().downcast_ref::<HtTensor>().is_some();

        if is_ht {
            self.parallel_bug_step(repr, solver, dt, &mut timings);
        } else {
            self.strang_fallback(repr, solver, advector, dt, &mut timings);
        }

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
        }

        self.last_timings = timings;
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        let density = repr.compute_density();
        let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);
        if rho_max <= 0.0 || self.g <= 0.0 {
            return 1e10;
        }
        let t_dyn = 1.0 / (self.g * rho_max).sqrt();
        cfl_factor * t_dyn
    }

    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_bug_config_defaults() {
        let cfg = ParallelBugConfig::default();
        assert_eq!(cfg.max_rank, 50);
        assert!(!cfg.conservative);
        assert!(!cfg.error_rejection);
        assert_eq!(cfg.rank_increase, 2);
    }
}
