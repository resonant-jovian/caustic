//! Step-level rank diagnostics for monitoring HT tensor rank amplification.
//!
//! `InstrumentedStrangSplitting` wraps `StrangSplitting` and records ranks
//! before/after each sub-step (drift, kick, Poisson). This lets us attribute
//! rank growth to specific algorithmic phases.

use std::time::Instant;

use super::super::{
    advecator::Advector,
    integrator::{StepTimings, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    solver::PoissonSolver,
};

/// Per-step rank diagnostics, populated by `InstrumentedStrangSplitting`.
/// Fields are `None` when the representation is not `HtTensor`.
#[derive(Clone, Debug, Default)]
pub struct StepRankDiagnostics {
    /// Ranks of each HT node before the first drift half-step.
    pub pre_drift_ranks: Option<Vec<usize>>,
    /// Ranks after drift(Δt/2).
    pub post_drift_ranks: Option<Vec<usize>>,
    /// Ranks after the velocity kick (Poisson + advect_v).
    pub post_kick_ranks: Option<Vec<usize>>,
    /// Ranks after the second drift half-step (final state).
    pub post_final_ranks: Option<Vec<usize>>,
    /// Poisson rank amplification: max(post_kick_rank) / max(post_drift_rank).
    pub poisson_rank_amplification: Option<f64>,
    /// Advection rank amplification: max(post_drift_rank) / max(pre_drift_rank).
    pub advection_rank_amplification: Option<f64>,
}

/// Extract per-node ranks from a `PhaseSpaceRepr` if it's an `HtTensor`.
fn extract_ranks(repr: &dyn PhaseSpaceRepr) -> Option<Vec<usize>> {
    use super::super::algos::ht::HtTensor;
    let any = repr.as_any();
    any.downcast_ref::<HtTensor>()
        .map(|ht| ht.nodes.iter().map(|n| n.rank()).collect())
}

fn max_rank(ranks: &[usize]) -> f64 {
    ranks.iter().copied().max().unwrap_or(1).max(1) as f64
}

/// Instrumented version of `StrangSplitting` that records per-sub-step ranks and timings.
pub struct InstrumentedStrangSplitting {
    pub inner: super::strang::StrangSplitting,
    /// The diagnostics from the most recent `advance()` call.
    pub last_diagnostics: StepRankDiagnostics,
    last_timings: StepTimings,
}

impl InstrumentedStrangSplitting {
    pub fn new(g: f64) -> Self {
        Self {
            inner: super::strang::StrangSplitting::new(g),
            last_diagnostics: StepRankDiagnostics::default(),
            last_timings: StepTimings::default(),
        }
    }
}

impl TimeIntegrator for InstrumentedStrangSplitting {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) {
        let _span = tracing::info_span!("instrumented_strang_advance").entered();
        let mut timings = StepTimings::default();

        let mut diag = StepRankDiagnostics {
            pre_drift_ranks: extract_ranks(&*repr),
            ..Default::default()
        };

        // Drift half-step
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        diag.post_drift_ranks = extract_ranks(&*repr);

        // Poisson solve + kick
        let t0 = Instant::now();
        let density = repr.compute_density();
        let potential = solver.solve(&density, self.inner.g);
        let accel = solver.compute_acceleration(&potential);
        timings.poisson_ms += t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        advector.kick(repr, &accel, dt);
        timings.kick_ms += t0.elapsed().as_secs_f64() * 1000.0;
        diag.post_kick_ranks = extract_ranks(&*repr);

        // Drift half-step
        let t0 = Instant::now();
        advector.drift(repr, dt / 2.0);
        timings.drift_ms += t0.elapsed().as_secs_f64() * 1000.0;
        diag.post_final_ranks = extract_ranks(&*repr);

        // Compute amplification ratios
        if let (Some(pre), Some(post_drift)) = (&diag.pre_drift_ranks, &diag.post_drift_ranks) {
            let r_pre = max_rank(pre);
            let r_post = max_rank(post_drift);
            diag.advection_rank_amplification = Some(r_post / r_pre);
        }

        if let (Some(post_drift), Some(post_kick)) = (&diag.post_drift_ranks, &diag.post_kick_ranks)
        {
            let r_drift = max_rank(post_drift);
            let r_kick = max_rank(post_kick);
            diag.poisson_rank_amplification = Some(r_kick / r_drift);
        }

        self.last_diagnostics = diag;
        self.last_timings = timings;
    }

    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        self.inner.max_dt(repr, cfl_factor)
    }

    fn last_step_timings(&self) -> Option<&StepTimings> {
        Some(&self.last_timings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instrumented_strang_on_uniform_grid() {
        // When repr is UniformGrid6D, all rank fields should be None
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
        use crate::tooling::core::poisson::fft::FftPoisson;

        let domain = Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let mut grid = UniformGrid6D::new(domain.clone());
        // Fill with small constant to avoid trivial density
        for v in grid.data.iter_mut() {
            *v = 1.0;
        }

        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut integrator = InstrumentedStrangSplitting::new(1.0);

        integrator.advance(&mut grid, &poisson, &advector, 0.01);

        // UniformGrid6D is not HtTensor, so all rank fields should be None
        assert!(integrator.last_diagnostics.pre_drift_ranks.is_none());
        assert!(integrator.last_diagnostics.post_drift_ranks.is_none());
        assert!(integrator.last_diagnostics.post_kick_ranks.is_none());
        assert!(
            integrator
                .last_diagnostics
                .poisson_rank_amplification
                .is_none()
        );
        assert!(
            integrator
                .last_diagnostics
                .advection_rank_amplification
                .is_none()
        );
    }
}
