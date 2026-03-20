#![allow(non_ascii_idents)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// Feature-gated global allocators
#[cfg(feature = "jemalloc")]
#[allow(unsafe_code)]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "mimalloc-alloc")]
#[allow(unsafe_code)]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "dhat-heap")]
#[allow(unsafe_code)]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub(crate) mod sim;
pub mod tooling;

pub use sim::Simulation;
pub use tooling::core::advecator::Advector;
pub use tooling::core::conditions::ExitReason;
pub use tooling::core::init::domain::{Domain, DomainBuilder};
pub use tooling::core::integrator::{StepTimings, TimeIntegrator};
pub use tooling::core::phasespace::PhaseSpaceRepr;
pub use tooling::core::solver::PoissonSolver;
pub use tooling::core::types::*;

// Concrete implementations — needed by phasma and other binary consumers
pub use tooling::core::algos::ht::HtTensor;
pub use tooling::core::algos::lagrangian::SemiLagrangian;
pub use tooling::core::conditions::{
    CasimirDriftCondition, CausticFormationCondition, CflViolationCondition, ExitCondition,
    MassLossCondition, SteadyStateCondition, VirialRelaxedCondition, WallClockCondition,
};
pub use tooling::core::diagnostics::GlobalDiagnostics;
pub use tooling::core::init::arbitrary::CustomICArray;
pub use tooling::core::init::cosmological::{ZeldovichIC, ZeldovichSingleMode};
pub use tooling::core::init::domain::{SpatialBoundType, VelocityBoundType};
pub use tooling::core::init::isolated::{
    HernquistIC, IsolatedEquilibrium, KingIC, NfwIC, PlummerIC, sample_on_grid,
    sample_on_grid_with_progress,
};
pub use tooling::core::init::mergers::MergerIC;
pub use tooling::core::init::stability::DiskStabilityIC;
pub use tooling::core::init::tidal::TidalIC;
pub use tooling::core::output::exit::standard::ExitEvaluator;
pub use tooling::core::output::phasespace::{PhaseSpaceDiagnostics, field_energy_spectrum};
pub use tooling::core::poisson::fft::{FftIsolated, FftPoisson};
pub use tooling::core::poisson::ht_poisson::HtPoisson;
pub use tooling::core::poisson::multigrid::Multigrid;
pub use tooling::core::poisson::spherical::SphericalHarmonicsPoisson;
pub use tooling::core::poisson::tensor_poisson::TensorPoisson;
pub use tooling::core::poisson::tree::TreePoisson;
pub use tooling::core::time::lie::LieSplitting;
pub use tooling::core::time::rank_monitor::{InstrumentedStrangSplitting, StepRankDiagnostics};
pub use tooling::core::time::rkei::RkeiIntegrator;
pub use tooling::core::time::strang::StrangSplitting;
pub use tooling::core::time::unsplit::UnsplitIntegrator;
pub use tooling::core::time::yoshida::YoshidaSplitting;

// Progress tracking (shared atomics for TUI intra-step visibility)
pub use tooling::core::progress::{ProgressSnapshot, StepPhase, StepProgress};

// Phase-space representations
pub use tooling::core::algos::amr::AmrGrid;
pub use tooling::core::algos::hybrid::HybridRepr;
pub use tooling::core::algos::sheet::SheetTracker;
pub use tooling::core::algos::spectral::SpectralV;
pub use tooling::core::algos::tensor::TensorTrain;
pub use tooling::core::algos::uniform::UniformGrid6D;

// Conservation framework (Phase 4: LoMaC)
pub use tooling::core::conservation::conservative_svd;
pub use tooling::core::conservation::kfvs::{KfvsSolver, MacroState};
pub use tooling::core::conservation::lomac::LoMaC;
pub use tooling::core::conservation::rank_adaptive::RankAdaptiveController;

/// Top-level error type for caustic operations.
#[derive(Debug, thiserror::Error)]
pub enum CausticError {
    /// Domain validation failed.
    #[error("Domain validation failed: {0}")]
    Domain(String),
    /// Solver error.
    #[error("Solver error: {0}")]
    Solver(String),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Exit condition triggered.
    #[error("Exit condition triggered: {0:?}")]
    Exit(ExitReason),
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke_test() {
        use crate::sim::Simulation;
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::init::{
            domain::{Domain, SpatialBoundType, VelocityBoundType},
            isolated::{PlummerIC, sample_on_grid},
        };
        use crate::tooling::core::poisson::fft::FftPoisson;
        use crate::tooling::core::time::strang::StrangSplitting;

        let domain = Domain::builder()
            .spatial_extent(10.0)
            .velocity_extent(5.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = sample_on_grid(&ic, &domain);

        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain)
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new(1.0))
            .initial_conditions(snap)
            .time_final(0.1)
            .build()
            .unwrap();

        sim.step().unwrap();
        assert!(
            sim.current_time() > 0.0,
            "simulation time should advance after one step"
        );
    }

    /// Full pipeline: Domain → PlummerIC → SimulationBuilder → run() → ExitPackage.
    /// Exercises every component in the primary code path from start to finish.
    #[test]
    fn end_to_end_run() {
        use crate::sim::Simulation;
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::conditions::ExitReason;
        use crate::tooling::core::init::{
            domain::{Domain, SpatialBoundType, VelocityBoundType},
            isolated::{PlummerIC, sample_on_grid},
        };
        use crate::tooling::core::poisson::fft::FftPoisson;
        use crate::tooling::core::time::strang::StrangSplitting;

        // ── 1. Domain ─────────────────────────────────────────────────────
        let domain = Domain::builder()
            .spatial_extent(8.0)
            .velocity_extent(2.5)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        // ── 2. Initial conditions ─────────────────────────────────────────
        let ic = PlummerIC::new(1.0, 1.0, 1.0); // M=1, a=1, G=1
        let snap = sample_on_grid(&ic, &domain);
        assert!(
            !snap.data.iter().any(|f| f.is_nan()),
            "IC snapshot must not contain NaN"
        );
        let mass_ic: f64 = snap.data.iter().sum::<f64>() * {
            let dx = domain.dx();
            let dv = domain.dv();
            dx[0] * dx[1] * dx[2] * dv[0] * dv[1] * dv[2]
        };
        assert!(
            mass_ic > 0.0,
            "Sampled IC must have positive mass, got {mass_ic}"
        );

        // ── 3. Build Simulation ───────────────────────────────────────────
        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain)
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new(1.0))
            .initial_conditions(snap)
            .time_final(1.0)
            .build()
            .unwrap();

        // Initial diagnostics must be sane
        assert!(
            !sim.diagnostics.history.is_empty(),
            "Initial diagnostics must be recorded"
        );
        let d0 = &sim.diagnostics.history[0];
        assert!(!d0.total_energy.is_nan(), "Initial energy must not be NaN");
        assert!(d0.casimir_c2 >= 0.0, "Initial C2 must be non-negative");

        // ── 4. Run to completion ──────────────────────────────────────────
        let pkg = sim.run().unwrap();

        // ── 5. Verify ExitPackage ─────────────────────────────────────────
        assert!(
            matches!(pkg.exit_reason, ExitReason::TimeLimitReached),
            "Expected TimeLimitReached, got {:?}",
            pkg.exit_reason
        );
        assert!(pkg.total_steps > 0, "Must have taken at least one step");
        assert!(
            !pkg.final_snapshot.data.iter().any(|f| f.is_nan()),
            "Final snapshot must not contain NaN"
        );
        assert!(
            !pkg.diagnostics_history.is_empty(),
            "Diagnostics history must not be empty"
        );

        let final_diag = pkg.diagnostics_history.last().unwrap();
        assert!(
            !final_diag.total_energy.is_nan(),
            "Final energy must not be NaN"
        );
        assert!(final_diag.time > 0.0, "Final time must be positive");

        // ── 6. Conservation summary ───────────────────────────────────────
        pkg.print_summary();
        assert!(
            pkg.conservation_summary.max_energy_drift.is_finite(),
            "Energy drift must be finite"
        );

        // ── 7. IO: save snapshot to temp directory ────────────────────────
        use crate::tooling::core::io::{IOManager, OutputFormat};
        let tmp = std::env::temp_dir().join("caustic_e2e_test");
        let io_mgr = IOManager::new(tmp.to_str().unwrap(), OutputFormat::Binary);
        io_mgr
            .save_snapshot(&pkg.final_snapshot, "snap_final.bin")
            .unwrap();
        assert!(
            tmp.join("snap_final.bin").exists(),
            "Snapshot file must exist"
        );
        io_mgr.append_diagnostics(final_diag).unwrap();
        assert!(
            tmp.join("diagnostics.csv").exists(),
            "Diagnostics CSV must exist"
        );

        // Cleanup
        let _ = std::fs::remove_dir_all(tmp);

        println!(
            "End-to-end: {} steps, t_final={:.3}, E_drift={:.2e}",
            pkg.total_steps, final_diag.time, pkg.conservation_summary.max_energy_drift
        );
    }
}

/// Convenience re-exports for the most commonly used items.
pub mod prelude {
    pub use crate::tooling::core::{
        conditions::ExitCondition,
        diagnostics::{Diagnostics, GlobalDiagnostics},
        init::input::optional::OptionalParams,
        init::{
            cosmological::ZeldovichIC,
            isolated::{HernquistIC, IsolatedEquilibrium, KingIC, NfwIC, PlummerIC},
            mergers::MergerIC,
        },
        integrator::SimState,
        io::IOManager,
        output::exit::package::ExitPackage,
        types::*,
    };
    pub use crate::{
        Advector, CausticError, Domain, DomainBuilder, ExitReason, PhaseSpaceRepr, PoissonSolver,
        Simulation, TimeIntegrator,
    };
}
