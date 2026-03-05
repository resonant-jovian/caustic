#![allow(non_ascii_idents)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub(crate) mod tooling;
pub(crate) mod sim;

pub use tooling::core::types::*;
pub use tooling::core::phasespace::PhaseSpaceRepr;
pub use tooling::core::solver::PoissonSolver;
pub use tooling::core::advecator::Advector;
pub use tooling::core::integrator::TimeIntegrator;
pub use tooling::core::init::domain::{Domain, DomainBuilder};
pub use tooling::core::conditions::ExitReason;
pub use sim::Simulation;

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
        use crate::tooling::core::init::{
            domain::{Domain, SpatialBoundType, VelocityBoundType},
            isolated::{PlummerIC, sample_on_grid},
        };
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::poisson::fft::FftPoisson;
        use crate::tooling::core::time::strang::StrangSplitting;
        use crate::sim::Simulation;

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
        assert!(sim.current_time() > 0.0, "simulation time should advance after one step");
    }
}

/// Convenience re-exports for the most commonly used items.
pub mod prelude {
    pub use crate::{
        CausticError,
        PhaseSpaceRepr,
        PoissonSolver,
        Advector,
        TimeIntegrator,
        Domain,
        DomainBuilder,
        ExitReason,
        Simulation,
    };
    pub use crate::tooling::core::{
        types::*,
        diagnostics::{Diagnostics, GlobalDiagnostics},
        io::IOManager,
        conditions::ExitCondition,
        init::{
            isolated::{PlummerIC, KingIC, HernquistIC, NfwIC, IsolatedEquilibrium},
            cosmological::ZeldovichIC,
            mergers::MergerIC,
        },
        output::exit::package::ExitPackage,
        integrator::SimState,
        init::input::optional::OptionalParams,
    };
}

