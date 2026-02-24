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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
