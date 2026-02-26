//! Top-level `Simulation` struct. Entry point for library users; matches the README builder API.

use crate::tooling::core::{
    types::*,
    phasespace::PhaseSpaceRepr,
    solver::PoissonSolver,
    advecator::Advector,
    integrator::TimeIntegrator,
    diagnostics::Diagnostics,
    io::IOManager,
    conditions::ExitReason,
    output::exit::{standard::ExitEvaluator, package::ExitPackage},
    init::{domain::Domain, input::optional::OptionalParams},
};

/// The top-level simulation object. Owns all solver components.
pub struct Simulation {
    pub domain: Domain,
    pub repr: Box<dyn PhaseSpaceRepr>,
    pub poisson: Box<dyn PoissonSolver>,
    pub advector: Box<dyn Advector>,
    pub integrator: Box<dyn TimeIntegrator>,
    pub diagnostics: Diagnostics,
    pub io: IOManager,
    pub exit_evaluator: ExitEvaluator,
    pub opts: OptionalParams,
    pub time: f64,
    pub step: u64,
}

impl Simulation {
    /// Create a `SimulationBuilder` with no components set.
    pub fn builder() -> SimulationBuilder {
        todo!()
    }

    /// Run the simulation to completion and return the exit package.
    ///
    /// Main loop:
    /// 1. Advance integrator by Δt
    /// 2. Compute diagnostics
    /// 3. Check exit conditions
    /// 4. Save snapshots / checkpoints at configured intervals
    /// 5. Return `ExitPackage` when done
    pub fn run(&mut self) -> anyhow::Result<ExitPackage> {
        todo!("main loop: advance, diagnose, check exit, save, return ExitPackage")
    }

    /// Advance by a single timestep. Returns `Some(reason)` if the simulation should stop.
    pub fn step(&mut self) -> anyhow::Result<Option<ExitReason>> {
        todo!("single timestep; returns Some(reason) if done")
    }

    /// Current simulation time.
    pub fn current_time(&self) -> f64 {
        self.time
    }
}

/// Builder for `Simulation` using a fluent API.
pub struct SimulationBuilder {
    domain: Option<Domain>,
    repr: Option<Box<dyn PhaseSpaceRepr>>,
    poisson: Option<Box<dyn PoissonSolver>>,
    advector: Option<Box<dyn Advector>>,
    integrator: Option<Box<dyn TimeIntegrator>>,
    opts: Option<OptionalParams>,
    ic: Option<PhaseSpaceSnapshot>,
    t_final: Option<f64>,
    output_interval: Option<f64>,
    energy_tolerance: Option<f64>,
}

impl SimulationBuilder {
    pub fn new() -> Self {
        todo!()
    }

    pub fn domain(mut self, d: Domain) -> Self {
        todo!()
    }

    pub fn representation(mut self, r: impl PhaseSpaceRepr + 'static) -> Self {
        todo!()
    }

    pub fn poisson_solver(mut self, p: impl PoissonSolver + 'static) -> Self {
        todo!()
    }

    pub fn advector(mut self, a: impl Advector + 'static) -> Self {
        todo!()
    }

    pub fn integrator(mut self, i: impl TimeIntegrator + 'static) -> Self {
        todo!()
    }

    pub fn initial_conditions(mut self, ic: PhaseSpaceSnapshot) -> Self {
        todo!()
    }

    pub fn time_final(mut self, t: f64) -> Self {
        todo!()
    }

    pub fn output_interval(mut self, dt: f64) -> Self {
        todo!()
    }

    pub fn exit_on_energy_drift(mut self, tol: f64) -> Self {
        todo!()
    }

    /// Validate all required fields are set, initialise repr from IC snapshot.
    pub fn build(self) -> anyhow::Result<Simulation> {
        todo!("validate all required fields are set, init repr from IC snapshot")
    }
}
