//! Top-level `Simulation` struct. Entry point for library users; matches the README builder API.

use crate::tooling::core::{
    advecator::Advector,
    conditions::{ExitReason, TimeLimitCondition},
    diagnostics::{Diagnostics, GlobalDiagnostics},
    init::{domain::Domain, input::optional::OptionalParams},
    integrator::TimeIntegrator,
    io::{IOManager, OutputFormat},
    output::exit::{package::ExitPackage, standard::ExitEvaluator},
    phasespace::PhaseSpaceRepr,
    solver::PoissonSolver,
    types::*,
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
    pub g: f64,
    pub time: f64,
    pub step: u64,
    pub start_time: std::time::Instant,
}

impl Simulation {
    /// Create a `SimulationBuilder` with no components set.
    pub fn builder() -> SimulationBuilder {
        SimulationBuilder::new()
    }

    /// Run the simulation to completion and return the exit package.
    pub fn run(&mut self) -> anyhow::Result<ExitPackage> {
        loop {
            if let Some(reason) = self.step()? {
                let snapshot = self.repr.to_snapshot(self.time);
                let history = self.diagnostics.history.clone();
                let wall_secs = self.start_time.elapsed().as_secs_f64();
                return Ok(ExitPackage::assemble(
                    snapshot,
                    history,
                    reason,
                    String::new(),
                    wall_secs,
                    self.step,
                    0,
                ));
            }
        }
    }

    /// Advance by a single timestep. Returns `Some(reason)` if the simulation should stop.
    pub fn step(&mut self) -> anyhow::Result<Option<ExitReason>> {
        let cfl_factor = self.opts.cfl_factor;
        let mut dt = self.integrator.max_dt(&*self.repr, cfl_factor);

        // Clamp dt to not overshoot t_final by more than necessary
        let t_final = {
            use rust_decimal::prelude::ToPrimitive;
            self.domain.time_range.t_final.to_f64().unwrap_or(f64::MAX)
        };
        if self.time + dt > t_final {
            dt = (t_final - self.time).max(0.0);
        }

        self.integrator
            .advance(&mut *self.repr, &*self.poisson, &*self.advector, dt);

        self.time += dt;
        self.step += 1;

        let density = self.repr.compute_density();
        let potential = self.poisson.solve(&density, self.g);
        let dx = self.domain.dx();
        let dx3 = dx[0] * dx[1] * dx[2];
        let diag = self
            .diagnostics
            .compute(&*self.repr, &potential, self.time, dx3);

        Ok(self.exit_evaluator.check(&diag))
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
    g: Option<f64>,
}

impl SimulationBuilder {
    pub fn new() -> Self {
        Self {
            domain: None,
            repr: None,
            poisson: None,
            advector: None,
            integrator: None,
            opts: None,
            ic: None,
            t_final: None,
            output_interval: None,
            energy_tolerance: None,
            g: None,
        }
    }

    pub fn domain(mut self, d: Domain) -> Self {
        self.domain = Some(d);
        self
    }

    pub fn representation(mut self, r: impl PhaseSpaceRepr + 'static) -> Self {
        self.repr = Some(Box::new(r));
        self
    }

    pub fn poisson_solver(mut self, p: impl PoissonSolver + 'static) -> Self {
        self.poisson = Some(Box::new(p));
        self
    }

    pub fn advector(mut self, a: impl Advector + 'static) -> Self {
        self.advector = Some(Box::new(a));
        self
    }

    pub fn integrator(mut self, i: impl TimeIntegrator + 'static) -> Self {
        self.integrator = Some(Box::new(i));
        self
    }

    pub fn poisson_solver_boxed(mut self, p: Box<dyn PoissonSolver>) -> Self {
        self.poisson = Some(p);
        self
    }

    pub fn integrator_boxed(mut self, i: Box<dyn TimeIntegrator>) -> Self {
        self.integrator = Some(i);
        self
    }

    pub fn initial_conditions(mut self, ic: PhaseSpaceSnapshot) -> Self {
        self.ic = Some(ic);
        self
    }

    pub fn time_final(mut self, t: f64) -> Self {
        self.t_final = Some(t);
        self
    }

    pub fn output_interval(mut self, dt: f64) -> Self {
        self.output_interval = Some(dt);
        self
    }

    pub fn exit_on_energy_drift(mut self, tol: f64) -> Self {
        self.energy_tolerance = Some(tol);
        self
    }

    pub fn gravitational_constant(mut self, g: f64) -> Self {
        self.g = Some(g);
        self
    }

    pub fn cfl_factor(mut self, cfl: f64) -> Self {
        let mut opts = self.opts.unwrap_or_default();
        opts.cfl_factor = cfl;
        self.opts = Some(opts);
        self
    }

    /// Validate all required fields are set, initialise repr from IC snapshot.
    pub fn build(self) -> anyhow::Result<Simulation> {
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use rust_decimal::prelude::ToPrimitive;

        let domain = self
            .domain
            .ok_or_else(|| anyhow::anyhow!("domain not set"))?;
        let poisson = self
            .poisson
            .ok_or_else(|| anyhow::anyhow!("poisson_solver not set"))?;
        let advector = self
            .advector
            .ok_or_else(|| anyhow::anyhow!("advector not set"))?;
        let integrator = self
            .integrator
            .ok_or_else(|| anyhow::anyhow!("integrator not set"))?;

        // t_final: prefer explicit t_final, else read from domain
        let t_final = self
            .t_final
            .unwrap_or_else(|| domain.time_range.t_final.to_f64().unwrap_or(1.0));

        let g = self.g.unwrap_or(1.0);

        // Build the phase-space representation
        let repr: Box<dyn PhaseSpaceRepr> = if let Some(ic) = self.ic {
            Box::new(UniformGrid6D::from_snapshot(ic, domain.clone()))
        } else if let Some(r) = self.repr {
            r
        } else {
            anyhow::bail!("either initial_conditions or representation must be set");
        };

        // Compute initial diagnostics
        let dx = domain.dx();
        let dx3 = dx[0] * dx[1] * dx[2];
        let density = repr.compute_density();
        let potential = poisson.solve(&density, g);

        let mut diagnostics = Diagnostics {
            history: Vec::new(),
        };
        let initial_diag = diagnostics.compute(&*repr, &potential, 0.0, dx3);

        // Build exit evaluator with time limit as default condition
        let mut exit_evaluator = ExitEvaluator::new(initial_diag);
        exit_evaluator.add_condition(Box::new(TimeLimitCondition { t_final }));

        if let Some(tol) = self.energy_tolerance {
            use crate::tooling::core::conditions::EnergyDriftCondition;
            exit_evaluator.add_condition(Box::new(EnergyDriftCondition { tolerance: tol }));
        }

        Ok(Simulation {
            domain,
            repr,
            poisson,
            advector,
            integrator,
            diagnostics,
            io: IOManager::new("output", OutputFormat::Binary),
            exit_evaluator,
            opts: self.opts.unwrap_or_default(),
            g,
            time: 0.0,
            step: 0,
            start_time: std::time::Instant::now(),
        })
    }
}
