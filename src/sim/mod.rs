//! Top-level `Simulation` struct. Entry point for library users; matches the README builder API.

use std::sync::Arc;

use rayon::prelude::*;
use rust_decimal::Decimal;

use crate::tooling::core::{
    advecator::Advector,
    conditions::{ExitReason, TimeLimitCondition},
    conservation::lomac::LoMaC,
    diagnostics::{Diagnostics, GlobalDiagnostics},
    init::{domain::Domain, input::optional::OptionalParams},
    integrator::{StepTimings, TimeIntegrator},
    io::{IOManager, OutputFormat},
    output::exit::{package::ExitPackage, standard::ExitEvaluator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};

/// Convert f64 to Decimal with a warning on unrepresentable values (NaN, Inf, subnormal).
fn f64_to_decimal(v: f64, field: &str) -> Decimal {
    Decimal::from_f64_retain(v).unwrap_or_else(|| {
        tracing::warn!(
            "{field}({v}) is not representable as Decimal (NaN/Inf/subnormal); defaulting to 0. \
             Use the _decimal() variant for exact values."
        );
        Decimal::ZERO
    })
}

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
    /// Optional LoMaC conservation framework. When active, projects the
    /// distribution function after each time step to restore exact local
    /// conservation of mass, momentum, and energy.
    pub lomac: Option<LoMaC>,
    pub g: f64,
    pub time: f64,
    pub step: u64,
    pub start_time: std::time::Instant,
    /// Cached ρ_max from the previous step's density computation.
    /// Used to compute Δt without a redundant `compute_density()` call.
    cached_rho_max: Option<f64>,
    /// Per-step phase timing breakdown from the most recent `step()` call.
    /// Includes integrator sub-step timings + post-advance diagnostics timing.
    pub last_step_timings: StepTimings,
    /// Cached density from the most recent step, for TUI reuse.
    pub cached_density: Option<DensityField>,
    /// Cached potential from the most recent step, for TUI reuse.
    pub cached_potential: Option<PotentialField>,
    /// Optional shared progress state for intra-step TUI visibility.
    progress: Option<Arc<StepProgress>>,
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
                let snapshot = self.repr.to_snapshot(self.time).unwrap_or_else(|| {
                    PhaseSpaceSnapshot {
                        data: vec![],
                        shape: [0; 6],
                        time: self.time,
                    }
                });
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
        let cfl_factor = self.opts.cfl_factor_f64();

        // Use cached ρ_max from previous step to avoid redundant compute_density().
        // Falls back to the integrator's max_dt() on the first step.
        let mut dt = if let Some(rho_max) = self.cached_rho_max.take() {
            if rho_max <= 0.0 || self.g <= 0.0 {
                1e10
            } else {
                let t_dyn = 1.0 / (self.g * rho_max).sqrt();
                cfl_factor * t_dyn
            }
        } else {
            self.integrator.max_dt(&*self.repr, cfl_factor)
        };

        // Use adaptive integrator's suggestion if available (CFL as hard upper bound)
        if let Some(adaptive_dt) = self.integrator.suggested_dt() {
            dt = dt.min(adaptive_dt);
        }

        // Clamp dt to not overshoot t_final by more than necessary
        let t_final = {
            use rust_decimal::prelude::ToPrimitive;
            self.domain.time_range.t_final.to_f64().unwrap_or(f64::MAX)
        };
        if self.time + dt > t_final {
            dt = (t_final - self.time).max(0.0);
        }

        let products =
            self.integrator
                .advance(&mut *self.repr, &*self.poisson, &*self.advector, dt)?;

        // Capture integrator sub-step timings (drift, poisson, kick)
        let mut timings = self
            .integrator
            .last_step_timings()
            .cloned()
            .unwrap_or_default();

        // Extend sub_step tracking to include post-advance phases
        if let Some(ref p) = self.progress {
            let snap = p.read();
            let post_count: u8 = 2 + u8::from(self.lomac.is_some());
            p.set_sub_step(snap.sub_step, snap.sub_step_total + post_count);
        }

        // LoMaC conservation projection: advance macroscopic state in sync
        // with kinetic, then project f to restore exact moments.
        if self.lomac.is_some()
            && let Some(ref p) = self.progress
        {
            let sub = p.read().sub_step;
            p.set_phase(StepPhase::LoMaC);
            p.set_sub_step(sub + 1, p.read().sub_step_total);
        }
        if let Some(ref mut lomac) = self.lomac {
            let t0 = std::time::Instant::now();

            // Reuse end-of-step acceleration from integrator (avoids redundant
            // compute_density + Poisson solve + compute_acceleration).
            let gx = &products.acceleration.gx;
            let gy = &products.acceleration.gy;
            let gz = &products.acceleration.gz;

            // HtTensor path: use project_ht to avoid unnecessary dense→HT→dense round-trips
            if let Some(ht) = self
                .repr
                .as_any()
                .downcast_ref::<crate::tooling::core::algos::ht::HtTensor>()
            {
                if ht.can_materialize() {
                    lomac.advance_macroscopic(dt, gx, gy, gz);
                    let corrected = lomac.project_ht(ht);
                    let shape = ht.shape;
                    let corrected_snap = PhaseSpaceSnapshot {
                        data: corrected,
                        shape,
                        time: self.time,
                    };
                    self.repr = Box::new(
                        crate::tooling::core::algos::uniform::UniformGrid6D::from_snapshot(
                            corrected_snap,
                            self.domain.clone(),
                        ),
                    );
                    if let Some(ref p) = self.progress {
                        self.repr.set_progress(p.clone());
                    }
                } else {
                    // Skip dense LoMaC projection for large HT tensors that cannot
                    // be materialized. Advance macroscopic state for diagnostics only.
                    tracing::warn!(
                        "LoMaC projection skipped: HT tensor too large to materialize ({} elements)",
                        ht.shape.iter().product::<usize>()
                    );
                    lomac.advance_macroscopic(dt, gx, gy, gz);
                }
            } else {
                // Dense path (UniformGrid6D, etc.)
                let snapshot = self.repr.to_snapshot(self.time).ok_or_else(|| {
                    anyhow::anyhow!("LoMaC dense path requires to_snapshot support")
                })?;
                let corrected = lomac.apply(dt, gx, gy, gz, &snapshot.data);
                let corrected_snap = PhaseSpaceSnapshot {
                    data: corrected,
                    shape: snapshot.shape,
                    time: self.time,
                };
                self.repr = Box::new(
                    crate::tooling::core::algos::uniform::UniformGrid6D::from_snapshot(
                        corrected_snap,
                        self.domain.clone(),
                    ),
                );
                if let Some(ref p) = self.progress {
                    self.repr.set_progress(p.clone());
                }
            }
            timings.other_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        self.time += dt;
        self.step += 1;

        // Post-advance density computation (for diagnostics + caching)
        if let Some(ref p) = self.progress {
            let sub = p.read().sub_step;
            p.set_phase(StepPhase::PostDensity);
            p.set_sub_step(sub + 1, p.read().sub_step_total);
        }
        let t0 = std::time::Instant::now();
        // When LoMaC is active, the post-projection density equals the KFVS
        // target density by construction — need a fresh Poisson solve for it.
        // Without LoMaC, reuse the end-of-step products from the integrator
        // (avoids redundant compute_density + Poisson solve).
        let (density, potential) = if let Some(ref lomac) = self.lomac {
            let density = DensityField {
                data: lomac.kfvs.state.iter().map(|m| m.density).collect(),
                shape: lomac.spatial_shape,
            };
            let potential = self.poisson.solve(&density, self.g);
            (density, potential)
        } else {
            (products.density, products.potential)
        };
        timings.density_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let dx = self.domain.dx();
        let dx3 = dx[0] * dx[1] * dx[2];

        // Cache ρ_max for next step's dt computation (avoids redundant compute_density)
        self.cached_rho_max = Some(
            density
                .data
                .par_iter()
                .cloned()
                .reduce(|| 0.0_f64, f64::max),
        );

        if let Some(ref p) = self.progress {
            let sub = p.read().sub_step;
            p.set_phase(StepPhase::Diagnostics);
            p.set_sub_step(sub + 1, p.read().sub_step_total);
        }
        let t0 = std::time::Instant::now();
        let diag = self.diagnostics.compute_with_density(
            &*self.repr,
            &density,
            &potential,
            self.time,
            dx3,
        );
        timings.diagnostics_ms += t0.elapsed().as_secs_f64() * 1000.0;

        self.last_step_timings = timings;

        // Cache density and potential for TUI reuse (avoids redundant recomputation)
        self.cached_density = Some(density);
        self.cached_potential = Some(potential);

        if let Some(ref p) = self.progress {
            p.set_phase(StepPhase::StepComplete);
        }

        Ok(self.exit_evaluator.check(&diag))
    }

    /// Attach shared progress state for intra-step TUI visibility.
    /// Propagates to the integrator so sub-step phases are reported.
    pub fn set_progress(&mut self, p: Arc<StepProgress>) {
        self.integrator.set_progress(p.clone());
        self.repr.set_progress(p.clone());
        self.poisson.set_progress(p.clone());
        if let Some(ref mut lomac) = self.lomac {
            lomac.set_progress(p.clone());
        }
        self.progress = Some(p);
    }

    /// Current simulation time.
    pub fn current_time(&self) -> f64 {
        self.time
    }
}

/// Builder for [`Simulation`] using a fluent API.
///
/// All four core components ([`Domain`], [`PhaseSpaceRepr`], [`PoissonSolver`],
/// [`Advector`], [`TimeIntegrator`]) and initial conditions must be set before
/// calling [`build()`](SimulationBuilder::build). Numeric config fields are
/// stored as `Decimal` for exact arithmetic; each has an `f64` setter and a
/// `_decimal` variant.
///
/// # Examples
///
/// ```no_run
/// use caustic::prelude::*;
/// use caustic::{
///     FftPoisson, SemiLagrangian, StrangSplitting,
///     Domain, DomainBuilder, SpatialBoundType, VelocityBoundType,
///     Simulation, sample_on_grid,
/// };
///
/// let domain = Domain::builder()
///     .spatial_extent(10.0)
///     .velocity_extent(5.0)
///     .spatial_resolution(16)
///     .velocity_resolution(16)
///     .spatial_bc(SpatialBoundType::Periodic)
///     .velocity_bc(VelocityBoundType::Open)
///     .build()
///     .unwrap();
///
/// let snap = sample_on_grid(&PlummerIC::new(1.0, 1.0, 1.0), &domain);
///
/// let mut sim = Simulation::builder()
///     .domain(domain.clone())
///     .poisson_solver(FftPoisson::new(&domain))
///     .advector(SemiLagrangian::new())
///     .integrator(StrangSplitting::new(1.0))
///     .initial_conditions(snap)
///     .time_final(10.0)
///     .gravitational_constant(1.0)
///     .cfl_factor(0.5)
///     .build()
///     .unwrap();
/// ```
pub struct SimulationBuilder {
    domain: Option<Domain>,
    repr: Option<Box<dyn PhaseSpaceRepr>>,
    poisson: Option<Box<dyn PoissonSolver>>,
    advector: Option<Box<dyn Advector>>,
    integrator: Option<Box<dyn TimeIntegrator>>,
    opts: Option<OptionalParams>,
    ic: Option<PhaseSpaceSnapshot>,
    t_final: Option<Decimal>,
    output_interval: Option<Decimal>,
    energy_tolerance: Option<Decimal>,
    g: Option<Decimal>,
    enable_lomac: bool,
}

impl SimulationBuilder {
    /// Create an empty builder with no components set.
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
            enable_lomac: false,
        }
    }

    /// Set the computational domain (spatial/velocity extents, resolution, BCs).
    pub fn domain(mut self, d: Domain) -> Self {
        self.domain = Some(d);
        self
    }

    /// Set the phase-space representation (concrete type, auto-boxed).
    pub fn representation(mut self, r: impl PhaseSpaceRepr + 'static) -> Self {
        self.repr = Some(Box::new(r));
        self
    }

    /// Set the phase-space representation (pre-boxed, for dynamic dispatch).
    pub fn representation_boxed(mut self, r: Box<dyn PhaseSpaceRepr>) -> Self {
        self.repr = Some(r);
        self
    }

    /// Set the Poisson solver (concrete type, auto-boxed).
    pub fn poisson_solver(mut self, p: impl PoissonSolver + 'static) -> Self {
        self.poisson = Some(Box::new(p));
        self
    }

    /// Set the advection scheme (concrete type, auto-boxed).
    pub fn advector(mut self, a: impl Advector + 'static) -> Self {
        self.advector = Some(Box::new(a));
        self
    }

    /// Set the time integrator / operator splitting (concrete type, auto-boxed).
    pub fn integrator(mut self, i: impl TimeIntegrator + 'static) -> Self {
        self.integrator = Some(Box::new(i));
        self
    }

    /// Set the Poisson solver (pre-boxed, for dynamic dispatch).
    pub fn poisson_solver_boxed(mut self, p: Box<dyn PoissonSolver>) -> Self {
        self.poisson = Some(p);
        self
    }

    /// Set the time integrator (pre-boxed, for dynamic dispatch).
    pub fn integrator_boxed(mut self, i: Box<dyn TimeIntegrator>) -> Self {
        self.integrator = Some(i);
        self
    }

    /// Set the initial condition snapshot. Use [`sample_on_grid`] to generate from an IC type.
    pub fn initial_conditions(mut self, ic: PhaseSpaceSnapshot) -> Self {
        self.ic = Some(ic);
        self
    }

    /// Set final simulation time. See also [`time_final_decimal`](Self::time_final_decimal).
    pub fn time_final(mut self, t: f64) -> Self {
        self.t_final = Some(f64_to_decimal(t, "time_final"));
        self
    }

    /// Set final simulation time from an exact `Decimal` value.
    pub fn time_final_decimal(mut self, t: Decimal) -> Self {
        self.t_final = Some(t);
        self
    }

    /// Set snapshot output interval. See also [`output_interval_decimal`](Self::output_interval_decimal).
    pub fn output_interval(mut self, dt: f64) -> Self {
        self.output_interval = Some(f64_to_decimal(dt, "output_interval"));
        self
    }

    /// Set output interval from an exact `Decimal` value.
    pub fn output_interval_decimal(mut self, dt: Decimal) -> Self {
        self.output_interval = Some(dt);
        self
    }

    /// Add an energy drift exit condition with the given relative tolerance.
    pub fn exit_on_energy_drift(mut self, tol: f64) -> Self {
        self.energy_tolerance = Some(f64_to_decimal(tol, "exit_on_energy_drift"));
        self
    }

    /// Set energy drift tolerance from an exact `Decimal` value.
    pub fn exit_on_energy_drift_decimal(mut self, tol: Decimal) -> Self {
        self.energy_tolerance = Some(tol);
        self
    }

    /// Set Newton's gravitational constant G. Default is 1.0.
    pub fn gravitational_constant(mut self, g: f64) -> Self {
        self.g = Some(f64_to_decimal(g, "gravitational_constant"));
        self
    }

    /// Set gravitational constant from an exact `Decimal` value.
    pub fn gravitational_constant_decimal(mut self, g: Decimal) -> Self {
        self.g = Some(g);
        self
    }

    /// Enable LoMaC conservation framework. After each time step, projects
    /// the distribution function to restore exact local conservation of
    /// mass, momentum, and energy.
    pub fn lomac(mut self, enable: bool) -> Self {
        self.enable_lomac = enable;
        self
    }

    /// Set the CFL safety factor for adaptive timestep control. Default is from domain.
    pub fn cfl_factor(mut self, cfl: f64) -> Self {
        let mut opts = self.opts.unwrap_or_default();
        opts.cfl_factor = f64_to_decimal(cfl, "cfl_factor");
        self.opts = Some(opts);
        self
    }

    /// Set CFL safety factor from an exact `Decimal` value.
    pub fn cfl_factor_decimal(mut self, cfl: Decimal) -> Self {
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

        // t_final: prefer explicit t_final (Decimal), else read from domain
        let t_final_dec = self.t_final.unwrap_or(domain.time_range.t_final);
        let t_final = t_final_dec.to_f64().unwrap_or(1.0);

        let g = self.g.unwrap_or(Decimal::ONE).to_f64().unwrap_or(1.0);

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

        if let Some(tol_dec) = self.energy_tolerance {
            use crate::tooling::core::conditions::EnergyDriftCondition;
            let tol = tol_dec.to_f64().unwrap_or(1e-6);
            exit_evaluator.add_condition(Box::new(EnergyDriftCondition { tolerance: tol }));
        }

        // Initialize LoMaC if requested
        let lomac = if self.enable_lomac {
            let dv = domain.dv();
            let lv = domain.lv();
            let v_min = [-lv[0], -lv[1], -lv[2]];
            let spatial_shape = [
                domain.spatial_res.x1 as usize,
                domain.spatial_res.x2 as usize,
                domain.spatial_res.x3 as usize,
            ];
            let velocity_shape = [
                domain.velocity_res.v1 as usize,
                domain.velocity_res.v2 as usize,
                domain.velocity_res.v3 as usize,
            ];
            let mut lom = LoMaC::new(spatial_shape, velocity_shape, dx, dv, v_min);

            // Initialize from the IC distribution — use HT-native path if available
            if let Some(ht) = repr
                .as_any()
                .downcast_ref::<crate::tooling::core::algos::ht::HtTensor>()
            {
                lom.initialize_from_ht(ht);
            } else {
                let snapshot = repr.to_snapshot(0.0).ok_or_else(|| {
                    anyhow::anyhow!("LoMaC initialization requires to_snapshot support")
                })?;
                lom.initialize_from_kinetic(&snapshot.data);
            }
            Some(lom)
        } else {
            None
        };

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
            lomac,
            g,
            time: 0.0,
            step: 0,
            start_time: std::time::Instant::now(),
            cached_rho_max: None,
            last_step_timings: StepTimings::default(),
            cached_density: None,
            cached_potential: None,
            progress: None,
        })
    }
}
