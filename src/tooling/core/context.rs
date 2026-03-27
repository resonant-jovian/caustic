//! Simulation execution context — the universal parameter bundle for all trait methods.
//!
//! `SimContext` replaces the separate `solver`, `advector`, `dt`, `g` parameters
//! that were previously passed individually to trait methods. It also carries the
//! `EventEmitter` for structured observability and `StepProgress` for real-time
//! TUI phase display.
//!
//! Constructed per-step by `Simulation::step()`. Time integrators create
//! sub-contexts with modified `dt` for splitting sub-steps via [`SimContext::with_dt()`].

use super::advecator::Advector;
use super::events::EventEmitter;
use super::progress::StepProgress;
use super::solver::PoissonSolver;

/// Simulation execution context passed to all trait methods.
///
/// Bundles the Poisson solver, advector, event emitter, progress state, and
/// per-step metadata into a single struct. This avoids growing parameter lists
/// and makes the API extensible — new fields can be added without changing
/// trait signatures.
///
/// # Sub-step contexts
///
/// Splitting methods (Strang, Yoshida, etc.) create sub-contexts with modified
/// `dt` for each sub-step:
///
/// ```ignore
/// // In a Strang integrator:
/// ctx.advector.drift(repr, &ctx.with_dt(ctx.dt / 2.0));
/// // ... Poisson solve ...
/// ctx.advector.kick(repr, &accel, &ctx.with_dt(ctx.dt));
/// ctx.advector.drift(repr, &ctx.with_dt(ctx.dt / 2.0));
/// ```
pub struct SimContext<'a> {
    /// Poisson solver reference (used by integrators to orchestrate solves).
    pub solver: &'a dyn PoissonSolver,
    /// Advector reference (used by integrators to dispatch drift/kick).
    pub advector: &'a dyn Advector,
    /// Event emitter for structured observability.
    pub emitter: &'a EventEmitter,
    /// Lock-free atomic progress for real-time TUI phase display.
    pub progress: &'a StepProgress,
    /// Current simulation step number.
    pub step: u64,
    /// Current simulation time.
    pub time: f64,
    /// Timestep for this sub-step. Splitting methods modify this via `with_dt()`.
    pub dt: f64,
    /// Gravitational constant G.
    pub g: f64,
}

impl<'a> SimContext<'a> {
    /// Create a sub-context with a different timestep.
    ///
    /// Used by splitting methods for sub-step durations (e.g. `dt/2` for Strang drift).
    /// All other fields (solver, advector, emitter, progress, step, time, g) are shared.
    pub fn with_dt(&self, dt: f64) -> SimContext<'a> {
        SimContext { dt, ..*self }
    }

    /// Create a sub-context with a different gravitational constant.
    ///
    /// Used by cosmological integrators where `g_eff = g * a^2` varies with scale factor.
    pub fn with_g(&self, g: f64) -> SimContext<'a> {
        SimContext { g, ..*self }
    }
}
