//! Shared helper functions and macros for time integrators.
//!
//! Eliminates duplication of the dynamical-time CFL estimate, the
//! density→potential→acceleration Poisson pipeline, timing instrumentation,
//! progress reporting, and SpectralV hypercollision dispatch.

use super::super::{
    context::SimContext,
    events::{EventEmitter, SimEvent},
    phasespace::PhaseSpaceRepr,
    types::{AccelerationField, DensityField, PotentialField},
};

/// Compute the maximum stable timestep from the dynamical time.
///
/// dt = cfl_factor / sqrt(G * rho_max)
///
/// Returns a large sentinel (1e10) if the density or G is non-positive.
/// Used by all splitting integrators (Strang, Yoshida, Lie, BM4, RKN6, etc.).
/// The unsplit integrator uses a different CFL based on spatial/velocity grids.
pub fn dynamical_timestep(repr: &dyn PhaseSpaceRepr, g: f64, cfl_factor: f64) -> f64 {
    let density = repr.compute_density();
    let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);
    if rho_max <= 0.0 || g <= 0.0 {
        return 1e10;
    }
    let t_dyn = 1.0 / (g * rho_max).sqrt();
    cfl_factor * t_dyn
}

/// Solve the full Poisson pipeline: density → potential → acceleration.
///
/// This is the most common 3-line sequence in time integrators. Returns all
/// three fields so the caller can use them for diagnostics and conservation.
pub fn solve_poisson(
    repr: &dyn PhaseSpaceRepr,
    ctx: &SimContext,
) -> (DensityField, PotentialField, AccelerationField) {
    let density = repr.compute_density();
    let potential = ctx.solver.solve(&density, ctx);
    let acceleration = ctx.solver.compute_acceleration(&potential);
    (density, potential, acceleration)
}

/// Apply SpectralV hypercollision damping if the representation is `SpectralV`.
///
/// This is a no-op for all other representation types. Used after each kick
/// sub-step in Strang, Yoshida, and Lie splitting to suppress Gibbs ringing.
pub fn apply_hypercollision_if_spectral(
    repr: &mut dyn PhaseSpaceRepr,
    dt: f64,
    emitter: &EventEmitter,
) {
    if let Some(spectral) = repr
        .as_any_mut()
        .downcast_mut::<super::super::algos::spectral::SpectralV>()
    {
        let max_before = spectral
            .coefficients
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        spectral.apply_hypercollision(dt);
        let max_after = spectral
            .coefficients
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        emitter.emit(SimEvent::SpectralHypercollisionApplied {
            damping_coefficient: spectral.hypercollision_nu,
            order: spectral.hypercollision_order,
            max_mode_dampening: if max_before > 0.0 {
                1.0 - max_after / max_before
            } else {
                0.0
            },
        });
    }
}

/// Time a block and accumulate milliseconds into a `StepTimings` field.
///
/// Usage: `time_ms!(timings, drift_ms, { advector.drift(repr, dt) });`
macro_rules! time_ms {
    ($timings:expr, $field:ident, $body:expr) => {{
        let _t0 = std::time::Instant::now();
        let _result = $body;
        $timings.$field += _t0.elapsed().as_secs_f64() * 1000.0;
        _result
    }};
}

/// Report integrator phase and sub-step progress to the TUI.
///
/// Usage:
/// `report_phase!(ctx, StepPhase::DriftHalf1, 0, 5);`
macro_rules! report_phase {
    ($ctx:expr, $phase:expr, $step:expr, $total:expr) => {
        $ctx.progress.set_phase($phase);
        $ctx.progress.set_sub_step($step, $total);
    };
}

#[allow(unused_imports)]
pub(crate) use report_phase;
#[allow(unused_imports)]
pub(crate) use time_ms;
