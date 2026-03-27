//! The `Advector` trait for the 6D transport step.
//! Advances f by one sub-step using the given force field.

use super::context::SimContext;
use super::phasespace::PhaseSpaceRepr;
use super::types::AccelerationField;

/// Trait for all phase-space advection schemes.
///
/// Uses `ctx.dt` for the sub-step duration. Integrators pass sub-contexts with
/// the appropriate dt (e.g. `ctx.with_dt(dt / 2.0)` for Strang drift half-steps).
pub trait Advector {
    /// Advance f in phase space by `ctx.dt` using the given acceleration.
    /// Mutates the representation in place.
    fn step(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, ctx: &SimContext);

    /// Spatial-only drift step (free streaming): advance f by v * ctx.dt in position.
    fn drift(&self, repr: &mut dyn PhaseSpaceRepr, ctx: &SimContext);

    /// Velocity-only kick step: advance f by g * ctx.dt in velocity.
    fn kick(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, ctx: &SimContext);
}
