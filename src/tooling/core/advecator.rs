//! The `Advector` trait for the 6D transport step.
//! Advances f by one sub-step using the given force field.

use super::phasespace::PhaseSpaceRepr;
use super::types::AccelerationField;

/// Trait for all phase-space advection schemes.
///
/// # Examples
///
/// ```no_run
/// use caustic::{Advector, SemiLagrangian, PhaseSpaceRepr};
///
/// let advector = SemiLagrangian::new();
/// // In a typical splitting step:
/// // advector.drift(repr, dt);       // spatial half-step
/// // advector.kick(repr, &accel, dt); // velocity full-step
/// // advector.drift(repr, dt);       // spatial half-step
/// ```
pub trait Advector {
    /// Advance f in phase space by Δt using the given acceleration.
    /// Returns nothing — mutates the representation in place.
    fn step(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64);

    /// Spatial-only drift step (free streaming): advance f by v·Δt in position at constant v.
    fn drift(&self, repr: &mut dyn PhaseSpaceRepr, dt: f64);

    /// Velocity-only kick step: advance f by g·Δt in velocity at constant x.
    fn kick(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64);
}
