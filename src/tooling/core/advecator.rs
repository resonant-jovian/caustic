//! The `Advector` trait for the 6D transport step.
//! Advances f by one sub-step using the given force field.

use super::types::AccelerationField;
use super::phasespace::PhaseSpaceRepr;

/// Trait for all phase-space advection schemes.
pub trait Advector {
    /// Advance f in phase space by Δt using the given acceleration.
    /// Returns nothing — mutates the representation in place.
    fn step(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64);

    /// Spatial-only drift step (free streaming): advance f by v·Δt in position at constant v.
    fn drift(&self, repr: &mut dyn PhaseSpaceRepr, dt: f64);

    /// Velocity-only kick step: advance f by g·Δt in velocity at constant x.
    fn kick(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64);
}
