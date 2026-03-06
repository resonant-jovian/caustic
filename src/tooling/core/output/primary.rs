//! Primary snapshot outputs: f(x,v,t), Φ(x,t), ρ(x,t), g(x,t).

use super::super::{types::*, phasespace::PhaseSpaceRepr};

/// All primary fields at one simulation time.
pub struct PrimarySnapshot {
    pub distribution: PhaseSpaceSnapshot,
    pub potential: PotentialField,
    pub density: DensityField,
    pub acceleration: AccelerationField,
    pub time: f64,
}

impl PrimarySnapshot {
    /// Collect all primary fields into a snapshot struct.
    pub fn capture(repr: &dyn PhaseSpaceRepr, potential: &PotentialField, time: f64) -> Self {
        todo!("collect all fields into snapshot struct")
    }
}
