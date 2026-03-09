//! Primary snapshot outputs: f(x,v,t), Φ(x,t), ρ(x,t), g(x,t).

use super::super::{phasespace::PhaseSpaceRepr, types::*};

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
    ///
    /// Computes density from repr and clones potential. Acceleration is left
    /// empty (no solver available in this signature — caller should populate
    /// separately if needed).
    pub fn capture(repr: &dyn PhaseSpaceRepr, potential: &PotentialField, time: f64) -> Self {
        let density = repr.compute_density();
        let n = density.data.len();
        let shape = density.shape;
        let distribution = repr.to_snapshot(time);

        Self {
            distribution,
            potential: PotentialField {
                data: potential.data.clone(),
                shape: potential.shape,
            },
            density,
            acceleration: AccelerationField {
                gx: vec![0.0; n],
                gy: vec![0.0; n],
                gz: vec![0.0; n],
                shape,
            },
            time,
        }
    }
}
