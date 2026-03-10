//! Hybrid representation: SheetTracker in single-stream regions, UniformGrid6D in
//! multi-stream (halo interior). Switches at caustic surfaces.

use super::super::{init::domain::Domain, phasespace::PhaseSpaceRepr, types::*};
use super::{sheet::SheetTracker, uniform::UniformGrid6D};
use std::any::Any;

/// Hybrid representation combining SheetTracker and UniformGrid6D.
pub struct HybridRepr {
    pub sheet: SheetTracker,
    pub grid: UniformGrid6D,
    pub domain: Domain,
    pub stream_threshold: u32,
}

impl HybridRepr {
    pub fn new(domain: Domain) -> Self {
        todo!()
    }

    /// Transfer particles from SheetTracker to grid where stream_count > stream_threshold.
    pub fn update_interface(&mut self) {
        todo!("detect caustics, rasterise sheet particles into grid in multi-stream regions")
    }
}

impl PhaseSpaceRepr for HybridRepr {
    fn compute_density(&self) -> DensityField {
        todo!("hybrid: sheet CIC density + grid density")
    }
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64) {
        todo!("hybrid: delegate to sheet or grid per cell stream count")
    }
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        todo!("hybrid: delegate to sheet or grid per cell stream count")
    }
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        todo!("hybrid: sum sheet and grid contributions")
    }
    fn total_mass(&self) -> f64 {
        todo!("hybrid: sheet particle masses + grid integral")
    }
    fn casimir_c2(&self) -> f64 {
        todo!("hybrid: sheet Jacobian C2 + grid C2")
    }
    fn entropy(&self) -> f64 {
        todo!("hybrid: sheet entropy + grid entropy")
    }
    fn stream_count(&self) -> StreamCountField {
        todo!("hybrid: max of sheet and grid stream counts")
    }
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        todo!("hybrid: combine sheet particle velocities and grid velocity slice")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
