//! Adaptive mesh refinement in 6D. Refines cells where f is large or has steep gradients.

use super::super::{init::domain::Domain, phasespace::PhaseSpaceRepr, types::*};
use std::any::Any;

/// One AMR leaf cell in 6D phase space.
pub struct AmrCell {
    pub center: [f64; 6],
    pub size: [f64; 6],
    pub value: f64,
    pub children: Option<Box<[AmrCell; 64]>>,
}

/// Adaptive mesh in 6D. Root cell spans the full domain.
pub struct AmrGrid {
    pub root: AmrCell,
    pub domain: Domain,
    pub refinement_threshold: f64,
    pub max_level: usize,
}

impl AmrGrid {
    pub fn new(domain: Domain, refinement_threshold: f64, max_levels: usize) -> Self {
        todo!()
    }

    /// Split cells where f > threshold or |∇f| > gradient_threshold.
    pub fn refine(&mut self) {
        todo!("octree split in 6D -- 2^6=64 children per cell")
    }

    /// Merge nearly-uniform children (coarsen).
    pub fn coarsen(&mut self) {
        todo!("merge children when values are nearly uniform")
    }
}

impl PhaseSpaceRepr for AmrGrid {
    fn compute_density(&self) -> DensityField {
        todo!("AMR: sum leaf cells over velocity dimensions")
    }
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64) {
        todo!("AMR: update spatial centres of all leaf cells")
    }
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        todo!("AMR: update velocity centres of all leaf cells")
    }
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        todo!("AMR: sum leaf cell contributions near given position")
    }
    fn total_mass(&self) -> f64 {
        todo!("AMR: sum f * cell_volume_6D over all leaf cells")
    }
    fn casimir_c2(&self) -> f64 {
        todo!("AMR: sum f^2 * cell_volume_6D over all leaf cells")
    }
    fn entropy(&self) -> f64 {
        todo!("AMR: sum -f*ln(f)*cell_volume_6D over leaf cells with f>0")
    }
    fn stream_count(&self) -> StreamCountField {
        todo!("AMR: count velocity maxima at each spatial position")
    }
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        todo!("AMR: collect f values from velocity leaf cells at given position")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
