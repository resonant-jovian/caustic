//! Lagrangian sheet tracker for cold initial conditions. Memory O(N³). Exact in
//! single-stream regions.

use super::super::{
    types::*,
    phasespace::PhaseSpaceRepr,
    init::{domain::Domain, cosmological::ZeldovichIC},
};

/// One Lagrangian tracer particle on the dark matter sheet.
pub struct SheetParticle {
    pub q: [f64; 3],
    pub x: [f64; 3],
    pub v: [f64; 3],
}

/// Lagrangian cold dark matter sheet.
pub struct SheetTracker {
    pub particles: Vec<SheetParticle>,
    pub shape: [usize; 3],
    pub domain: Domain,
    pub stream_threshold: f64,
}

impl SheetTracker {
    pub fn new(domain: Domain) -> Self {
        todo!()
    }

    /// Place one particle at each Lagrangian grid point, displaced by s(q).
    pub fn from_zeldovich(ic: &ZeldovichIC, domain: &Domain) -> Self {
        todo!("place one particle at each Lagrangian grid point q, displaced by s(q)")
    }

    /// Detect sheet folds (caustics) by computing det(∂x/∂q).
    pub fn detect_caustics(&self) -> StreamCountField {
        todo!("count sign changes in det(dx/dq) within each spatial cell")
    }

    /// Cloud-in-cell density interpolation from particle positions.
    pub fn interpolate_density(&self, domain: &Domain) -> DensityField {
        todo!("cloud-in-cell mass assignment")
    }
}

impl PhaseSpaceRepr for SheetTracker {
    fn compute_density(&self) -> DensityField {
        todo!("sheet tracker: CIC density from particle positions")
    }
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64) {
        todo!("sheet tracker: update x += v*dt for all particles")
    }
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        todo!("sheet tracker: update v += g(x)*dt via interpolated acceleration")
    }
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        todo!("sheet tracker: sum particle contributions at nearby cells")
    }
    fn total_mass(&self) -> f64 {
        todo!("sheet tracker: sum particle masses (conserved constant)")
    }
    fn casimir_c2(&self) -> f64 {
        todo!("sheet tracker: C2 from sheet Jacobian")
    }
    fn entropy(&self) -> f64 {
        todo!("sheet tracker: entropy from Jacobian (exact in single-stream region)")
    }
    fn stream_count(&self) -> StreamCountField {
        self.detect_caustics()
    }
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        todo!("sheet tracker: collect velocities of particles near position")
    }
}
