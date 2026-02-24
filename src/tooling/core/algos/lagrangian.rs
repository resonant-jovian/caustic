//! Semi-Lagrangian advector. Traces characteristics backwards from each grid point and
//! interpolates. Eliminates CFL constraint; Δt limited only by accuracy requirements.

use super::super::{types::*, phasespace::PhaseSpaceRepr, advecator::Advector};

/// Semi-Lagrangian advector with cubic spline interpolation.
pub struct SemiLagrangian {
    pub interpolation_order: usize,
}

impl SemiLagrangian {
    pub fn new() -> Self {
        todo!()
    }
}

impl Advector for SemiLagrangian {
    fn drift(&self, repr: &mut dyn PhaseSpaceRepr, dt: f64) {
        todo!("for each velocity cell v_j, shift f[:,v_j,:] by d=v_j*dt in x using cubic spline; departure point x-d need not be on grid")
    }
    fn kick(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64) {
        todo!("for each spatial cell x_i, shift f[x_i,:] by a_i*dt in v using cubic spline")
    }
    fn step(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64) {
        todo!("single full drift+kick; prefer TimeIntegrator splitting for accuracy")
    }
}

/// 4-point cubic spline interpolation at fractional position `x` in array of length `n`.
pub fn cubic_spline_interpolate(data: &[f64], x: f64, n: usize) -> f64 {
    todo!("4-point cubic interpolation with periodic/open BC")
}
