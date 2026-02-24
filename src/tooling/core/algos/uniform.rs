//! Brute-force uniform 6D grid. Memory O(N⁶). Simple and correct; primary reference
//! implementation for validation.

use super::super::{types::*, phasespace::PhaseSpaceRepr, init::domain::Domain};

/// Stores f on a uniform (Nx1×Nx2×Nx3×Nv1×Nv2×Nv3) grid as a flat `Vec<f64>`.
pub struct UniformGrid6D {
    pub data: Vec<f64>,
    pub domain: Domain,
}

impl UniformGrid6D {
    /// Allocate Nx³ × Nv³ floats, zero-initialised.
    pub fn new(domain: Domain) -> Self {
        todo!("allocate Nx^3 * Nv^3 floats, zero-initialised")
    }

    pub fn from_snapshot(snap: PhaseSpaceSnapshot, domain: Domain) -> Self {
        todo!()
    }

    /// Linear index into flat Vec from (ix1, ix2, ix3, iv1, iv2, iv3) — row-major 6D.
    pub fn index(&self, ix: [usize; 3], iv: [usize; 3]) -> usize {
        todo!("row-major 6D index")
    }
}

impl PhaseSpaceRepr for UniformGrid6D {
    fn compute_density(&self) -> DensityField {
        todo!("uniform grid: sum f over velocity axes, multiply by dv^3")
    }
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64) {
        todo!("uniform grid: semi-Lagrangian shift in x by v*dt with cubic spline")
    }
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        todo!("uniform grid: semi-Lagrangian shift in v by g*dt")
    }
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        todo!("uniform grid: integrate f * v^n over velocity at given x cell")
    }
    fn total_mass(&self) -> f64 {
        todo!("uniform grid: sum all cells * dx^3 * dv^3")
    }
    fn casimir_c2(&self) -> f64 {
        todo!("uniform grid: sum f^2 * dx^3 * dv^3")
    }
    fn entropy(&self) -> f64 {
        todo!("uniform grid: sum -f*ln(f) * dx^3 * dv^3, skip f=0 cells")
    }
    fn stream_count(&self) -> StreamCountField {
        todo!("uniform grid: count local maxima in f(v|x) at each spatial point")
    }
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        todo!("uniform grid: slice 6D array at nearest x grid cell")
    }
}
