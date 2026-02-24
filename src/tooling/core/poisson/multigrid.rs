//! Geometric multigrid solver for ∇²Φ = 4πGρ. Supports arbitrary BC. O(N³) per V-cycle.

use super::super::{types::*, solver::PoissonSolver, init::domain::{Domain, SpatialBoundType}};

/// Geometric multigrid Poisson solver.
pub struct Multigrid {
    pub levels: usize,
    pub shape: [usize; 3],
    pub bc: SpatialBoundType,
    pub n_smooth: usize,
}

impl Multigrid {
    pub fn new(domain: &Domain, levels: usize, smoothing_steps: usize) -> Self {
        todo!()
    }
}

impl PoissonSolver for Multigrid {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        todo!("V-cycle: pre-smooth (Gauss-Seidel), restrict residual, coarse solve, prolongate, post-smooth")
    }
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        todo!("FD gradient: gx = (Phi[i+1]-Phi[i-1]) / (2*dx)")
    }
}

fn smooth(phi: &mut Vec<f64>, rhs: &[f64], shape: [usize; 3], n: usize) {
    todo!("Gauss-Seidel red-black smoothing")
}

fn restrict(fine: &[f64], shape: [usize; 3]) -> Vec<f64> {
    todo!("full-weighting restriction to coarser grid")
}

fn prolongate(coarse: &[f64], shape: [usize; 3]) -> Vec<f64> {
    todo!("trilinear prolongation to finer grid")
}
