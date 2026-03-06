//! Spherical-harmonics Poisson solver. Natural for nearly-spherical halos.
//! O(N² l_max²).

use super::super::{solver::PoissonSolver, types::*};

/// Spherical-harmonics expansion Poisson solver.
pub struct SphericalHarmonicsPoisson {
    pub l_max: usize,
    pub n_radial: usize,
}

impl SphericalHarmonicsPoisson {
    pub fn new(l_max: usize, n_radial: usize) -> Self {
        todo!()
    }
}

impl PoissonSolver for SphericalHarmonicsPoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        todo!("decompose rho into Y_lm, solve 1D radial Poisson for each (l,m), reconstruct Phi")
    }
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        todo!("analytic gradient in spherical coords, convert to Cartesian")
    }
}
