//! Tree code Poisson solver (Barnes-Hut). O(N log N) per solve.

use super::super::{init::domain::Domain, solver::PoissonSolver, types::*};

/// One node of the Barnes-Hut octree.
pub struct OctreeNode {
    pub center_of_mass: [f64; 3],
    pub total_mass: f64,
    pub size: f64,
    pub children: Option<Box<[Option<OctreeNode>; 8]>>,
}

/// Barnes-Hut tree code Poisson solver.
pub struct TreePoisson {
    /// Opening angle criterion θ for multipole approximation.
    pub opening_angle: f64,
    pub domain: Domain,
}

impl TreePoisson {
    pub fn new(domain: Domain, opening_angle: f64) -> Self {
        todo!()
    }
}

impl PoissonSolver for TreePoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        todo!("build octree from density, walk tree to compute Phi(x) via Barnes-Hut multipole")
    }
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        todo!("reuse tree walk: compute g = -grad Phi during tree traversal")
    }
}

fn build_tree(density: &DensityField) -> OctreeNode {
    todo!()
}
