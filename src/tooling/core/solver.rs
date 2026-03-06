//! The `PoissonSolver` trait. Given a density field ρ(x), produce the gravitational
//! potential Φ(x) by solving ∇²Φ = 4πGρ.

use super::types::{AccelerationField, DensityField, PotentialField};

/// Trait for all gravitational Poisson solver implementations.
pub trait PoissonSolver {
    /// Solve ∇²Φ = 4πGρ and return the potential field.
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField;

    /// Compute the gravitational acceleration g = −∇Φ via spectral differentiation
    /// or finite differences.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField;
}
