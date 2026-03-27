//! The `PoissonSolver` trait. Given a density field ρ(x), produce the gravitational
//! potential Φ(x) by solving ∇²Φ = 4πGρ.

use super::context::SimContext;
use super::types::{AccelerationField, DensityField, PotentialField};

/// Trait for all gravitational Poisson solver implementations.
///
/// # Examples
///
/// ```no_run
/// use caustic::{FftPoisson, PoissonSolver, Domain, DomainBuilder, SpatialBoundType, VelocityBoundType};
///
/// let domain = Domain::builder()
///     .spatial_extent(10.0)
///     .velocity_extent(5.0)
///     .spatial_resolution(16)
///     .velocity_resolution(16)
///     .spatial_bc(SpatialBoundType::Periodic)
///     .velocity_bc(VelocityBoundType::Open)
///     .build()
///     .unwrap();
///
/// let poisson = FftPoisson::new(&domain);
/// ```
pub trait PoissonSolver {
    /// Solve ∇²Φ = 4πGρ and return the potential field. Uses `ctx.g` for the
    /// gravitational constant.
    fn solve(&self, density: &DensityField, ctx: &SimContext) -> PotentialField;

    /// Compute the gravitational acceleration g = −∇Φ via spectral differentiation
    /// or finite differences.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField;
}
