//! The `PhaseSpaceRepr` trait — the central abstraction of the library.
//! All phase-space storage strategies implement this.

use super::types::{
    AccelerationField, DensityField, DisplacementField, PhaseSpaceSnapshot, StreamCountField,
    Tensor,
};

/// Central trait for all phase-space storage and manipulation strategies.
///
/// Implementations differ in memory layout and algorithmic complexity:
/// - `UniformGrid6D`: O(N⁶) brute-force grid
/// - `TensorTrain`: O(N³r³) low-rank decomposition
/// - `SheetTracker`: O(N³) Lagrangian cold sheet
pub trait PhaseSpaceRepr: Send + Sync {
    /// Integrate f over all velocities: ρ(x) = ∫f dv³.
    /// This is the coupling moment to the Poisson equation.
    fn compute_density(&self) -> DensityField;

    /// Drift sub-step: advect f in spatial coordinates by displacement Δx = v·dt.
    /// Pure translation in x at constant v.
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64);

    /// Kick sub-step: advect f in velocity coordinates by Δv = g·dt.
    /// Pure translation in v at constant x.
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64);

    /// Compute velocity moment of order n at given spatial position.
    /// Order 0 = density, 1 = mean velocity, 2 = dispersion tensor.
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor;

    /// Total mass M = ∫f dx³dv³. Should be conserved to machine precision.
    fn total_mass(&self) -> f64;

    /// Casimir invariant C₂ = ∫f² dx³dv³.
    /// Increase over time indicates numerical diffusion.
    fn casimir_c2(&self) -> f64;

    /// Boltzmann entropy S = −∫f ln f dx³dv³.
    /// Should be exactly conserved; growth = numerical error.
    fn entropy(&self) -> f64;

    /// Number of distinct velocity streams at each spatial point.
    /// Detects caustic surfaces (sheet folds).
    fn stream_count(&self) -> StreamCountField;

    /// Extract the local velocity distribution f(v|x) at a given spatial position.
    /// Used for dark matter detection predictions.
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64>;

    /// Total kinetic energy T = ½∫fv² dx³dv³.
    fn total_kinetic_energy(&self) -> f64 {
        todo!("total_kinetic_energy not implemented for this PhaseSpaceRepr")
    }

    /// Extract a full 6D snapshot of the current state.
    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot {
        todo!("to_snapshot not implemented for this PhaseSpaceRepr")
    }
}
