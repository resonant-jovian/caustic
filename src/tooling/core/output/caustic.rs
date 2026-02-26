//! Caustic surface detection, analysis, and tracking.

use super::super::{types::*, init::domain::Domain};

/// Tracks formation and evolution of caustic surfaces.
pub struct CausticDetector;

impl CausticDetector {
    /// Find voxel faces where stream count changes — these are caustic surfaces.
    pub fn detect_surfaces(stream_count: &StreamCountField, domain: &Domain) -> Vec<[f64; 3]> {
        todo!("find voxel faces where stream_count changes")
    }

    /// First timestep index where max stream count > 1.
    pub fn first_caustic_time(history: &[StreamCountField]) -> Option<f64> {
        todo!("scan history for first step where max stream count > 1")
    }

    /// Interpolate density at caustic surface positions (formally diverges at a true caustic).
    pub fn caustic_density_at(density: &DensityField, surfaces: &[[f64; 3]]) -> Vec<f64> {
        todo!("interpolate density at caustic surface positions")
    }
}
