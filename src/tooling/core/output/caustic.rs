//! Caustic surface detection, analysis, and tracking.

use super::super::{init::domain::Domain, types::*};

/// Tracks formation and evolution of caustic surfaces.
pub struct CausticDetector;

impl CausticDetector {
    /// Find voxel faces where stream count changes — these are caustic surfaces.
    ///
    /// Checks adjacent cells in +x, +y, +z; where stream count differs,
    /// records the face center position.
    pub fn detect_surfaces(stream_count: &StreamCountField, domain: &Domain) -> Vec<[f64; 3]> {
        let [nx, ny, nz] = stream_count.shape;
        let dx = domain.dx();
        let lx = domain.lx();
        let mut surfaces = Vec::new();

        for ix in 0..nx {
            let x = -lx[0] + (ix as f64 + 0.5) * dx[0];
            for iy in 0..ny {
                let y = -lx[1] + (iy as f64 + 0.5) * dx[1];
                for iz in 0..nz {
                    let z = -lx[2] + (iz as f64 + 0.5) * dx[2];
                    let idx = ix * ny * nz + iy * nz + iz;
                    let sc = stream_count.data[idx];

                    // Check +x neighbor
                    if ix + 1 < nx {
                        let neighbor = stream_count.data[(ix + 1) * ny * nz + iy * nz + iz];
                        if sc != neighbor {
                            surfaces.push([x + 0.5 * dx[0], y, z]);
                        }
                    }
                    // Check +y neighbor
                    if iy + 1 < ny {
                        let neighbor = stream_count.data[ix * ny * nz + (iy + 1) * nz + iz];
                        if sc != neighbor {
                            surfaces.push([x, y + 0.5 * dx[1], z]);
                        }
                    }
                    // Check +z neighbor
                    if iz + 1 < nz {
                        let neighbor = stream_count.data[ix * ny * nz + iy * nz + iz + 1];
                        if sc != neighbor {
                            surfaces.push([x, y, z + 0.5 * dx[2]]);
                        }
                    }
                }
            }
        }
        surfaces
    }

    /// First timestep index where max stream count > 1.
    pub fn first_caustic_time(history: &[StreamCountField]) -> Option<f64> {
        for (i, sc) in history.iter().enumerate() {
            if sc.data.iter().any(|&c| c > 1) {
                return Some(i as f64);
            }
        }
        None
    }

    /// Nearest-cell density lookup at caustic surface positions.
    pub fn caustic_density_at(
        density: &DensityField,
        surfaces: &[[f64; 3]],
        domain: &Domain,
    ) -> Vec<f64> {
        let [nx, ny, nz] = density.shape;
        let dx = domain.dx();
        let lx = domain.lx();

        surfaces
            .iter()
            .map(|pos| {
                // Convert physical position to grid index (nearest cell)
                let ix = ((pos[0] + lx[0]) / dx[0]).floor() as usize;
                let iy = ((pos[1] + lx[1]) / dx[1]).floor() as usize;
                let iz = ((pos[2] + lx[2]) / dx[2]).floor() as usize;

                let ix = ix.min(nx.saturating_sub(1));
                let iy = iy.min(ny.saturating_sub(1));
                let iz = iz.min(nz.saturating_sub(1));

                density.data[ix * ny * nz + iy * nz + iz]
            })
            .collect()
    }
}
