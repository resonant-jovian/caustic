//! Hybrid representation: SheetTracker in single-stream regions, UniformGrid6D in
//! multi-stream (halo interior). Switches at caustic surfaces.

use super::super::{init::domain::Domain, phasespace::PhaseSpaceRepr, types::*};
use super::{sheet::SheetTracker, uniform::UniformGrid6D};
use std::any::Any;

/// Hybrid representation combining SheetTracker and UniformGrid6D.
///
/// Single-stream regions (stream_count ≤ threshold) are tracked by the
/// Lagrangian sheet (exact characteristics, zero diffusion). Multi-stream
/// regions are handled by the Eulerian 6D grid (captures velocity dispersion).
///
/// The `mask` vector marks each spatial cell: `true` = grid mode, `false` = sheet mode.
pub struct HybridRepr {
    pub sheet: SheetTracker,
    pub grid: UniformGrid6D,
    pub domain: Domain,
    pub stream_threshold: u32,
    /// Per spatial cell: true = grid mode (multi-stream), false = sheet mode.
    pub mask: Vec<bool>,
}

impl HybridRepr {
    pub fn new(domain: Domain) -> Self {
        let sheet = SheetTracker::new(domain.clone());
        let grid = UniformGrid6D::new(domain.clone());
        let n_spatial = domain.spatial_res.x1 as usize
            * domain.spatial_res.x2 as usize
            * domain.spatial_res.x3 as usize;
        let mask = vec![false; n_spatial]; // start all-sheet

        Self {
            sheet,
            grid,
            domain,
            stream_threshold: 1,
            mask,
        }
    }

    /// Transfer particles from SheetTracker to grid where stream_count > stream_threshold.
    ///
    /// For each spatial cell where the stream count exceeds the threshold and
    /// is not yet in grid mode: deposit the sheet particles into the 6D grid
    /// via CIC and mark the cell as grid mode.
    pub fn update_interface(&mut self) {
        let counts = self.sheet.detect_caustics();
        let [nx, ny, nz] = counts.shape;

        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.domain.lx();
        let lv = self.domain.lv();
        let nv = [
            self.domain.velocity_res.v1 as usize,
            self.domain.velocity_res.v2 as usize,
            self.domain.velocity_res.v3 as usize,
        ];

        let cell_vol_6d = dx[0] * dx[1] * dx[2] * dv[0] * dv[1] * dv[2];

        // 6D grid strides (row-major: x1, x2, x3, v1, v2, v3)
        let sv3 = 1usize;
        let sv2 = nv[2];
        let sv1 = nv[1] * nv[2];
        let sx3 = nv[0] * sv1;
        let sx2 = nz * sx3;
        let sx1 = ny * sx2;

        let is_periodic = matches!(
            self.domain.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );

        // Identify newly multi-stream cells
        for si in 0..counts.data.len() {
            if counts.data[si] > self.stream_threshold && !self.mask[si] {
                self.mask[si] = true;
            }
        }

        // Deposit sheet particles that sit in grid-mode cells into the 6D grid.
        // We iterate particles; for each in a grid-mode cell, CIC deposit.
        for p in &self.sheet.particles {
            // Find which spatial cell this particle is in
            let mut skip = false;
            let mut ci = [0usize; 3];
            let ns = [nx, ny, nz];
            for k in 0..3 {
                let idx = ((p.x[k] + lx[k]) / dx[k]).floor() as isize;
                if is_periodic {
                    ci[k] = idx.rem_euclid(ns[k] as isize) as usize;
                } else if idx < 0 || idx >= ns[k] as isize {
                    skip = true;
                    break;
                } else {
                    ci[k] = idx as usize;
                }
            }
            if skip {
                continue;
            }

            let flat = ci[0] * ny * nz + ci[1] * nz + ci[2];
            if !self.mask[flat] {
                continue; // still in sheet mode
            }

            // CIC deposit into 6D grid (spatial × velocity)
            // Spatial CIC
            let mut x_ci = [0isize; 3];
            let mut x_frac = [0.0f64; 3];
            for k in 0..3 {
                let s = (p.x[k] + lx[k]) / dx[k] - 0.5;
                x_ci[k] = s.floor() as isize;
                x_frac[k] = s - x_ci[k] as f64;
            }

            // Velocity CIC
            let mut v_ci = [0isize; 3];
            let mut v_frac = [0.0f64; 3];
            for k in 0..3 {
                let s = (p.v[k] + lv[k]) / dv[k] - 0.5;
                v_ci[k] = s.floor() as isize;
                v_frac[k] = s - v_ci[k] as f64;
            }

            // Deposit to 2³ × 2³ = 64 surrounding 6D cells
            for dix in 0..2isize {
                let wx0 = if dix == 0 { 1.0 - x_frac[0] } else { x_frac[0] };
                for diy in 0..2isize {
                    let wx1 = if diy == 0 { 1.0 - x_frac[1] } else { x_frac[1] };
                    for diz in 0..2isize {
                        let wx2 = if diz == 0 { 1.0 - x_frac[2] } else { x_frac[2] };
                        let wx = wx0 * wx1 * wx2;

                        let mut ii = x_ci[0] + dix;
                        let mut jj = x_ci[1] + diy;
                        let mut kk = x_ci[2] + diz;

                        if is_periodic {
                            ii = ii.rem_euclid(nx as isize);
                            jj = jj.rem_euclid(ny as isize);
                            kk = kk.rem_euclid(nz as isize);
                        } else if ii < 0
                            || ii >= nx as isize
                            || jj < 0
                            || jj >= ny as isize
                            || kk < 0
                            || kk >= nz as isize
                        {
                            continue;
                        }

                        for div1 in 0..2isize {
                            let wv0 = if div1 == 0 {
                                1.0 - v_frac[0]
                            } else {
                                v_frac[0]
                            };
                            for div2 in 0..2isize {
                                let wv1 = if div2 == 0 {
                                    1.0 - v_frac[1]
                                } else {
                                    v_frac[1]
                                };
                                for div3 in 0..2isize {
                                    let wv2 = if div3 == 0 {
                                        1.0 - v_frac[2]
                                    } else {
                                        v_frac[2]
                                    };
                                    let wv = wv0 * wv1 * wv2;
                                    let w = wx * wv;

                                    let iv1 = v_ci[0] + div1;
                                    let iv2 = v_ci[1] + div2;
                                    let iv3 = v_ci[2] + div3;

                                    // Open velocity BC: skip out-of-bounds
                                    if iv1 < 0
                                        || iv1 >= nv[0] as isize
                                        || iv2 < 0
                                        || iv2 >= nv[1] as isize
                                        || iv3 < 0
                                        || iv3 >= nv[2] as isize
                                    {
                                        continue;
                                    }

                                    let idx6d = ii as usize * sx1
                                        + jj as usize * sx2
                                        + kk as usize * sx3
                                        + iv1 as usize * sv1
                                        + iv2 as usize * sv2
                                        + iv3 as usize * sv3;
                                    self.grid.data[idx6d] +=
                                        self.sheet.particle_mass * w / cell_vol_6d;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Helper: compute spatial cell index for a position.
    fn cell_index_3d(&self, pos: &[f64; 3]) -> Option<usize> {
        let dx = self.domain.dx();
        let lx = self.domain.lx();
        let [nx, ny, nz] = self.sheet.shape;
        let is_periodic = matches!(
            self.domain.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );

        let mut ci = [0usize; 3];
        let ns = [nx, ny, nz];
        for k in 0..3 {
            let idx = ((pos[k] + lx[k]) / dx[k]).floor() as isize;
            if is_periodic {
                ci[k] = idx.rem_euclid(ns[k] as isize) as usize;
            } else if idx < 0 || idx >= ns[k] as isize {
                return None;
            } else {
                ci[k] = idx as usize;
            }
        }
        Some(ci[0] * ny * nz + ci[1] * nz + ci[2])
    }
}

impl PhaseSpaceRepr for HybridRepr {
    fn compute_density(&self) -> DensityField {
        let sheet_density = self.sheet.compute_density();
        let grid_density = self.grid.compute_density();
        let [nx, ny, nz] = sheet_density.shape;
        let n = nx * ny * nz;

        let data: Vec<f64> = self
            .mask
            .iter()
            .zip(grid_density.data.iter())
            .zip(sheet_density.data.iter())
            .map(|((&use_grid, &gd), &sd)| if use_grid { gd } else { sd })
            .collect();

        DensityField {
            data,
            shape: [nx, ny, nz],
        }
    }

    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64) {
        // Advect sheet particles everywhere (cheap, just x += v*dt)
        self.sheet.advect_x(displacement, dt);
        // Advect grid in grid-mode cells
        self.grid.advect_x(displacement, dt);
        // Update interface after advection
        self.update_interface();
    }

    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        // Advect sheet particles everywhere (v += a*dt via trilinear interp)
        self.sheet.advect_v(acceleration, dt);
        // Advect grid in grid-mode cells
        self.grid.advect_v(acceleration, dt);
        // Update interface after advection
        self.update_interface();
    }

    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        match self.cell_index_3d(position) {
            Some(flat) if self.mask[flat] => self.grid.moment(position, order),
            _ => self.sheet.moment(position, order),
        }
    }

    fn total_mass(&self) -> f64 {
        let sheet_density = self.sheet.compute_density();
        let grid_density = self.grid.compute_density();
        let dx = self.domain.dx();
        let cell_vol = dx[0] * dx[1] * dx[2];

        let mut total = 0.0;
        for i in 0..self.mask.len() {
            if self.mask[i] {
                total += grid_density.data[i] * cell_vol;
            } else {
                total += sheet_density.data[i] * cell_vol;
            }
        }
        total
    }

    fn casimir_c2(&self) -> f64 {
        // If any cells are in sheet mode, the sheet has delta-function f → C₂ diverges.
        if self.mask.iter().any(|&m| !m) {
            return f64::INFINITY;
        }
        // All grid mode: use grid C₂
        self.grid.casimir_c2()
    }

    fn entropy(&self) -> f64 {
        // Sheet cells have zero entropy; grid cells contribute normally.
        if self.mask.iter().all(|&m| !m) {
            return 0.0; // all sheet
        }
        if self.mask.iter().all(|&m| m) {
            return self.grid.entropy(); // all grid
        }
        // Mixed: grid entropy (sheet contribution is 0)
        self.grid.entropy()
    }

    fn stream_count(&self) -> StreamCountField {
        let sheet_streams = self.sheet.detect_caustics();
        let [nx, ny, nz] = sheet_streams.shape;
        let n = nx * ny * nz;

        // Stream counts come from the sheet regardless of mask state
        let data = sheet_streams.data.clone();

        StreamCountField {
            data,
            shape: [nx, ny, nz],
        }
    }

    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        match self.cell_index_3d(position) {
            Some(flat) if self.mask[flat] => self.grid.velocity_distribution(position),
            _ => self.sheet.velocity_distribution(position),
        }
    }

    fn total_kinetic_energy(&self) -> f64 {
        // Sheet particles carry kinetic energy in all regions.
        // Grid cells carry kinetic energy only in grid-mode regions.
        // Since sheet particles are everywhere but only contribute in sheet-mode cells,
        // we use the partitioned approach.
        let sheet_ke = self.sheet.total_kinetic_energy();
        let grid_ke = self.grid.total_kinetic_energy();

        // Simple approach: if no grid cells, pure sheet. If all grid, pure grid.
        // In mixed mode, sheet KE covers sheet cells, grid KE covers grid cells.
        // Since we can't easily partition sheet KE by cell, use sheet for all
        // (sheet tracks the canonical particle trajectories).
        if self.mask.iter().all(|&m| m) {
            grid_ke
        } else {
            sheet_ke
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::init::domain::{DomainBuilder, SpatialBoundType, VelocityBoundType};

    fn test_domain(n: i128) -> Domain {
        DomainBuilder::new()
            .spatial_extent(2.0)
            .velocity_extent(2.0)
            .spatial_resolution(n)
            .velocity_resolution(n)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn hybrid_single_stream_matches_sheet() {
        // When all cells are in sheet mode, hybrid should match pure SheetTracker.
        let domain = test_domain(4);
        let hybrid = HybridRepr::new(domain.clone());
        let sheet = SheetTracker::new(domain);

        let h_density = hybrid.compute_density();
        let s_density = sheet.compute_density();

        let max_diff: f64 = h_density
            .data
            .iter()
            .zip(s_density.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff < 1e-14,
            "Hybrid should match sheet in single-stream mode, max diff = {max_diff}"
        );
    }

    #[test]
    fn hybrid_mass_conservation() {
        let domain = test_domain(4);
        let hybrid = HybridRepr::new(domain);

        let mass = hybrid.total_mass();
        // Sheet places one particle per cell, particle_mass = 1/N^3 = 1/64
        // Total mass should be 1.0
        assert!(
            (mass - 1.0).abs() < 1e-10,
            "Total mass should be 1.0, got {mass}"
        );
    }

    #[test]
    fn hybrid_forced_transition() {
        // Place particles so that multiple land in the same spatial cell,
        // triggering the transition to grid mode.
        let domain = test_domain(4);
        let mut hybrid = HybridRepr::new(domain);

        // Move several particles to the same location → stream count > 1
        let n = hybrid.sheet.particles.len();
        if n > 2 {
            let target_x = hybrid.sheet.particles[0].x;
            hybrid.sheet.particles[1].x = target_x;
            hybrid.sheet.particles[2].x = target_x;
        }

        // Before update: all sheet mode
        assert!(
            hybrid.mask.iter().all(|&m| !m),
            "Should start in all-sheet mode"
        );

        // Update interface: should detect multi-stream and switch some cells
        hybrid.update_interface();

        // At least one cell should be in grid mode now
        let grid_cells = hybrid.mask.iter().filter(|&&m| m).count();
        assert!(
            grid_cells > 0,
            "Should have at least one grid-mode cell after forced multi-stream"
        );
    }
}
