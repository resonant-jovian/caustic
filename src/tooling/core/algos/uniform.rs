//! Brute-force uniform 6D grid. Memory O(N⁶). Simple and correct; primary reference
//! implementation for validation.

use super::super::{
    init::domain::{Domain, SpatialBoundType, VelocityBoundType},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use super::lagrangian::{sl_shift_1d, sl_shift_1d_into};
use rayon::prelude::*;
use std::any::Any;

/// Stores f on a uniform (Nx1×Nx2×Nx3×Nv1×Nv2×Nv3) grid as a flat `Vec<f64>`.
/// Index order: x1 fastest-changing outer, v3 fastest-changing inner (row-major).
pub struct UniformGrid6D {
    pub data: Vec<f64>,
    pub domain: Domain,
    // Cached derived values (computed once at construction, avoids repeated Decimal→f64).
    cached_sizes: [usize; 6],
    cached_lx: [f64; 3],
    cached_lv: [f64; 3],
    cached_dx: [f64; 3],
    cached_dv: [f64; 3],
}

struct CachedGrid {
    sizes: [usize; 6],
    lx: [f64; 3],
    lv: [f64; 3],
    dx: [f64; 3],
    dv: [f64; 3],
}

impl CachedGrid {
    fn from_domain(domain: &Domain) -> Self {
        Self {
            sizes: [
                domain.spatial_res.x1 as usize,
                domain.spatial_res.x2 as usize,
                domain.spatial_res.x3 as usize,
                domain.velocity_res.v1 as usize,
                domain.velocity_res.v2 as usize,
                domain.velocity_res.v3 as usize,
            ],
            lx: domain.lx(),
            lv: domain.lv(),
            dx: domain.dx(),
            dv: domain.dv(),
        }
    }
}

impl UniformGrid6D {
    /// Allocate Nx³ × Nv³ floats, zero-initialised.
    pub fn new(domain: Domain) -> Self {
        let n = domain.total_cells();
        let c = CachedGrid::from_domain(&domain);
        Self {
            data: vec![0.0; n],
            domain,
            cached_sizes: c.sizes,
            cached_lx: c.lx,
            cached_lv: c.lv,
            cached_dx: c.dx,
            cached_dv: c.dv,
        }
    }

    pub fn from_snapshot(snap: PhaseSpaceSnapshot, domain: Domain) -> Self {
        assert_eq!(
            snap.data.len(),
            domain.total_cells(),
            "snapshot size mismatch: {} vs {}",
            snap.data.len(),
            domain.total_cells()
        );
        let c = CachedGrid::from_domain(&domain);
        Self {
            data: snap.data,
            domain,
            cached_sizes: c.sizes,
            cached_lx: c.lx,
            cached_lv: c.lv,
            cached_dx: c.dx,
            cached_dv: c.dv,
        }
    }

    /// Linear index into flat Vec from (ix1, ix2, ix3, iv1, iv2, iv3) — row-major 6D.
    #[inline]
    pub fn index(&self, ix: [usize; 3], iv: [usize; 3]) -> usize {
        let [_, nx2, nx3, nv1, nv2, nv3] = self.cached_sizes;
        let s_v3 = 1;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;
        ix[0] * s_x1 + ix[1] * s_x2 + ix[2] * s_x3 + iv[0] * s_v1 + iv[1] * s_v2 + iv[2] * s_v3
    }

    #[inline]
    pub(crate) fn sizes(&self) -> [usize; 6] {
        self.cached_sizes
    }

    #[inline]
    fn lx(&self) -> [f64; 3] {
        self.cached_lx
    }

    #[inline]
    fn lv(&self) -> [f64; 3] {
        self.cached_lv
    }
}

impl PhaseSpaceRepr for UniformGrid6D {
    fn compute_density(&self) -> DensityField {
        let _span = tracing::info_span!("compute_density").entered();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.cached_dv;
        let dv3 = dv[0] * dv[1] * dv[2];

        let n_spatial = nx1 * nx2 * nx3;
        let n_vel = nv1 * nv2 * nv3;
        let data: Vec<f64> = (0..n_spatial)
            .into_par_iter()
            .map(|si| {
                let base = si * n_vel;
                self.data[base..base + n_vel].iter().sum::<f64>() * dv3
            })
            .collect();

        DensityField {
            data,
            shape: [nx1, nx2, nx3],
        }
    }

    fn advect_x(&mut self, _displacement: &DisplacementField, dt: f64) {
        let _span = tracing::info_span!("advect_x").entered();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        let lx = self.cached_lx;
        let lv = self.cached_lv;
        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let flat_idx = |ix: [usize; 3], iv: [usize; 3]| -> usize {
            ix[0] * s_x1 + ix[1] * s_x2 + ix[2] * s_x3 + iv[0] * s_v1 + iv[1] * s_v2 + iv[2] * s_v3
        };

        let n_vel = nv1 * nv2 * nv3;
        // Take ownership of data to avoid a full clone; the parallel iterator
        // reads from `src` and writes results into a new vec.
        let src = std::mem::take(&mut self.data);
        let n_total = src.len();
        let n_sp = nx1 * nx2 * nx3;
        let max_n = nx1.max(nx2).max(nx3);

        // Each velocity cell can be shifted independently in parallel.
        // Build new data directly instead of scatter-back.
        let mut new_data = vec![0.0f64; n_total];

        // Process each velocity cell in parallel, writing into new_data slices.
        // We partition new_data by velocity cell: new_data is indexed by
        // [ix1][ix2][ix3][iv1][iv2][iv3], so velocity cells are interleaved.
        // Collect shifted blocks and scatter back.
        let shifted_blocks: Vec<(usize, Vec<f64>)> = (0..n_vel)
            .into_par_iter()
            .map(|vi| {
                let iv3 = vi % nv3;
                let iv2 = (vi / nv3) % nv2;
                let iv1 = vi / (nv2 * nv3);
                let iv = [iv1, iv2, iv3];
                let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                let disp = [vx * dt, vy * dt, vz * dt];

                // Pre-allocate reusable line buffers
                let mut line = vec![0.0f64; max_n];
                let mut shifted = vec![0.0f64; max_n];
                let mut local = vec![0.0f64; n_sp];

                // Extract spatial slice for this velocity cell
                for ix1 in 0..nx1 {
                    for ix2 in 0..nx2 {
                        for ix3 in 0..nx3 {
                            let si = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                            local[si] = src[flat_idx([ix1, ix2, ix3], iv)];
                        }
                    }
                }

                // Shift along x1
                for ix2 in 0..nx2 {
                    for ix3 in 0..nx3 {
                        for ix1 in 0..nx1 {
                            line[ix1] = local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3];
                        }
                        sl_shift_1d_into(
                            &line[..nx1],
                            disp[0],
                            dx[0],
                            nx1,
                            lx[0],
                            periodic,
                            &mut shifted,
                        );
                        for ix1 in 0..nx1 {
                            local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3] = shifted[ix1];
                        }
                    }
                }

                // Shift along x2
                for ix1 in 0..nx1 {
                    for ix3 in 0..nx3 {
                        for ix2 in 0..nx2 {
                            line[ix2] = local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3];
                        }
                        sl_shift_1d_into(
                            &line[..nx2],
                            disp[1],
                            dx[1],
                            nx2,
                            lx[1],
                            periodic,
                            &mut shifted,
                        );
                        for ix2 in 0..nx2 {
                            local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3] = shifted[ix2];
                        }
                    }
                }

                // Shift along x3
                for ix1 in 0..nx1 {
                    for ix2 in 0..nx2 {
                        for ix3 in 0..nx3 {
                            line[ix3] = local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3];
                        }
                        sl_shift_1d_into(
                            &line[..nx3],
                            disp[2],
                            dx[2],
                            nx3,
                            lx[2],
                            periodic,
                            &mut shifted,
                        );
                        for ix3 in 0..nx3 {
                            local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3] = shifted[ix3];
                        }
                    }
                }

                (vi, local)
            })
            .collect();

        // Scatter results back
        for (vi, local) in shifted_blocks {
            let iv3 = vi % nv3;
            let iv2 = (vi / nv3) % nv2;
            let iv1 = vi / (nv2 * nv3);
            let iv = [iv1, iv2, iv3];
            for ix1 in 0..nx1 {
                for ix2 in 0..nx2 {
                    for ix3 in 0..nx3 {
                        let si = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                        new_data[flat_idx([ix1, ix2, ix3], iv)] = local[si];
                    }
                }
            }
        }
        self.data = new_data;
    }

    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        let _span = tracing::info_span!("advect_v").entered();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.cached_dv;
        let lv = self.cached_lv;
        let periodic_v = matches!(self.domain.velocity_bc, VelocityBoundType::Truncated);

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let flat_idx = |ix: [usize; 3], iv: [usize; 3]| -> usize {
            ix[0] * s_x1 + ix[1] * s_x2 + ix[2] * s_x3 + iv[0] * s_v1 + iv[1] * s_v2 + iv[2] * s_v3
        };

        let n_spatial = nx1 * nx2 * nx3;
        let n_vel = nv1 * nv2 * nv3;
        let max_nv = nv1.max(nv2).max(nv3);
        let src = std::mem::take(&mut self.data);
        let n_total = src.len();

        let shifted_blocks: Vec<(usize, Vec<f64>)> = (0..n_spatial)
            .into_par_iter()
            .map(|si| {
                let ix3 = si % nx3;
                let ix2 = (si / nx3) % nx2;
                let ix1 = si / (nx2 * nx3);
                let ix = [ix1, ix2, ix3];
                let flat_ix = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                let ax = acceleration.gx[flat_ix];
                let ay = acceleration.gy[flat_ix];
                let az = acceleration.gz[flat_ix];
                let disp = [ax * dt, ay * dt, az * dt];

                // Pre-allocate reusable line buffers
                let mut line = vec![0.0f64; max_nv];
                let mut shifted = vec![0.0f64; max_nv];
                let mut local = vec![0.0f64; n_vel];

                // Extract velocity slice for this spatial cell
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            let vi = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                            local[vi] = src[flat_idx(ix, [iv1, iv2, iv3])];
                        }
                    }
                }

                // Shift along v1
                for iv2 in 0..nv2 {
                    for iv3 in 0..nv3 {
                        for iv1 in 0..nv1 {
                            line[iv1] = local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3];
                        }
                        sl_shift_1d_into(
                            &line[..nv1],
                            disp[0],
                            dv[0],
                            nv1,
                            lv[0],
                            periodic_v,
                            &mut shifted,
                        );
                        for iv1 in 0..nv1 {
                            local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3] = shifted[iv1];
                        }
                    }
                }

                // Shift along v2
                for iv1 in 0..nv1 {
                    for iv3 in 0..nv3 {
                        for iv2 in 0..nv2 {
                            line[iv2] = local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3];
                        }
                        sl_shift_1d_into(
                            &line[..nv2],
                            disp[1],
                            dv[1],
                            nv2,
                            lv[1],
                            periodic_v,
                            &mut shifted,
                        );
                        for iv2 in 0..nv2 {
                            local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3] = shifted[iv2];
                        }
                    }
                }

                // Shift along v3
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            line[iv3] = local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3];
                        }
                        sl_shift_1d_into(
                            &line[..nv3],
                            disp[2],
                            dv[2],
                            nv3,
                            lv[2],
                            periodic_v,
                            &mut shifted,
                        );
                        for iv3 in 0..nv3 {
                            local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3] = shifted[iv3];
                        }
                    }
                }

                (si, local)
            })
            .collect();

        // Scatter results back
        let mut new_data = vec![0.0f64; n_total];
        for (si, local) in shifted_blocks {
            let ix3 = si % nx3;
            let ix2 = (si / nx3) % nx2;
            let ix1 = si / (nx2 * nx3);
            let ix = [ix1, ix2, ix3];
            for iv1 in 0..nv1 {
                for iv2 in 0..nv2 {
                    for iv3 in 0..nv3 {
                        let vi = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                        new_data[flat_idx(ix, [iv1, iv2, iv3])] = local[vi];
                    }
                }
            }
        }
        self.data = new_data;
    }

    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        let lx = self.cached_lx;
        let lv = self.cached_lv;
        let dv3 = dv[0] * dv[1] * dv[2];

        let ix1 = ((position[0] + lx[0]) / dx[0])
            .floor()
            .clamp(0.0, (nx1 - 1) as f64) as usize;
        let ix2 = ((position[1] + lx[1]) / dx[1])
            .floor()
            .clamp(0.0, (nx2 - 1) as f64) as usize;
        let ix3 = ((position[2] + lx[2]) / dx[2])
            .floor()
            .clamp(0.0, (nx3 - 1) as f64) as usize;

        match order {
            0 => {
                let base = self.index([ix1, ix2, ix3], [0, 0, 0]);
                let n_vel = nv1 * nv2 * nv3;
                let sum: f64 = self.data[base..base + n_vel].iter().sum::<f64>() * dv3;
                Tensor {
                    data: vec![sum],
                    rank: 0,
                    shape: vec![],
                }
            }
            1 => {
                let vc1: Vec<f64> = (0..nv1).map(|i| -lv[0] + (i as f64 + 0.5) * dv[0]).collect();
                let vc2: Vec<f64> = (0..nv2).map(|i| -lv[1] + (i as f64 + 0.5) * dv[1]).collect();
                let vc3: Vec<f64> = (0..nv3).map(|i| -lv[2] + (i as f64 + 0.5) * dv[2]).collect();
                let base = self.index([ix1, ix2, ix3], [0, 0, 0]);
                let block = &self.data[base..base + nv1 * nv2 * nv3];
                let mut vbar = [0.0f64; 3];
                let mut rho = 0.0f64;
                for (iv1, &vx) in vc1.iter().enumerate() {
                    for (iv2, &vy) in vc2.iter().enumerate() {
                        let row = &block[iv1 * nv2 * nv3 + iv2 * nv3..][..nv3];
                        for (&f, &vz) in row.iter().zip(vc3.iter()) {
                            vbar[0] += f * vx;
                            vbar[1] += f * vy;
                            vbar[2] += f * vz;
                            rho += f;
                        }
                    }
                }
                rho *= dv3;
                let scale = if rho > 1e-30 { dv3 / rho } else { 0.0 };
                Tensor {
                    data: vec![vbar[0] * scale, vbar[1] * scale, vbar[2] * scale],
                    rank: 1,
                    shape: vec![3],
                }
            }
            2 => {
                let vc1: Vec<f64> = (0..nv1).map(|i| -lv[0] + (i as f64 + 0.5) * dv[0]).collect();
                let vc2: Vec<f64> = (0..nv2).map(|i| -lv[1] + (i as f64 + 0.5) * dv[1]).collect();
                let vc3: Vec<f64> = (0..nv3).map(|i| -lv[2] + (i as f64 + 0.5) * dv[2]).collect();
                let base = self.index([ix1, ix2, ix3], [0, 0, 0]);
                let block = &self.data[base..base + nv1 * nv2 * nv3];
                let mut m2 = [0.0f64; 9];
                for (iv1, &vx) in vc1.iter().enumerate() {
                    for (iv2, &vy) in vc2.iter().enumerate() {
                        let row = &block[iv1 * nv2 * nv3 + iv2 * nv3..][..nv3];
                        for (&f, &vz) in row.iter().zip(vc3.iter()) {
                            let v = [vx, vy, vz];
                            for i in 0..3 {
                                for j in 0..3 {
                                    m2[i * 3 + j] += f * v[i] * v[j];
                                }
                            }
                        }
                    }
                }
                Tensor {
                    data: m2.iter().map(|&x| x * dv3).collect(),
                    rank: 2,
                    shape: vec![3, 3],
                }
            }
            _ => Tensor {
                data: vec![],
                rank: order,
                shape: vec![],
            },
        }
    }

    fn total_mass(&self) -> f64 {
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        self.data.par_iter().sum::<f64>() * dx3 * dv3
    }

    fn casimir_c2(&self) -> f64 {
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        self.data.par_iter().map(|&f| f * f).sum::<f64>() * dx3 * dv3
    }

    fn entropy(&self) -> f64 {
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        self.data
            .par_iter()
            .filter(|&&f| f > 0.0)
            .map(|&f| -f * f.ln())
            .sum::<f64>()
            * dx3
            * dv3
    }

    fn stream_count(&self) -> StreamCountField {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.cached_dv;
        let dv23 = dv[1] * dv[2];

        let out: Vec<u32> = (0..nx1 * nx2 * nx3)
            .into_par_iter()
            .map(|si| {
                let ix1 = si / (nx2 * nx3);
                let ix2 = (si / nx3) % nx2;
                let ix3 = si % nx3;
                let marginal: Vec<f64> = (0..nv1)
                    .map(|iv1| {
                        (0..nv2 * nv3)
                            .map(|vi23| {
                                let iv3 = vi23 % nv3;
                                let iv2 = vi23 / nv3;
                                self.data[self.index([ix1, ix2, ix3], [iv1, iv2, iv3])]
                            })
                            .sum::<f64>()
                            * dv23
                    })
                    .collect();

                let mut peaks = 0u32;
                for i in 1..nv1.saturating_sub(1) {
                    if marginal[i] > marginal[i - 1] && marginal[i] > marginal[i + 1] {
                        peaks += 1;
                    }
                }
                peaks
            })
            .collect();

        StreamCountField {
            data: out,
            shape: [nx1, nx2, nx3],
        }
    }

    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.cached_dx;
        let lx = self.cached_lx;

        let ix1 = ((position[0] + lx[0]) / dx[0])
            .floor()
            .clamp(0.0, (nx1 - 1) as f64) as usize;
        let ix2 = ((position[1] + lx[1]) / dx[1])
            .floor()
            .clamp(0.0, (nx2 - 1) as f64) as usize;
        let ix3 = ((position[2] + lx[2]) / dx[2])
            .floor()
            .clamp(0.0, (nx3 - 1) as f64) as usize;

        let base = self.index([ix1, ix2, ix3], [0, 0, 0]);
        let n_vel = nv1 * nv2 * nv3;
        self.data[base..base + n_vel].to_vec()
    }

    fn total_kinetic_energy(&self) -> f64 {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        let lv = self.cached_lv;

        let n_spatial = nx1 * nx2 * nx3;
        let n_vel = nv1 * nv2 * nv3;

        // Pre-compute v² table — same for every spatial cell
        let v2_table: Vec<f64> = (0..n_vel)
            .map(|vi| {
                let iv1 = vi / (nv2 * nv3);
                let iv2 = (vi / nv3) % nv2;
                let iv3 = vi % nv3;
                let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                vx * vx + vy * vy + vz * vz
            })
            .collect();

        let t: f64 = (0..n_spatial)
            .into_par_iter()
            .map(|si| {
                let base = si * n_vel;
                self.data[base..base + n_vel]
                    .iter()
                    .zip(v2_table.iter())
                    .map(|(&f, &v2)| f * v2)
                    .sum::<f64>()
            })
            .sum();

        0.5 * t * dx3 * dv3
    }

    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        PhaseSpaceSnapshot {
            data: self.data.clone(),
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time,
        }
    }

    fn load_snapshot(&mut self, snap: PhaseSpaceSnapshot) {
        assert_eq!(snap.data.len(), self.data.len(), "snapshot size mismatch");
        self.data = snap.data;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
    }
}
