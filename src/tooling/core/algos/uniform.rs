//! Brute-force uniform 6D grid. Memory O(N⁶). Simple and correct; primary reference
//! implementation for validation.

use super::super::{
    init::domain::{Domain, SpatialBoundType, VelocityBoundType},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use super::lagrangian::{sl_shift_1d, sl_shift_1d_into};
use super::mp7::mp7_shift_1d_into;
use super::wpfc::{AdvectionScheme, wpfc_shift_1d_into, zhang_shu_limiter};
use rayon::prelude::*;
use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    static SHIFT_SCRATCH: RefCell<(Vec<f64>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// Dispatch 1D shift to the selected advection scheme.
#[inline]
#[allow(clippy::too_many_arguments)]
fn shift_1d_dispatch(
    scheme: AdvectionScheme,
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    periodic: bool,
    out: &mut [f64],
) {
    match scheme {
        AdvectionScheme::CatmullRom => sl_shift_1d_into(data, disp, cell_size, n, l, periodic, out),
        AdvectionScheme::Wpfc => wpfc_shift_1d_into(data, disp, cell_size, n, l, periodic, out),
        AdvectionScheme::Mp7 => mp7_shift_1d_into(data, disp, cell_size, n, l, periodic, out),
    }
}

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
    /// Persistent scratch buffer (same size as `data`), reused across steps to
    /// eliminate per-step heap allocations in advect_x / advect_v.
    scratch: Vec<f64>,
    advection_scheme: AdvectionScheme,
    positivity_limiter: bool,
    progress: Option<Arc<super::super::progress::StepProgress>>,
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
            scratch: vec![0.0; n],
            advection_scheme: AdvectionScheme::default(),
            positivity_limiter: false,
            progress: None,
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
        let n = snap.data.len();
        let c = CachedGrid::from_domain(&domain);
        Self {
            data: snap.data,
            domain,
            cached_sizes: c.sizes,
            cached_lx: c.lx,
            cached_lv: c.lv,
            cached_dx: c.dx,
            cached_dv: c.dv,
            scratch: vec![0.0; n],
            advection_scheme: AdvectionScheme::default(),
            positivity_limiter: false,
            progress: None,
        }
    }

    /// Select the 1D interpolation scheme used in advect_x / advect_v.
    pub fn with_advection_scheme(mut self, scheme: AdvectionScheme) -> Self {
        self.advection_scheme = scheme;
        self
    }

    /// Enable the Zhang-Shu positivity-preserving limiter after each advection step.
    pub fn with_positivity_limiter(mut self, enabled: bool) -> Self {
        self.positivity_limiter = enabled;
        self
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
    fn set_progress(&mut self, p: std::sync::Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    fn compute_density(&self) -> DensityField {
        let _span = tracing::info_span!("compute_density").entered();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.cached_dv;
        let dv3 = dv[0] * dv[1] * dv[2];

        let n_spatial = nx1 * nx2 * nx3;
        let n_vel = nv1 * nv2 * nv3;
        let counter = AtomicU64::new(0);
        let report_interval = (n_spatial / 100).max(1) as u64;
        let data: Vec<f64> = (0..n_spatial)
            .into_par_iter()
            .map(|si| {
                let base = si * n_vel;
                let result = self.data[base..base + n_vel].iter().sum::<f64>() * dv3;
                if let Some(ref p) = self.progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, n_spatial as u64);
                    }
                }
                result
            })
            .collect();

        DensityField {
            data,
            shape: [nx1, nx2, nx3],
        }
    }

    fn advect_x(&mut self, _displacement: &DisplacementField, dt: f64) {
        let _span = tracing::info_span!("advect_x").entered();
        let progress = self.progress.clone();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        let lx = self.cached_lx;
        let lv = self.cached_lv;
        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);
        let scheme = self.advection_scheme;
        let positivity = self.positivity_limiter;

        let n_vel = nv1 * nv2 * nv3;
        let n_sp = nx1 * nx2 * nx3;
        let max_n = nx1.max(nx2).max(nx3);

        // Reuse persistent buffers: swap data ↔ scratch to eliminate per-step
        // heap allocations. After the swap, scratch holds source data and data
        // holds a pre-allocated workspace for intermediates.
        std::mem::swap(&mut self.data, &mut self.scratch);
        let src = std::mem::take(&mut self.scratch);
        let mut intermediates = std::mem::take(&mut self.data);

        let counter = AtomicU64::new(0);
        let report_interval = (n_vel / 100).max(1) as u64;

        intermediates
            .par_chunks_mut(n_sp)
            .enumerate()
            .for_each(|(vi, local)| {
                let iv3 = vi % nv3;
                let iv2 = (vi / nv3) % nv2;
                let iv1 = vi / (nv2 * nv3);
                let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                let disp = [vx * dt, vy * dt, vz * dt];

                // Extract spatial slice for this velocity cell
                let iv_offset = iv1 * (nv2 * nv3) + iv2 * nv3 + iv3;
                for si in 0..n_sp {
                    local[si] = src[si * n_vel + iv_offset];
                }

                // Thread-local scratch for 1D shifts (zero contention, no atomics).
                SHIFT_SCRATCH.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let (ref mut line, ref mut shifted) = *guard;
                    if line.len() < max_n {
                        line.resize(max_n, 0.0);
                        shifted.resize(max_n, 0.0);
                    }

                    // Shift along x1
                    for ix2 in 0..nx2 {
                        for ix3 in 0..nx3 {
                            for ix1 in 0..nx1 {
                                line[ix1] = local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3];
                            }
                            shift_1d_dispatch(
                                scheme,
                                &line[..nx1],
                                disp[0],
                                dx[0],
                                nx1,
                                lx[0],
                                periodic,
                                shifted,
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
                            shift_1d_dispatch(
                                scheme,
                                &line[..nx2],
                                disp[1],
                                dx[1],
                                nx2,
                                lx[1],
                                periodic,
                                shifted,
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
                            shift_1d_dispatch(
                                scheme,
                                &line[..nx3],
                                disp[2],
                                dx[2],
                                nx3,
                                lx[2],
                                periodic,
                                shifted,
                            );
                            for ix3 in 0..nx3 {
                                local[ix1 * nx2 * nx3 + ix2 * nx3 + ix3] = shifted[ix3];
                            }
                        }
                    }
                });

                if positivity {
                    zhang_shu_limiter(local, 0.0);
                }

                if let Some(ref p) = progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, n_vel as u64);
                    }
                }
            });

        // Parallel transpose: intermediates[vi * n_sp + si] → new_data[si * n_vel + vi]
        // Reuse the src allocation (reading is done) as the output buffer.
        let mut new_data = src;
        new_data
            .par_chunks_mut(n_vel)
            .enumerate()
            .for_each(|(si, vel_block)| {
                for vi in 0..n_vel {
                    vel_block[vi] = intermediates[vi * n_sp + si];
                }
            });
        self.data = new_data;
        self.scratch = intermediates;
    }

    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        let _span = tracing::info_span!("advect_v").entered();
        let progress = self.progress.clone();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.cached_dv;
        let lv = self.cached_lv;
        let periodic_v = matches!(self.domain.velocity_bc, VelocityBoundType::Truncated);
        let scheme = self.advection_scheme;
        let positivity = self.positivity_limiter;

        let n_vel = nv1 * nv2 * nv3;
        let max_nv = nv1.max(nv2).max(nv3);
        let n_sp = nx1 * nx2 * nx3;

        // Reuse persistent buffers: swap data ↔ scratch to eliminate allocations.
        std::mem::swap(&mut self.data, &mut self.scratch);
        let src = std::mem::take(&mut self.scratch);
        let mut result = std::mem::take(&mut self.data);

        let counter = AtomicU64::new(0);
        let report_interval = (n_sp / 100).max(1) as u64;

        result
            .par_chunks_mut(n_vel)
            .enumerate()
            .for_each(|(si, local)| {
                let ax = acceleration.gx[si];
                let ay = acceleration.gy[si];
                let az = acceleration.gz[si];
                let disp = [ax * dt, ay * dt, az * dt];

                // Copy velocity slice (contiguous in memory)
                let base = si * n_vel;
                local.copy_from_slice(&src[base..base + n_vel]);

                // Thread-local scratch for 1D shifts (zero contention, no atomics).
                SHIFT_SCRATCH.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let (ref mut line, ref mut shifted) = *guard;
                    if line.len() < max_nv {
                        line.resize(max_nv, 0.0);
                        shifted.resize(max_nv, 0.0);
                    }

                    // Shift along v1
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            for iv1 in 0..nv1 {
                                line[iv1] = local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3];
                            }
                            shift_1d_dispatch(
                                scheme,
                                &line[..nv1],
                                disp[0],
                                dv[0],
                                nv1,
                                lv[0],
                                periodic_v,
                                shifted,
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
                            shift_1d_dispatch(
                                scheme,
                                &line[..nv2],
                                disp[1],
                                dv[1],
                                nv2,
                                lv[1],
                                periodic_v,
                                shifted,
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
                            shift_1d_dispatch(
                                scheme,
                                &line[..nv3],
                                disp[2],
                                dv[2],
                                nv3,
                                lv[2],
                                periodic_v,
                                shifted,
                            );
                            for iv3 in 0..nv3 {
                                local[iv1 * nv2 * nv3 + iv2 * nv3 + iv3] = shifted[iv3];
                            }
                        }
                    }
                });

                if positivity {
                    zhang_shu_limiter(local, 0.0);
                }

                if let Some(ref p) = progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, n_sp as u64);
                    }
                }
            });

        self.data = result;
        self.scratch = src;
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
                let vc1: Vec<f64> = (0..nv1)
                    .map(|i| -lv[0] + (i as f64 + 0.5) * dv[0])
                    .collect();
                let vc2: Vec<f64> = (0..nv2)
                    .map(|i| -lv[1] + (i as f64 + 0.5) * dv[1])
                    .collect();
                let vc3: Vec<f64> = (0..nv3)
                    .map(|i| -lv[2] + (i as f64 + 0.5) * dv[2])
                    .collect();
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
                let vc1: Vec<f64> = (0..nv1)
                    .map(|i| -lv[0] + (i as f64 + 0.5) * dv[0])
                    .collect();
                let vc2: Vec<f64> = (0..nv2)
                    .map(|i| -lv[1] + (i as f64 + 0.5) * dv[1])
                    .collect();
                let vc3: Vec<f64> = (0..nv3)
                    .map(|i| -lv[2] + (i as f64 + 0.5) * dv[2])
                    .collect();
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

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
    }
}
