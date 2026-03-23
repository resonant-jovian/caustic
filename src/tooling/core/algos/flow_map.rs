//! Flow-map Lagrangian representation.
//! Stores forward maps X(t;q) and V(t;q) on a Lagrangian grid.
//! Reconstructs f via Liouville's theorem: f(x,v,t) = f₀(q,p) where (q,p) are pre-images.
//!
//! Unlike `SheetTracker` which models cold (delta-function) distributions, `FlowMapRepr`
//! samples f₀(q,p) at each Lagrangian point and carries a finite mass per tracer.
//! The distribution function remains smooth even when f develops fine filaments,
//! because the map coordinates X(t;q) and V(t;q) stay smooth.

use rayon::prelude::*;

use super::super::{
    init::domain::{Domain, SpatialBoundType},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Flow-map Lagrangian representation of the 6D distribution function.
///
/// Stores `n_lag³ * nv_lag³` Lagrangian tracer points, each carrying:
/// - Initial position q and velocity p (from IC)
/// - Current position X(t;q,p) and velocity V(t;q,p)
/// - Initial distribution value f₀(q,p) and particle mass
///
/// Density is recovered via CIC deposition, following the same
/// pattern as `SheetTracker`.
pub struct FlowMapRepr {
    /// Lagrangian grid positions X(t;q) — flat [x0,y0,z0, x1,y1,z1, ...]
    pub positions: Vec<f64>,
    /// Lagrangian grid velocities V(t;q) — flat [vx0,vy0,vz0, ...]
    pub velocities: Vec<f64>,
    /// Initial distribution function value f₀(q,p) at each Lagrangian point
    pub f0_values: Vec<f64>,
    /// Mass of each Lagrangian tracer (= f₀ * dq³ * dp³)
    pub masses: Vec<f64>,
    /// Number of Lagrangian points per spatial dimension
    pub n_lag: usize,
    /// Number of Lagrangian points per velocity dimension
    pub nv_lag: usize,
    /// Spatial grid dimensions for density deposition [nx, ny, nz]
    pub spatial_shape: [usize; 3],
    /// Domain specification
    pub domain: Domain,
    /// Cached total mass (constant by Liouville)
    total_mass_cached: f64,
    /// Cached entropy (constant by Liouville)
    entropy_cached: f64,
    // Cached domain values
    cached_dx: [f64; 3],
    cached_lx: [f64; 3],
    cached_lv: [f64; 3],
    cached_is_periodic: bool,
    /// Optional progress reporter
    progress: Option<Arc<super::super::progress::StepProgress>>,
}

impl FlowMapRepr {
    /// Create a new FlowMapRepr with all tracers at their initial positions and zero f₀.
    ///
    /// The Lagrangian grid spans the full domain: `n_lag³` points in spatial dimensions,
    /// `nv_lag³` points in velocity dimensions. Each dimension is uniformly spaced.
    pub fn new(domain: &Domain, n_lag: usize, nv_lag: usize) -> Self {
        let dx = domain.dx();
        let lx = domain.lx();
        let lv = domain.lv();
        let dv = domain.dv();

        let spatial_shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];

        let n_total = n_lag * n_lag * n_lag * nv_lag * nv_lag * nv_lag;
        let dq = [
            2.0 * lx[0] / n_lag as f64,
            2.0 * lx[1] / n_lag as f64,
            2.0 * lx[2] / n_lag as f64,
        ];
        let dp = [
            2.0 * lv[0] / nv_lag as f64,
            2.0 * lv[1] / nv_lag as f64,
            2.0 * lv[2] / nv_lag as f64,
        ];

        let mut positions = Vec::with_capacity(n_total * 3);
        let mut velocities = Vec::with_capacity(n_total * 3);

        for ix in 0..n_lag {
            for iy in 0..n_lag {
                for iz in 0..n_lag {
                    for iv0 in 0..nv_lag {
                        for iv1 in 0..nv_lag {
                            for iv2 in 0..nv_lag {
                                let x = -lx[0] + (ix as f64 + 0.5) * dq[0];
                                let y = -lx[1] + (iy as f64 + 0.5) * dq[1];
                                let z = -lx[2] + (iz as f64 + 0.5) * dq[2];
                                let vx = -lv[0] + (iv0 as f64 + 0.5) * dp[0];
                                let vy = -lv[1] + (iv1 as f64 + 0.5) * dp[1];
                                let vz = -lv[2] + (iv2 as f64 + 0.5) * dp[2];
                                positions.extend_from_slice(&[x, y, z]);
                                velocities.extend_from_slice(&[vx, vy, vz]);
                            }
                        }
                    }
                }
            }
        }

        let f0_values = vec![0.0; n_total];
        let masses = vec![0.0; n_total];

        let is_periodic = matches!(domain.spatial_bc, SpatialBoundType::Periodic);

        Self {
            positions,
            velocities,
            f0_values,
            masses,
            n_lag,
            nv_lag,
            spatial_shape,
            domain: domain.clone(),
            total_mass_cached: 0.0,
            entropy_cached: 0.0,
            cached_dx: dx,
            cached_lx: lx,
            cached_lv: lv,
            cached_is_periodic: is_periodic,
            progress: None,
        }
    }

    /// Initialize from a `PhaseSpaceSnapshot` by sampling f₀ at Lagrangian grid points.
    ///
    /// Places `n_lag³ * nv_lag³` tracers on a regular grid in (x,v) space.
    /// Records f₀ at each point via trilinear interpolation of the snapshot data,
    /// and computes particle masses as `m_i = f₀(q_i, p_i) * dq³ * dp³`.
    pub fn from_snapshot(
        snap: &PhaseSpaceSnapshot,
        domain: &Domain,
        n_lag: usize,
        nv_lag: usize,
    ) -> Self {
        let mut repr = Self::new(domain, n_lag, nv_lag);
        let n_total = repr.num_tracers();

        let lx = domain.lx();
        let lv = domain.lv();
        let dq = [
            2.0 * lx[0] / n_lag as f64,
            2.0 * lx[1] / n_lag as f64,
            2.0 * lx[2] / n_lag as f64,
        ];
        let dp = [
            2.0 * lv[0] / nv_lag as f64,
            2.0 * lv[1] / nv_lag as f64,
            2.0 * lv[2] / nv_lag as f64,
        ];
        let phase_vol = dq[0] * dq[1] * dq[2] * dp[0] * dp[1] * dp[2];

        // Snapshot grid parameters
        let [nx0, nx1, nx2, nv0, nv1, nv2] = snap.shape;
        let snap_dx = domain.dx();
        let snap_dv = domain.dv();

        let mut total_mass = 0.0;
        let mut entropy = 0.0;

        for i in 0..n_total {
            let pos = [
                repr.positions[3 * i],
                repr.positions[3 * i + 1],
                repr.positions[3 * i + 2],
            ];
            let vel = [
                repr.velocities[3 * i],
                repr.velocities[3 * i + 1],
                repr.velocities[3 * i + 2],
            ];

            // Interpolate f₀ from the snapshot at this (x, v) point
            let f0 = Self::interpolate_6d(
                &snap.data, snap.shape, &pos, &vel, &snap_dx, &snap_dv, &lx, &lv,
            );

            let f0 = f0.max(0.0);
            repr.f0_values[i] = f0;
            repr.masses[i] = f0 * phase_vol;
            total_mass += repr.masses[i];

            if f0 > 0.0 {
                entropy -= f0 * f0.ln() * phase_vol;
            }
        }

        repr.total_mass_cached = total_mass;
        repr.entropy_cached = entropy;
        repr
    }

    /// Total number of Lagrangian tracer points.
    #[inline]
    pub fn num_tracers(&self) -> usize {
        self.n_lag * self.n_lag * self.n_lag * self.nv_lag * self.nv_lag * self.nv_lag
    }

    /// Trilinear interpolation of a 6D field at (x, v).
    ///
    /// The field is stored as a flat array with shape [nx0, nx1, nx2, nv0, nv1, nv2]
    /// in row-major order.
    fn interpolate_6d(
        data: &[f64],
        shape: [usize; 6],
        pos: &[f64; 3],
        vel: &[f64; 3],
        dx: &[f64; 3],
        dv: &[f64; 3],
        lx: &[f64; 3],
        lv: &[f64; 3],
    ) -> f64 {
        let [nx0, nx1, nx2, nv0, nv1, nv2] = shape;
        let ns_x = [nx0, nx1, nx2];
        let ns_v = [nv0, nv1, nv2];

        // Compute spatial CIC indices and fractions
        let mut x_ci = [0isize; 3];
        let mut x_frac = [0.0f64; 3];
        for k in 0..3 {
            let s = (pos[k] + lx[k]) / dx[k] - 0.5;
            x_ci[k] = s.floor() as isize;
            x_frac[k] = s - x_ci[k] as f64;
        }

        // Compute velocity CIC indices and fractions
        let mut v_ci = [0isize; 3];
        let mut v_frac = [0.0f64; 3];
        for k in 0..3 {
            let s = (vel[k] + lv[k]) / dv[k] - 0.5;
            v_ci[k] = s.floor() as isize;
            v_frac[k] = s - v_ci[k] as f64;
        }

        // Strides for row-major 6D: [nx0, nx1, nx2, nv0, nv1, nv2]
        let sv3 = 1usize;
        let sv2 = nv2;
        let sv1 = nv1 * nv2;
        let sx3 = nv0 * sv1;
        let sx2 = nx2 * sx3;
        let sx1 = nx1 * sx2;

        let mut result = 0.0;

        for dix in 0..2isize {
            let wx0 = if dix == 0 { 1.0 - x_frac[0] } else { x_frac[0] };
            let ix0 = x_ci[0] + dix;
            if ix0 < 0 || ix0 >= ns_x[0] as isize {
                continue;
            }
            for diy in 0..2isize {
                let wx1 = if diy == 0 { 1.0 - x_frac[1] } else { x_frac[1] };
                let ix1 = x_ci[1] + diy;
                if ix1 < 0 || ix1 >= ns_x[1] as isize {
                    continue;
                }
                for diz in 0..2isize {
                    let wx2 = if diz == 0 { 1.0 - x_frac[2] } else { x_frac[2] };
                    let ix2 = x_ci[2] + diz;
                    if ix2 < 0 || ix2 >= ns_x[2] as isize {
                        continue;
                    }
                    let wx = wx0 * wx1 * wx2;

                    for div0 in 0..2isize {
                        let wv0 = if div0 == 0 {
                            1.0 - v_frac[0]
                        } else {
                            v_frac[0]
                        };
                        let iv0 = v_ci[0] + div0;
                        if iv0 < 0 || iv0 >= ns_v[0] as isize {
                            continue;
                        }
                        for div1 in 0..2isize {
                            let wv1 = if div1 == 0 {
                                1.0 - v_frac[1]
                            } else {
                                v_frac[1]
                            };
                            let iv1 = v_ci[1] + div1;
                            if iv1 < 0 || iv1 >= ns_v[1] as isize {
                                continue;
                            }
                            for div2 in 0..2isize {
                                let wv2 = if div2 == 0 {
                                    1.0 - v_frac[2]
                                } else {
                                    v_frac[2]
                                };
                                let iv2 = v_ci[2] + div2;
                                if iv2 < 0 || iv2 >= ns_v[2] as isize {
                                    continue;
                                }
                                let wv = wv0 * wv1 * wv2;
                                let flat = ix0 as usize * sx1
                                    + ix1 as usize * sx2
                                    + ix2 as usize * sx3
                                    + iv0 as usize * sv1
                                    + iv1 as usize * sv2
                                    + iv2 as usize * sv3;
                                result += wx * wv * data[flat];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Trilinear interpolation of a 3D vector field at an arbitrary position.
    ///
    /// Used by `advect_v` to obtain the acceleration at each particle's position.
    /// Same algorithm as `SheetTracker::interpolate_vec_field`.
    fn interpolate_vec_field(
        field_x: &[f64],
        field_y: &[f64],
        field_z: &[f64],
        shape: [usize; 3],
        pos: &[f64; 3],
        dx: &[f64; 3],
        lx: &[f64; 3],
        periodic: bool,
    ) -> [f64; 3] {
        let [nx, ny, nz] = shape;

        // Grid node at index i is at coordinate -L + (i + 0.5) * dx.
        // Find the nearest lower node and fractional offset.
        let mut ci = [0isize; 3];
        let mut frac = [0.0f64; 3];
        for k in 0..3 {
            let s = (pos[k] + lx[k]) / dx[k] - 0.5;
            ci[k] = s.floor() as isize;
            frac[k] = s - ci[k] as f64;
        }

        let mut result = [0.0f64; 3];

        for di in 0..2isize {
            let wx = if di == 0 { 1.0 - frac[0] } else { frac[0] };
            for dj in 0..2isize {
                let wy = if dj == 0 { 1.0 - frac[1] } else { frac[1] };
                for dk in 0..2isize {
                    let wz = if dk == 0 { 1.0 - frac[2] } else { frac[2] };
                    let w = wx * wy * wz;

                    let mut ii = ci[0] + di;
                    let mut jj = ci[1] + dj;
                    let mut kk = ci[2] + dk;

                    if periodic {
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

                    let flat = ii as usize * ny * nz + jj as usize * nz + kk as usize;
                    result[0] += w * field_x[flat];
                    result[1] += w * field_y[flat];
                    result[2] += w * field_z[flat];
                }
            }
        }

        result
    }

    /// Find the flat spatial cell index for a position, or None if outside domain.
    fn cell_index(&self, pos: &[f64; 3]) -> Option<usize> {
        let dx = self.cached_dx;
        let lx = self.cached_lx;
        let [nx, ny, nz] = self.spatial_shape;

        let mut ci = [0usize; 3];
        let ns = [nx, ny, nz];
        for k in 0..3 {
            let idx = ((pos[k] + lx[k]) / dx[k]).floor() as isize;
            if self.cached_is_periodic {
                ci[k] = idx.rem_euclid(ns[k] as isize) as usize;
            } else if idx < 0 || idx >= ns[k] as isize {
                return None;
            } else {
                ci[k] = idx as usize;
            }
        }

        Some(ci[0] * ny * nz + ci[1] * nz + ci[2])
    }

    /// Collect indices of all tracers that lie in the same spatial cell as `position`.
    fn tracers_in_cell(&self, position: &[f64; 3]) -> Vec<usize> {
        let target = match self.cell_index(position) {
            Some(c) => c,
            None => return Vec::new(),
        };

        let n = self.num_tracers();
        let mut result = Vec::new();
        for i in 0..n {
            let pos = [
                self.positions[3 * i],
                self.positions[3 * i + 1],
                self.positions[3 * i + 2],
            ];
            if let Some(c) = self.cell_index(&pos)
                && c == target
            {
                result.push(i);
            }
        }
        result
    }
}

impl PhaseSpaceRepr for FlowMapRepr {
    fn set_progress(&mut self, p: Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    /// CIC deposition of tracer masses onto the spatial grid.
    ///
    /// For each Lagrangian tracer, finds the 8 nearest spatial cells and distributes
    /// its mass using trilinear weights, then divides by cell volume to get density.
    /// Uses rayon parallelism with fold/reduce (same pattern as `SheetTracker`).
    fn compute_density(&self) -> DensityField {
        let [nx, ny, nz] = self.spatial_shape;
        let n_cells = nx * ny * nz;
        let dx = self.cached_dx;
        let lx = self.cached_lx;
        let cell_vol = dx[0] * dx[1] * dx[2];
        let is_periodic = self.cached_is_periodic;
        let n_tracers = self.num_tracers();

        let n_tracers_u64 = n_tracers as u64;
        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_tracers_u64);
        }

        // Build a slice of (position_slice, mass) pairs for parallel iteration
        let density_data: Vec<f64> = (0..n_tracers)
            .into_par_iter()
            .fold(
                || vec![0.0f64; n_cells],
                |mut local, i| {
                    let mass = self.masses[i];
                    if mass <= 0.0 {
                        return local;
                    }
                    let pos = [
                        self.positions[3 * i],
                        self.positions[3 * i + 1],
                        self.positions[3 * i + 2],
                    ];

                    let mut ci = [0isize; 3];
                    let mut frac = [0.0f64; 3];
                    for k in 0..3 {
                        let s = (pos[k] + lx[k]) / dx[k] - 0.5;
                        ci[k] = s.floor() as isize;
                        frac[k] = s - ci[k] as f64;
                    }
                    for di in 0..2isize {
                        let wx = if di == 0 { 1.0 - frac[0] } else { frac[0] };
                        for dj in 0..2isize {
                            let wy = if dj == 0 { 1.0 - frac[1] } else { frac[1] };
                            for dk in 0..2isize {
                                let wz = if dk == 0 { 1.0 - frac[2] } else { frac[2] };
                                let w = wx * wy * wz;
                                let mut ii = ci[0] + di;
                                let mut jj = ci[1] + dj;
                                let mut kk = ci[2] + dk;
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
                                let flat = ii as usize * ny * nz + jj as usize * nz + kk as usize;
                                local[flat] += mass * w;
                            }
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0f64; n_cells],
                |mut a, b| {
                    for i in 0..n_cells {
                        a[i] += b[i];
                    }
                    a
                },
            );

        // Convert mass per cell to mass density
        let mut density = density_data;
        for d in &mut density {
            *d /= cell_vol;
        }

        DensityField {
            data: density,
            shape: [nx, ny, nz],
        }
    }

    /// Drift sub-step: X[i] += V[i] * dt for each tracer. Exact — no interpolation.
    ///
    /// For periodic domains, positions are wrapped into [-L, L].
    fn advect_x(&mut self, _displacement: &DisplacementField, dt: f64) {
        let is_periodic = self.cached_is_periodic;
        let lx = self.cached_lx;
        let n_tracers = self.num_tracers();
        let progress = self.progress.clone();
        let n_tracers_u64 = n_tracers as u64;
        let counter = AtomicU64::new(0);
        let report_interval = (n_tracers_u64 / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_tracers_u64);
        }

        // Process positions and velocities as chunks of 3 (x,y,z)
        let positions = &mut self.positions;
        let velocities = &self.velocities;

        // Use par_chunks_mut for parallel update
        positions
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i, pos_chunk)| {
                let vel_base = 3 * i;
                for k in 0..3 {
                    pos_chunk[k] += velocities[vel_base + k] * dt;
                    if is_periodic {
                        let two_l = 2.0 * lx[k];
                        pos_chunk[k] = ((pos_chunk[k] + lx[k]).rem_euclid(two_l)) - lx[k];
                    }
                }
                if let Some(ref prog) = progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        prog.set_intra_progress(c, n_tracers_u64);
                    }
                }
            });
    }

    /// Kick sub-step: V[i] += g(X[i]) * dt for each tracer.
    ///
    /// The acceleration at each tracer's current position is obtained by trilinear
    /// interpolation of the acceleration field.
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        let dx = self.cached_dx;
        let lx = self.cached_lx;
        let is_periodic = self.cached_is_periodic;
        let n_tracers = self.num_tracers();
        let progress = self.progress.clone();
        let n_tracers_u64 = n_tracers as u64;
        let counter = AtomicU64::new(0);
        let report_interval = (n_tracers_u64 / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_tracers_u64);
        }

        let positions = &self.positions;
        let velocities = &mut self.velocities;

        velocities
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(i, vel_chunk)| {
                let pos = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
                let a = Self::interpolate_vec_field(
                    &acceleration.gx,
                    &acceleration.gy,
                    &acceleration.gz,
                    acceleration.shape,
                    &pos,
                    &dx,
                    &lx,
                    is_periodic,
                );
                for k in 0..3 {
                    vel_chunk[k] += a[k] * dt;
                }
                if let Some(ref prog) = progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        prog.set_intra_progress(c, n_tracers_u64);
                    }
                }
            });
    }

    /// Velocity moment at a given spatial position.
    ///
    /// Finds all tracers in the same spatial cell and computes the requested moment
    /// from their velocities and masses.
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let indices = self.tracers_in_cell(position);
        let dx = self.cached_dx;
        let cell_vol = dx[0] * dx[1] * dx[2];

        match order {
            0 => {
                // Zeroth moment: density = sum(masses) / cell_volume
                let rho: f64 = indices.iter().map(|&i| self.masses[i]).sum::<f64>() / cell_vol;
                Tensor {
                    data: vec![rho],
                    rank: 0,
                    shape: vec![],
                }
            }
            1 => {
                // First moment: mass-weighted mean velocity
                let mut mean_v = [0.0f64; 3];
                let total_mass: f64 = indices.iter().map(|&i| self.masses[i]).sum();
                if total_mass > 0.0 {
                    for &i in &indices {
                        let m = self.masses[i];
                        for k in 0..3 {
                            mean_v[k] += m * self.velocities[3 * i + k];
                        }
                    }
                    for k in 0..3 {
                        mean_v[k] /= total_mass;
                    }
                }
                Tensor {
                    data: mean_v.to_vec(),
                    rank: 1,
                    shape: vec![3],
                }
            }
            2 => {
                // Second moment: mass-weighted velocity dispersion tensor
                let mut mean_v = [0.0f64; 3];
                let mut tensor = [0.0f64; 9];
                let total_mass: f64 = indices.iter().map(|&i| self.masses[i]).sum();
                if total_mass > 0.0 {
                    for &i in &indices {
                        let m = self.masses[i];
                        for k in 0..3 {
                            mean_v[k] += m * self.velocities[3 * i + k];
                        }
                    }
                    for k in 0..3 {
                        mean_v[k] /= total_mass;
                    }

                    for &i in &indices {
                        let m = self.masses[i];
                        for a in 0..3 {
                            for b in 0..3 {
                                let dv_a = self.velocities[3 * i + a] - mean_v[a];
                                let dv_b = self.velocities[3 * i + b] - mean_v[b];
                                tensor[a * 3 + b] += m * dv_a * dv_b;
                            }
                        }
                    }
                    for val in &mut tensor {
                        *val /= total_mass;
                    }
                }
                Tensor {
                    data: tensor.to_vec(),
                    rank: 2,
                    shape: vec![3, 3],
                }
            }
            _ => {
                let dim = 3usize.pow(order as u32);
                Tensor {
                    data: vec![0.0; dim],
                    rank: order,
                    shape: vec![3; order],
                }
            }
        }
    }

    /// Total mass. Constant by Liouville's theorem — returns the cached value.
    fn total_mass(&self) -> f64 {
        self.total_mass_cached
    }

    /// Casimir invariant C₂ = integral of f² over phase space.
    ///
    /// Approximated by depositing onto the spatial grid and computing integral of rho^2.
    /// This is a spatial-only approximation; the true C₂ involves the 6D distribution.
    fn casimir_c2(&self) -> f64 {
        let density = self.compute_density();
        let dx = self.cached_dx;
        let cell_vol = dx[0] * dx[1] * dx[2];
        density.data.iter().map(|&rho| rho * rho).sum::<f64>() * cell_vol
    }

    /// Entropy S = -integral of f ln f over phase space.
    /// Constant by Liouville's theorem — returns the cached value.
    fn entropy(&self) -> f64 {
        self.entropy_cached
    }

    /// Number of distinct velocity streams at each spatial point.
    ///
    /// Counts the number of distinct Lagrangian tracers per spatial cell.
    fn stream_count(&self) -> StreamCountField {
        let [nx, ny, nz] = self.spatial_shape;
        let n_cells = nx * ny * nz;
        let n_tracers = self.num_tracers();
        let dx = self.cached_dx;
        let lx = self.cached_lx;
        let is_periodic = self.cached_is_periodic;

        let counts: Vec<u32> = (0..n_tracers)
            .into_par_iter()
            .fold(
                || vec![0u32; n_cells],
                |mut local, i| {
                    let mass = self.masses[i];
                    if mass <= 0.0 {
                        return local;
                    }
                    let pos = [
                        self.positions[3 * i],
                        self.positions[3 * i + 1],
                        self.positions[3 * i + 2],
                    ];
                    let mut skip = false;
                    let mut ci = [0usize; 3];
                    let ns = [nx, ny, nz];
                    for k in 0..3 {
                        let idx = ((pos[k] + lx[k]) / dx[k]).floor() as isize;
                        if is_periodic {
                            ci[k] = idx.rem_euclid(ns[k] as isize) as usize;
                        } else if idx < 0 || idx >= ns[k] as isize {
                            skip = true;
                            break;
                        } else {
                            ci[k] = idx as usize;
                        }
                    }
                    if !skip {
                        let flat = ci[0] * ny * nz + ci[1] * nz + ci[2];
                        local[flat] += 1;
                    }
                    local
                },
            )
            .reduce(
                || vec![0u32; n_cells],
                |mut a, b| {
                    for i in 0..n_cells {
                        a[i] += b[i];
                    }
                    a
                },
            );

        StreamCountField {
            data: counts,
            shape: [nx, ny, nz],
        }
    }

    /// Local velocity distribution at a given spatial position.
    ///
    /// Collects the speed |v| of all tracers in the same cell as the given position.
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let indices = self.tracers_in_cell(position);
        indices
            .iter()
            .map(|&i| {
                let vx = self.velocities[3 * i];
                let vy = self.velocities[3 * i + 1];
                let vz = self.velocities[3 * i + 2];
                (vx * vx + vy * vy + vz * vz).sqrt()
            })
            .collect()
    }

    /// Total kinetic energy T = sum of 0.5 * mass[i] * |V[i]|^2.
    fn total_kinetic_energy(&self) -> Option<f64> {
        let n = self.num_tracers();
        Some(
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let vx = self.velocities[3 * i];
                    let vy = self.velocities[3 * i + 1];
                    let vz = self.velocities[3 * i + 2];
                    0.5 * self.masses[i] * (vx * vx + vy * vy + vz * vz)
                })
                .sum(),
        )
    }

    /// Extract a full 6D snapshot by CIC deposition of all tracers.
    ///
    /// Each tracer is deposited onto 2^3 x 2^3 = 64 surrounding cells in the 6D grid.
    /// This is expensive for large grids and should only be used for checkpoints.
    fn to_snapshot(&self, time: f64) -> Option<PhaseSpaceSnapshot> {
        let d = &self.domain;
        let nx = [
            d.spatial_res.x1 as usize,
            d.spatial_res.x2 as usize,
            d.spatial_res.x3 as usize,
        ];
        let nv = [
            d.velocity_res.v1 as usize,
            d.velocity_res.v2 as usize,
            d.velocity_res.v3 as usize,
        ];
        let dx = d.dx();
        let dv = d.dv();
        let lx = d.lx();
        let lv = d.lv();

        let total_6d = nx[0] * nx[1] * nx[2] * nv[0] * nv[1] * nv[2];
        let mut data = vec![0.0f64; total_6d];

        let cell_vol_6d = dx[0] * dx[1] * dx[2] * dv[0] * dv[1] * dv[2];
        let is_periodic = self.cached_is_periodic;

        // Strides for row-major 6D: x1, x2, x3, v1, v2, v3
        let sv3 = 1;
        let sv2 = nv[2];
        let sv1 = nv[1] * nv[2];
        let sx3 = nv[0] * sv1;
        let sx2 = nx[2] * sx3;
        let sx1 = nx[1] * sx2;

        let n_tracers = self.num_tracers();

        for i in 0..n_tracers {
            let mass = self.masses[i];
            if mass <= 0.0 {
                continue;
            }
            let pos = [
                self.positions[3 * i],
                self.positions[3 * i + 1],
                self.positions[3 * i + 2],
            ];
            let vel = [
                self.velocities[3 * i],
                self.velocities[3 * i + 1],
                self.velocities[3 * i + 2],
            ];

            // Spatial CIC indices
            let mut x_ci = [0isize; 3];
            let mut x_frac = [0.0f64; 3];
            for k in 0..3 {
                let s = (pos[k] + lx[k]) / dx[k] - 0.5;
                x_ci[k] = s.floor() as isize;
                x_frac[k] = s - x_ci[k] as f64;
            }

            // Velocity CIC indices
            let mut v_ci = [0isize; 3];
            let mut v_frac = [0.0f64; 3];
            for k in 0..3 {
                let s = (vel[k] + lv[k]) / dv[k] - 0.5;
                v_ci[k] = s.floor() as isize;
                v_frac[k] = s - v_ci[k] as f64;
            }

            // Deposit to 2^3 x 2^3 = 64 surrounding 6D cells
            for dix in 0..2isize {
                let wx0 = if dix == 0 { 1.0 - x_frac[0] } else { x_frac[0] };
                let ix0 = x_ci[0] + dix;
                if is_periodic {
                    // handled below
                } else if ix0 < 0 || ix0 >= nx[0] as isize {
                    continue;
                }
                let ix0_w = if is_periodic {
                    ix0.rem_euclid(nx[0] as isize) as usize
                } else {
                    ix0 as usize
                };

                for diy in 0..2isize {
                    let wx1 = if diy == 0 { 1.0 - x_frac[1] } else { x_frac[1] };
                    let ix1 = x_ci[1] + diy;
                    if is_periodic {
                        // handled below
                    } else if ix1 < 0 || ix1 >= nx[1] as isize {
                        continue;
                    }
                    let ix1_w = if is_periodic {
                        ix1.rem_euclid(nx[1] as isize) as usize
                    } else {
                        ix1 as usize
                    };

                    for diz in 0..2isize {
                        let wx2 = if diz == 0 { 1.0 - x_frac[2] } else { x_frac[2] };
                        let ix2 = x_ci[2] + diz;
                        if is_periodic {
                            // handled below
                        } else if ix2 < 0 || ix2 >= nx[2] as isize {
                            continue;
                        }
                        let ix2_w = if is_periodic {
                            ix2.rem_euclid(nx[2] as isize) as usize
                        } else {
                            ix2 as usize
                        };
                        let wx = wx0 * wx1 * wx2;

                        for div0 in 0..2isize {
                            let wv0 = if div0 == 0 {
                                1.0 - v_frac[0]
                            } else {
                                v_frac[0]
                            };
                            let iv0 = v_ci[0] + div0;
                            if iv0 < 0 || iv0 >= nv[0] as isize {
                                continue;
                            }
                            for div1 in 0..2isize {
                                let wv1 = if div1 == 0 {
                                    1.0 - v_frac[1]
                                } else {
                                    v_frac[1]
                                };
                                let iv1 = v_ci[1] + div1;
                                if iv1 < 0 || iv1 >= nv[1] as isize {
                                    continue;
                                }
                                for div2 in 0..2isize {
                                    let wv2 = if div2 == 0 {
                                        1.0 - v_frac[2]
                                    } else {
                                        v_frac[2]
                                    };
                                    let iv2 = v_ci[2] + div2;
                                    if iv2 < 0 || iv2 >= nv[2] as isize {
                                        continue;
                                    }
                                    let wv = wv0 * wv1 * wv2;

                                    let flat = ix0_w * sx1
                                        + ix1_w * sx2
                                        + ix2_w * sx3
                                        + iv0 as usize * sv1
                                        + iv1 as usize * sv2
                                        + iv2 as usize * sv3;

                                    data[flat] += mass * wx * wv / cell_vol_6d;
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(PhaseSpaceSnapshot {
            data,
            shape: [nx[0], nx[1], nx[2], nv[0], nv[1], nv[2]],
            time,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn memory_bytes(&self) -> usize {
        let n = self.num_tracers();
        // positions (3*n f64) + velocities (3*n f64) + f0_values (n f64) + masses (n f64)
        (3 * n + 3 * n + n + n) * std::mem::size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};

    fn test_domain() -> Domain {
        Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    /// Create a FlowMapRepr with a Gaussian IC in both position and velocity.
    /// Returns the representation and expected total mass.
    fn gaussian_flow_map(domain: &Domain, n_lag: usize, nv_lag: usize) -> FlowMapRepr {
        let lx = domain.lx();
        let lv = domain.lv();
        let dq = [
            2.0 * lx[0] / n_lag as f64,
            2.0 * lx[1] / n_lag as f64,
            2.0 * lx[2] / n_lag as f64,
        ];
        let dp = [
            2.0 * lv[0] / nv_lag as f64,
            2.0 * lv[1] / nv_lag as f64,
            2.0 * lv[2] / nv_lag as f64,
        ];
        let phase_vol = dq[0] * dq[1] * dq[2] * dp[0] * dp[1] * dp[2];

        let mut repr = FlowMapRepr::new(domain, n_lag, nv_lag);
        let n = repr.num_tracers();
        let sigma_x = 0.3;
        let sigma_v = 0.3;

        let mut total_mass = 0.0;
        let mut entropy = 0.0;

        for i in 0..n {
            let x = repr.positions[3 * i];
            let y = repr.positions[3 * i + 1];
            let z = repr.positions[3 * i + 2];
            let vx = repr.velocities[3 * i];
            let vy = repr.velocities[3 * i + 1];
            let vz = repr.velocities[3 * i + 2];

            let r2 = x * x + y * y + z * z;
            let v2 = vx * vx + vy * vy + vz * vz;
            let f0 =
                (-r2 / (2.0 * sigma_x * sigma_x)).exp() * (-v2 / (2.0 * sigma_v * sigma_v)).exp();

            repr.f0_values[i] = f0;
            repr.masses[i] = f0 * phase_vol;
            total_mass += repr.masses[i];
            if f0 > 0.0 {
                entropy -= f0 * f0.ln() * phase_vol;
            }
        }

        repr.total_mass_cached = total_mass;
        repr.entropy_cached = entropy;
        repr
    }

    #[test]
    fn test_flow_map_free_streaming() {
        let domain = test_domain();
        let n_lag = 4;
        let nv_lag = 4;
        let mut repr = gaussian_flow_map(&domain, n_lag, nv_lag);
        let n = repr.num_tracers();

        // Record initial positions
        let x0: Vec<f64> = repr.positions.clone();
        let v0: Vec<f64> = repr.velocities.clone();

        let dt = 0.1;
        let dummy_disp = DisplacementField {
            dx: vec![0.0; 8 * 8 * 8],
            dy: vec![0.0; 8 * 8 * 8],
            dz: vec![0.0; 8 * 8 * 8],
            shape: [8, 8, 8],
        };

        repr.advect_x(&dummy_disp, dt);

        let lx = domain.lx();
        for i in 0..n {
            for k in 0..3 {
                let expected = x0[3 * i + k] + v0[3 * i + k] * dt;
                // With periodic wrapping
                let two_l = 2.0 * lx[k];
                let wrapped = ((expected + lx[k]).rem_euclid(two_l)) - lx[k];
                assert!(
                    (repr.positions[3 * i + k] - wrapped).abs() < 1e-12,
                    "tracer {i} dim {k}: expected {wrapped}, got {}",
                    repr.positions[3 * i + k]
                );
            }
        }
    }

    #[test]
    fn test_flow_map_mass_conservation() {
        let domain = test_domain();
        let mut repr = gaussian_flow_map(&domain, 4, 4);

        let mass_before = repr.total_mass();
        assert!(mass_before > 0.0, "initial mass must be positive");

        // Advect in x
        let dummy_disp = DisplacementField {
            dx: vec![0.0; 8 * 8 * 8],
            dy: vec![0.0; 8 * 8 * 8],
            dz: vec![0.0; 8 * 8 * 8],
            shape: [8, 8, 8],
        };
        repr.advect_x(&dummy_disp, 0.1);

        let mass_after = repr.total_mass();
        assert!(
            (mass_after - mass_before).abs() < 1e-14,
            "mass must be conserved: before={mass_before}, after={mass_after}"
        );

        // Advect in v with a uniform acceleration field
        let n_cells = 8 * 8 * 8;
        let acc = AccelerationField {
            gx: vec![0.1; n_cells],
            gy: vec![0.0; n_cells],
            gz: vec![0.0; n_cells],
            shape: [8, 8, 8],
        };
        repr.advect_v(&acc, 0.05);

        let mass_after_kick = repr.total_mass();
        assert!(
            (mass_after_kick - mass_before).abs() < 1e-14,
            "mass must be conserved after kick: before={mass_before}, after={mass_after_kick}"
        );
    }

    #[test]
    fn test_flow_map_density_recovery() {
        let domain = test_domain();
        let repr = gaussian_flow_map(&domain, 6, 6);

        let density = repr.compute_density();
        let dx = domain.dx();
        let cell_vol = dx[0] * dx[1] * dx[2];

        // Total mass from density field should match total_mass()
        let mass_from_density: f64 = density.data.iter().sum::<f64>() * cell_vol;
        let mass_from_repr = repr.total_mass();

        assert!(
            (mass_from_density - mass_from_repr).abs() / mass_from_repr.max(1e-15) < 0.05,
            "density-integrated mass ({mass_from_density}) should match total_mass ({mass_from_repr})"
        );

        // Density should be positive or zero everywhere
        for (i, &rho) in density.data.iter().enumerate() {
            assert!(
                rho >= 0.0,
                "density must be non-negative at cell {i}, got {rho}"
            );
        }

        // Peak density should be near the center (Gaussian)
        let [nx, ny, nz] = density.shape;
        let center = (nx / 2) * ny * nz + (ny / 2) * nz + nz / 2;
        let center_rho = density.data[center];
        assert!(
            center_rho > 0.0,
            "center density should be positive for a Gaussian IC"
        );
    }

    #[test]
    fn test_flow_map_kinetic_energy() {
        let domain = test_domain();
        let repr = gaussian_flow_map(&domain, 4, 4);
        let n = repr.num_tracers();

        // Compute expected KE manually
        let mut expected_ke = 0.0;
        for i in 0..n {
            let vx = repr.velocities[3 * i];
            let vy = repr.velocities[3 * i + 1];
            let vz = repr.velocities[3 * i + 2];
            expected_ke += 0.5 * repr.masses[i] * (vx * vx + vy * vy + vz * vz);
        }

        let ke = repr.total_kinetic_energy().unwrap();
        assert!(
            (ke - expected_ke).abs() < 1e-14,
            "kinetic energy mismatch: expected {expected_ke}, got {ke}"
        );
        assert!(ke >= 0.0, "kinetic energy must be non-negative");
    }
}
