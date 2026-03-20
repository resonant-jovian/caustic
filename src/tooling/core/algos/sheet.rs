//! Lagrangian sheet tracker for cold initial conditions. Memory O(N³). Exact in
//! single-stream regions.
//!
//! The dark matter sheet is a 3D manifold embedded in 6D phase space. Each
//! Lagrangian tracer particle carries its initial coordinate q, its current
//! Eulerian position x(q,t), and velocity v(q,t). The distribution function is
//! implicitly f(x,v,t) = Σ_q m_q δ³(x − x_q(t)) δ³(v − v_q(t)).
//!
//! Density is recovered by Cloud-in-Cell (CIC) deposition of particle masses
//! onto the spatial grid. Stream count at each cell equals the number of
//! distinct Lagrangian elements mapping to that cell.

use rayon::prelude::*;

use super::super::{
    init::{cosmological::ZeldovichIC, domain::Domain},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// One Lagrangian tracer particle on the dark matter sheet.
pub struct SheetParticle {
    /// Lagrangian coordinate (initial position, never changes).
    pub q: [f64; 3],
    /// Current Eulerian position.
    pub x: [f64; 3],
    /// Current velocity.
    pub v: [f64; 3],
}

/// Lagrangian cold dark matter sheet tracker.
///
/// Stores N³ particles on a Lagrangian grid. Each particle tracks its
/// initial coordinate q, current position x, and velocity v. The distribution
/// function is a sum of delta functions — a cold (zero-entropy) system.
pub struct SheetTracker {
    pub particles: Vec<SheetParticle>,
    pub shape: [usize; 3],
    pub domain: Domain,
    pub stream_threshold: f64,
    pub particle_mass: f64,
    // Cached domain values
    cached_dx: [f64; 3],
    cached_lx: [f64; 3],
    cached_is_periodic: bool,
    progress: Option<Arc<super::super::progress::StepProgress>>,
}

impl SheetTracker {
    /// Create a new sheet tracker with N³ particles at spatial cell centers, v = 0.
    ///
    /// Domain spans [−L_k, L_k] in each dimension. Particles are placed at
    /// x_k = −L_k + (i_k + 0.5) * dx_k, with q = x (identity mapping).
    pub fn new(domain: Domain) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];
        let dx = domain.dx();
        let lx = domain.lx();

        let n_total = shape[0] * shape[1] * shape[2];
        let particle_mass = 1.0 / n_total as f64;

        let mut particles = Vec::with_capacity(n_total);
        for i0 in 0..shape[0] {
            for i1 in 0..shape[1] {
                for i2 in 0..shape[2] {
                    let x = [
                        -lx[0] + (i0 as f64 + 0.5) * dx[0],
                        -lx[1] + (i1 as f64 + 0.5) * dx[1],
                        -lx[2] + (i2 as f64 + 0.5) * dx[2],
                    ];
                    particles.push(SheetParticle {
                        q: x,
                        x,
                        v: [0.0; 3],
                    });
                }
            }
        }

        let is_periodic = matches!(
            domain.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );
        Self {
            particles,
            shape,
            domain,
            stream_threshold: 1.0,
            particle_mass,
            cached_dx: dx,
            cached_lx: lx,
            cached_is_periodic: is_periodic,
            progress: None,
        }
    }

    /// Place one particle at each Lagrangian grid point, displaced by s(q).
    ///
    /// Uses `ZeldovichIC::displacement_field()` and `velocity_field()` to set
    /// x = q + s(q), v = v₀(q).
    pub fn from_zeldovich(ic: &ZeldovichIC, domain: &Domain) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];
        let dx = domain.dx();
        let lx = domain.lx();

        let [sx, sy, sz] = ic.displacement_field(domain);
        let [vx, vy, vz] = ic.velocity_field(domain);

        let n_total = shape[0] * shape[1] * shape[2];
        let particle_mass = 1.0 / n_total as f64;

        let mut particles = Vec::with_capacity(n_total);
        for i0 in 0..shape[0] {
            for i1 in 0..shape[1] {
                for i2 in 0..shape[2] {
                    let idx = i0 * shape[1] * shape[2] + i1 * shape[2] + i2;
                    let q = [
                        -lx[0] + (i0 as f64 + 0.5) * dx[0],
                        -lx[1] + (i1 as f64 + 0.5) * dx[1],
                        -lx[2] + (i2 as f64 + 0.5) * dx[2],
                    ];
                    particles.push(SheetParticle {
                        q,
                        x: [q[0] + sx[idx], q[1] + sy[idx], q[2] + sz[idx]],
                        v: [vx[idx], vy[idx], vz[idx]],
                    });
                }
            }
        }

        let is_periodic = matches!(
            domain.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );
        Self {
            particles,
            shape,
            domain: domain.clone(),
            stream_threshold: 1.0,
            particle_mass,
            cached_dx: dx,
            cached_lx: lx,
            cached_is_periodic: is_periodic,
            progress: None,
        }
    }

    /// Detect stream crossings by counting the number of particles per spatial cell.
    ///
    /// In a single-stream region each cell contains at most one particle.
    /// When the sheet folds (caustic formation), multiple Lagrangian elements
    /// map to the same Eulerian cell, so the count exceeds 1.
    pub fn detect_caustics(&self) -> StreamCountField {
        let [nx, ny, nz] = self.shape;
        let n_cells = nx * ny * nz;
        let mut counts = vec![0u32; n_cells];

        let dx = self.cached_dx;
        let lx = self.cached_lx;
        let is_periodic = self.cached_is_periodic;

        let n_particles = self.particles.len() as u64;
        let report_interval = (n_particles / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_particles);
        }

        for (pi, p) in self.particles.iter().enumerate() {
            if let Some(ref prog) = self.progress
                && (pi as u64).is_multiple_of(report_interval) {
                    prog.set_intra_progress(pi as u64, n_particles);
                }
            let mut skip = false;
            let mut ci = [0usize; 3];
            for k in 0..3 {
                let idx = ((p.x[k] + lx[k]) / dx[k]).floor() as isize;
                if is_periodic {
                    ci[k] = idx.rem_euclid(self.shape[k] as isize) as usize;
                } else if idx < 0 || idx >= self.shape[k] as isize {
                    skip = true;
                    break;
                } else {
                    ci[k] = idx as usize;
                }
            }
            if !skip {
                let flat = ci[0] * ny * nz + ci[1] * nz + ci[2];
                counts[flat] += 1;
            }
        }

        StreamCountField {
            data: counts,
            shape: [nx, ny, nz],
        }
    }

    /// Cloud-in-Cell (CIC) density deposition from particle positions.
    ///
    /// Each particle of mass `particle_mass` at position x is distributed
    /// to the 8 surrounding grid nodes using trilinear weights. The result
    /// is divided by cell volume to give density ρ(x).
    pub fn interpolate_density(&self, domain: &Domain) -> DensityField {
        let [nx, ny, nz] = self.shape;
        let n_cells = nx * ny * nz;
        let dx = domain.dx();
        let lx = domain.lx();
        let cell_vol = dx[0] * dx[1] * dx[2];

        let is_periodic = matches!(
            domain.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );

        let mut density = vec![0.0f64; n_cells];

        let n_particles = self.particles.len() as u64;
        let report_interval = (n_particles / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_particles);
        }

        for (pi, p) in self.particles.iter().enumerate() {
            if let Some(ref prog) = self.progress
                && (pi as u64).is_multiple_of(report_interval) {
                    prog.set_intra_progress(pi as u64, n_particles);
                }
            // Find the cell index and fractional position for CIC.
            // The grid node at index i is at x = -L + (i + 0.5) * dx.
            // For CIC, we find the nearest lower grid node:
            //   cell_idx[k] = floor((x[k] + L[k]) / dx[k] - 0.5)
            //   frac[k]     = (x[k] + L[k]) / dx[k] - 0.5 - cell_idx[k]
            let mut ci = [0isize; 3];
            let mut frac = [0.0f64; 3];

            for k in 0..3 {
                let s = (p.x[k] + lx[k]) / dx[k] - 0.5;
                ci[k] = s.floor() as isize;
                frac[k] = s - ci[k] as f64;
            }

            // Deposit to 2x2x2 neighboring cells with trilinear weights.
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
                        } else {
                            // Open / Isolated / Reflecting: skip deposits outside domain.
                            if ii < 0
                                || ii >= nx as isize
                                || jj < 0
                                || jj >= ny as isize
                                || kk < 0
                                || kk >= nz as isize
                            {
                                continue;
                            }
                        }

                        let flat = ii as usize * ny * nz + jj as usize * nz + kk as usize;
                        density[flat] += self.particle_mass * w;
                    }
                }
            }
        }

        // Convert from mass per cell to mass density (divide by cell volume).
        for d in &mut density {
            *d /= cell_vol;
        }

        DensityField {
            data: density,
            shape: [nx, ny, nz],
        }
    }

    /// Trilinear interpolation of a 3D vector field at an arbitrary position.
    ///
    /// Used by `advect_v` to obtain the acceleration at each particle's position.
    #[allow(clippy::too_many_arguments)]
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
        let [nx, ny, nz] = self.shape;
        let is_periodic = self.cached_is_periodic;

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

    /// Collect indices of all particles that lie in the same cell as `position`.
    fn particles_in_cell(&self, position: &[f64; 3]) -> Vec<usize> {
        let target = match self.cell_index(position) {
            Some(c) => c,
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        for (i, p) in self.particles.iter().enumerate() {
            if let Some(c) = self.cell_index(&p.x)
                && c == target
            {
                result.push(i);
            }
        }
        result
    }
}

impl PhaseSpaceRepr for SheetTracker {
    fn set_progress(&mut self, p: std::sync::Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    fn compute_density(&self) -> DensityField {
        self.interpolate_density(&self.domain)
    }

    fn advect_x(&mut self, _displacement: &DisplacementField, dt: f64) {
        let is_periodic = self.cached_is_periodic;
        let lx = self.cached_lx;
        let progress = self.progress.clone();
        let n_particles = self.particles.len() as u64;
        let counter = AtomicU64::new(0);
        let report_interval = (n_particles / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_particles);
        }

        self.particles.par_iter_mut().for_each(|p| {
            for (k, &l) in lx.iter().enumerate() {
                p.x[k] += p.v[k] * dt;
                if is_periodic {
                    let two_l = 2.0 * l;
                    p.x[k] = ((p.x[k] + l).rem_euclid(two_l)) - l;
                }
            }
            if let Some(ref prog) = progress {
                let c = counter.fetch_add(1, Ordering::Relaxed);
                if c.is_multiple_of(report_interval) {
                    prog.set_intra_progress(c, n_particles);
                }
            }
        });
    }

    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        let dx = self.cached_dx;
        let lx = self.cached_lx;
        let is_periodic = self.cached_is_periodic;
        let progress = self.progress.clone();
        let n_particles = self.particles.len() as u64;
        let counter = AtomicU64::new(0);
        let report_interval = (n_particles / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_particles);
        }

        self.particles.par_iter_mut().for_each(|p| {
            let a = Self::interpolate_vec_field(
                &acceleration.gx,
                &acceleration.gy,
                &acceleration.gz,
                acceleration.shape,
                &p.x,
                &dx,
                &lx,
                is_periodic,
            );
            for (v, &acc) in p.v.iter_mut().zip(a.iter()) {
                *v += acc * dt;
            }
            if let Some(ref prog) = progress {
                let c = counter.fetch_add(1, Ordering::Relaxed);
                if c.is_multiple_of(report_interval) {
                    prog.set_intra_progress(c, n_particles);
                }
            }
        });
    }

    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let indices = self.particles_in_cell(position);
        let dx = self.cached_dx;
        let cell_vol = dx[0] * dx[1] * dx[2];

        match order {
            0 => {
                // Zeroth moment: density = (count * particle_mass) / cell_volume
                let rho = indices.len() as f64 * self.particle_mass / cell_vol;
                Tensor {
                    data: vec![rho],
                    rank: 0,
                    shape: vec![],
                }
            }
            1 => {
                // First moment: mass-weighted mean velocity
                let mut mean_v = [0.0f64; 3];
                if !indices.is_empty() {
                    for &i in &indices {
                        for (mv, &pv) in mean_v.iter_mut().zip(self.particles[i].v.iter()) {
                            *mv += pv;
                        }
                    }
                    let n = indices.len() as f64;
                    for mv in &mut mean_v {
                        *mv /= n;
                    }
                }
                Tensor {
                    data: mean_v.to_vec(),
                    rank: 1,
                    shape: vec![3],
                }
            }
            2 => {
                // Second moment: mass-weighted velocity dispersion tensor σ²_{ij}
                // σ²_{ij} = <v_i v_j> − <v_i><v_j>
                let mut mean_v = [0.0f64; 3];
                let mut tensor = [0.0f64; 9]; // 3x3 flat

                if !indices.is_empty() {
                    let n = indices.len() as f64;
                    for &i in &indices {
                        for (mv, &pv) in mean_v.iter_mut().zip(self.particles[i].v.iter()) {
                            *mv += pv;
                        }
                    }
                    for mv in &mut mean_v {
                        *mv /= n;
                    }

                    for &i in &indices {
                        let vp = &self.particles[i].v;
                        for a in 0..3 {
                            for b in 0..3 {
                                tensor[a * 3 + b] += (vp[a] - mean_v[a]) * (vp[b] - mean_v[b]);
                            }
                        }
                    }
                    for val in &mut tensor {
                        *val /= n;
                    }
                }
                Tensor {
                    data: tensor.to_vec(),
                    rank: 2,
                    shape: vec![3, 3],
                }
            }
            _ => {
                // Higher-order moments: return zero tensor of appropriate shape.
                let dim = 3usize.pow(order as u32);
                Tensor {
                    data: vec![0.0; dim],
                    rank: order,
                    shape: vec![3; order],
                }
            }
        }
    }

    fn total_mass(&self) -> f64 {
        self.particle_mass * self.particles.len() as f64
    }

    fn casimir_c2(&self) -> f64 {
        // For a cold (delta-function) distribution f = Σ m_q δ³(x−x_q) δ³(v−v_q),
        // C₂ = ∫ f² dx³dv³ diverges (product of delta functions).
        f64::INFINITY
    }

    fn entropy(&self) -> f64 {
        // Cold distribution: S = −∫ f ln f = 0 (zero entropy).
        // The sheet is a perfectly cold system with no velocity dispersion.
        0.0
    }

    fn total_kinetic_energy(&self) -> f64 {
        let ke: f64 = self
            .particles
            .par_iter()
            .map(|p| p.v[0] * p.v[0] + p.v[1] * p.v[1] + p.v[2] * p.v[2])
            .sum();
        0.5 * self.particle_mass * ke
    }

    fn stream_count(&self) -> StreamCountField {
        self.detect_caustics()
    }

    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let indices = self.particles_in_cell(position);
        indices
            .iter()
            .map(|&i| {
                let v = &self.particles[i].v;
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
            })
            .collect()
    }

    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot {
        // Approximate: CIC deposit each particle onto the full 6D grid.
        // Each particle is a delta in both x and v; we smear it across the
        // 8 nearest x-cells and 8 nearest v-cells (64 contributions total).
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
        let is_periodic = matches!(
            d.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );

        // Strides for row-major 6D: x1, x2, x3, v1, v2, v3
        let sv3 = 1;
        let sv2 = nv[2];
        let sv1 = nv[1] * nv[2];
        let sx3 = nv[0] * sv1;
        let sx2 = nx[2] * sx3;
        let sx1 = nx[1] * sx2;

        let n_particles = self.particles.len() as u64;
        let report_interval = (n_particles / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, n_particles);
        }

        for (pi, p) in self.particles.iter().enumerate() {
            if let Some(ref prog) = self.progress
                && (pi as u64).is_multiple_of(report_interval) {
                    prog.set_intra_progress(pi as u64, n_particles);
                }

            // Spatial CIC indices
            let mut x_ci = [0isize; 3];
            let mut x_frac = [0.0f64; 3];
            for k in 0..3 {
                let s = (p.x[k] + lx[k]) / dx[k] - 0.5;
                x_ci[k] = s.floor() as isize;
                x_frac[k] = s - x_ci[k] as f64;
            }

            // Velocity CIC indices
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

                        let mut ix0 = x_ci[0] + dix;
                        let mut ix1 = x_ci[1] + diy;
                        let mut ix2 = x_ci[2] + diz;

                        if is_periodic {
                            ix0 = ix0.rem_euclid(nx[0] as isize);
                            ix1 = ix1.rem_euclid(nx[1] as isize);
                            ix2 = ix2.rem_euclid(nx[2] as isize);
                        } else if ix0 < 0
                            || ix0 >= nx[0] as isize
                            || ix1 < 0
                            || ix1 >= nx[1] as isize
                            || ix2 < 0
                            || ix2 >= nx[2] as isize
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

                                    let iv0 = v_ci[0] + div1;
                                    let iv1 = v_ci[1] + div2;
                                    let iv2 = v_ci[2] + div3;

                                    // Velocity: always clamp/skip for out-of-bounds
                                    if iv0 < 0
                                        || iv0 >= nv[0] as isize
                                        || iv1 < 0
                                        || iv1 >= nv[1] as isize
                                        || iv2 < 0
                                        || iv2 >= nv[2] as isize
                                    {
                                        continue;
                                    }

                                    let flat = ix0 as usize * sx1
                                        + ix1 as usize * sx2
                                        + ix2 as usize * sx3
                                        + iv0 as usize * sv1
                                        + iv1 as usize * sv2
                                        + iv2 as usize * sv3;

                                    data[flat] += self.particle_mass * wx * wv / cell_vol_6d;
                                }
                            }
                        }
                    }
                }
            }
        }

        PhaseSpaceSnapshot {
            data,
            shape: [nx[0], nx[1], nx[2], nv[0], nv[1], nv[2]],
            time,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
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

    #[test]
    fn sheet_free_streaming() {
        let domain = test_domain();
        let mut sheet = SheetTracker::new(domain.clone());
        // Give all particles uniform velocity
        for p in &mut sheet.particles {
            p.v = [0.1, 0.0, 0.0];
        }
        let dt = 0.5;
        let dummy_disp = DisplacementField {
            dx: vec![0.0; 8 * 8 * 8],
            dy: vec![0.0; 8 * 8 * 8],
            dz: vec![0.0; 8 * 8 * 8],
            shape: [8, 8, 8],
        };
        let x0: Vec<[f64; 3]> = sheet.particles.iter().map(|p| p.x).collect();
        sheet.advect_x(&dummy_disp, dt);
        let lx = 1.0;
        for (i, p) in sheet.particles.iter().enumerate() {
            let expected = x0[i][0] + 0.1 * dt;
            // With periodic wrapping, check modular
            let wrapped = ((expected + lx) % (2.0 * lx)) - lx;
            assert!(
                (p.x[0] - wrapped).abs() < 1e-10 || (p.x[0] - expected).abs() < 1e-10,
                "particle {i}: expected ~{expected}, got {}",
                p.x[0]
            );
        }
    }

    #[test]
    fn sheet_cic_density() {
        let domain = test_domain();
        let mut sheet = SheetTracker::new(domain.clone());
        // Move all particles to center cell
        for p in &mut sheet.particles {
            p.x = [0.0, 0.0, 0.0];
        }
        let density = sheet.interpolate_density(&domain);
        let dx = domain.dx();
        let total: f64 = density.data.iter().sum::<f64>() * dx[0] * dx[1] * dx[2];
        // Total mass should equal particle_mass * n_particles
        let expected_mass = sheet.particle_mass * sheet.particles.len() as f64;
        assert!(
            (total - expected_mass).abs() / expected_mass < 0.01,
            "CIC total mass {total} != expected {expected_mass}"
        );
    }

    #[test]
    fn sheet_mass_conservation() {
        let domain = test_domain();
        let mut sheet = SheetTracker::new(domain.clone());
        for p in &mut sheet.particles {
            p.v = [0.3, -0.1, 0.2];
        }
        let m0 = sheet.total_mass();
        let dummy = DisplacementField {
            dx: vec![0.0; 8 * 8 * 8],
            dy: vec![0.0; 8 * 8 * 8],
            dz: vec![0.0; 8 * 8 * 8],
            shape: [8, 8, 8],
        };
        for _ in 0..10 {
            sheet.advect_x(&dummy, 0.01);
        }
        let m1 = sheet.total_mass();
        assert!((m0 - m1).abs() < 1e-14, "Mass not conserved: {m0} vs {m1}");
    }

    #[test]
    fn sheet_caustic_detection() {
        let domain = test_domain();
        let mut sheet = SheetTracker::new(domain.clone());
        // Create crossing: move half the particles to overlap with the other half
        let n = sheet.particles.len();
        for i in n / 2..n {
            sheet.particles[i].x = sheet.particles[i % (n / 2)].x;
        }
        let sc = sheet.stream_count();
        let max_count = *sc.data.iter().max().unwrap();
        assert!(
            max_count > 1,
            "Should detect stream crossing, max count = {max_count}"
        );
    }

    #[test]
    fn sheet_kinetic_energy() {
        let domain = test_domain();
        let mut sheet = SheetTracker::new(domain);
        // Set known velocities
        for p in &mut sheet.particles {
            p.v = [1.0, 0.0, 0.0];
        }
        let ke = sheet.total_kinetic_energy();
        let expected = 0.5 * sheet.particle_mass * sheet.particles.len() as f64 * 1.0;
        assert!(
            (ke - expected).abs() < 1e-14,
            "KE {ke} != expected {expected}"
        );
    }

    #[test]
    fn sheet_entropy_and_casimir() {
        let domain = test_domain();
        let sheet = SheetTracker::new(domain);
        assert_eq!(sheet.entropy(), 0.0, "Cold sheet should have zero entropy");
        assert!(
            sheet.casimir_c2().is_infinite(),
            "Cold sheet C₂ should diverge"
        );
    }

    #[test]
    fn sheet_velocity_distribution() {
        let domain = test_domain();
        let sheet = SheetTracker::new(domain);
        // All particles at cell centers with zero velocity should all be findable
        let pos = sheet.particles[0].x;
        let vdist = sheet.velocity_distribution(&pos);
        // At least the particle at this position should be found
        assert!(
            !vdist.is_empty(),
            "Should find at least one particle at cell center"
        );
        // With zero velocity, all magnitudes should be 0
        for &v in &vdist {
            assert!(v.abs() < 1e-14, "Zero-velocity particle has |v| = {v}");
        }
    }

    #[test]
    fn sheet_moment_order0() {
        let domain = test_domain();
        let sheet = SheetTracker::new(domain.clone());
        let pos = sheet.particles[0].x;
        let m = sheet.moment(&pos, 0);
        assert_eq!(m.rank, 0);
        // Density should be particle_mass / cell_volume for a single particle per cell
        let dx = domain.dx();
        let cell_vol = dx[0] * dx[1] * dx[2];
        let expected_rho = sheet.particle_mass / cell_vol;
        assert!(
            (m.data[0] - expected_rho).abs() / expected_rho < 1e-10,
            "moment order 0: {} vs expected {}",
            m.data[0],
            expected_rho
        );
    }

    #[test]
    fn sheet_advect_v() {
        let domain = test_domain();
        let mut sheet = SheetTracker::new(domain.clone());
        let n = 8 * 8 * 8;
        // Uniform acceleration field: gx = 0.5 everywhere
        let accel = AccelerationField {
            gx: vec![0.5; n],
            gy: vec![0.0; n],
            gz: vec![0.0; n],
            shape: [8, 8, 8],
        };
        let dt = 0.1;
        sheet.advect_v(&accel, dt);
        // All particles should have v[0] ≈ 0.5 * 0.1 = 0.05
        for (i, p) in sheet.particles.iter().enumerate() {
            assert!(
                (p.v[0] - 0.05).abs() < 1e-10,
                "particle {i}: v[0] = {}, expected 0.05",
                p.v[0]
            );
            assert!(
                p.v[1].abs() < 1e-14,
                "particle {i}: v[1] should be 0, got {}",
                p.v[1]
            );
        }
    }
}
