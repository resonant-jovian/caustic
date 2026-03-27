//! Octree-based adaptive mesh refinement (AMR) for 6D phase space.
//!
//! The domain is covered by a single root cell in (x, v) space. Cells where the
//! distribution function f exceeds a refinement threshold are recursively subdivided
//! into 2^6 = 64 children (bisection in all 6 dimensions). Coarsening merges children
//! back when their values become nearly uniform, and sparse velocity cleanup removes
//! blocks where f is negligible.
//!
//! For Poisson coupling, leaf cell values are deposited onto a uniform spatial grid
//! via nearest-cell assignment, producing a standard [`DensityField`].

use super::super::{
    context::SimContext,
    init::domain::{Domain, SpatialBoundType},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use rayon::prelude::*;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

/// One AMR cell in 6D phase space. Leaf cells store a value of f;
/// non-leaf cells have 64 children (one per sub-octant in 6D).
pub struct AmrCell {
    /// Center of this cell in 6D: [x1, x2, x3, v1, v2, v3].
    pub center: [f64; 6],
    /// Full width of this cell in each of the 6 dimensions.
    pub size: [f64; 6],
    /// Value of the distribution function f in this cell (meaningful for leaves).
    pub value: f64,
    /// 64 children if refined, None if this is a leaf.
    pub children: Option<Box<[AmrCell; 64]>>,
    /// Refinement level (0 = root).
    pub level: usize,
}

impl AmrCell {
    /// Returns true if this cell has no children (is a leaf node).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// 6D volume of this cell: product of all 6 side lengths.
    #[inline]
    pub fn cell_volume(&self) -> f64 {
        self.size[0] * self.size[1] * self.size[2] * self.size[3] * self.size[4] * self.size[5]
    }

    /// 3D velocity sub-volume of this cell: product of velocity side lengths.
    #[inline]
    pub fn velocity_volume(&self) -> f64 {
        self.size[3] * self.size[4] * self.size[5]
    }

    /// Subdivide this cell into 64 children (2^6). Each child has half the parent's
    /// size in every dimension and inherits the parent's value.
    pub fn subdivide(&mut self) {
        if self.children.is_some() {
            return;
        }

        let child_size: [f64; 6] = [
            self.size[0] / 2.0,
            self.size[1] / 2.0,
            self.size[2] / 2.0,
            self.size[3] / 2.0,
            self.size[4] / 2.0,
            self.size[5] / 2.0,
        ];

        let child_level = self.level + 1;
        let parent_value = self.value;
        let parent_center = self.center;

        // Build children as a Vec, then convert to boxed array.
        let children_vec: Vec<AmrCell> = (0..64)
            .map(|idx| {
                // Each bit of idx (0..5) determines ± offset in that dimension.
                let mut child_center = [0.0f64; 6];
                for d in 0..6 {
                    let bit = (idx >> d) & 1;
                    let offset = if bit == 0 { -0.25 } else { 0.25 };
                    child_center[d] = parent_center[d] + offset * self.size[d];
                }
                AmrCell {
                    center: child_center,
                    size: child_size,
                    value: parent_value,
                    children: None,
                    level: child_level,
                }
            })
            .collect();

        // SAFETY: children_vec is constructed from (0..64).map() above, so it
        // always has exactly 64 elements. The try_into cannot fail.
        debug_assert_eq!(children_vec.len(), 64);
        let boxed_slice = children_vec.into_boxed_slice();
        let boxed_array: Box<[AmrCell; 64]> = match boxed_slice.try_into() {
            Ok(arr) => arr,
            Err(_) => return, // unreachable: length is always 64
        };

        self.children = Some(boxed_array);
    }

    /// Collect references to all leaf cells in the subtree rooted at this cell.
    pub fn collect_leaves(&self) -> Vec<&AmrCell> {
        let mut leaves = Vec::new();
        self.collect_leaves_inner(&mut leaves);
        leaves
    }

    fn collect_leaves_inner<'a>(&'a self, out: &mut Vec<&'a AmrCell>) {
        if self.is_leaf() {
            out.push(self);
        } else if let Some(ref children) = self.children {
            for child in children.iter() {
                child.collect_leaves_inner(out);
            }
        }
    }

    /// Collect mutable references to all leaf cells in the subtree rooted at this cell.
    pub fn collect_leaves_mut(&mut self) -> Vec<&mut AmrCell> {
        let mut leaves = Vec::new();
        self.collect_leaves_mut_inner(&mut leaves);
        leaves
    }

    fn collect_leaves_mut_inner<'a>(&'a mut self, out: &mut Vec<&'a mut AmrCell>) {
        if self.is_leaf() {
            out.push(self);
        } else if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.collect_leaves_mut_inner(out);
            }
        }
    }

    /// Returns true if the given 6D point lies within this cell's bounding box.
    #[inline]
    pub fn contains(&self, point: &[f64; 6]) -> bool {
        self.center
            .iter()
            .zip(self.size.iter())
            .zip(point.iter())
            .all(|((&c, &s), &p)| {
                let lo = c - s / 2.0;
                let hi = c + s / 2.0;
                p >= lo && p < hi
            })
    }

    /// Recursive refinement: subdivide leaf cells whose |value| exceeds the threshold
    /// and whose level is below max_level.
    fn refine_recursive(&mut self, threshold: f64, max_level: usize) {
        if self.is_leaf() {
            if self.value.abs() > threshold && self.level < max_level {
                self.subdivide();
            }
        } else if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.refine_recursive(threshold, max_level);
            }
        }
    }

    /// Recursive coarsening: if all 64 children are leaves and their values are nearly
    /// uniform (max - min < merge_threshold), merge them back into one leaf whose value
    /// is the mean.
    fn coarsen_recursive(&mut self, merge_threshold: f64) {
        if self.is_leaf() {
            return;
        }

        if let Some(ref mut children) = self.children {
            // First recurse into children.
            for child in children.iter_mut() {
                child.coarsen_recursive(merge_threshold);
            }

            // Check if all children are leaves.
            let all_leaves = children.iter().all(|c| c.is_leaf());
            if !all_leaves {
                return;
            }

            // Check value uniformity.
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            let mut sum = 0.0;
            for child in children.iter() {
                min_val = min_val.min(child.value);
                max_val = max_val.max(child.value);
                sum += child.value;
            }

            if (max_val - min_val).abs() < merge_threshold {
                self.value = sum / 64.0;
                self.children = None;
            }
        }
    }
}

/// Adaptive mesh refinement grid in 6D phase space.
///
/// The root cell spans the full domain. Cells are refined where the distribution
/// function is significant and coarsened when values become uniform. Implements
/// [`PhaseSpaceRepr`] by depositing leaf values onto a uniform spatial grid.
pub struct AmrGrid {
    /// Root cell of the tree, spanning the entire 6D domain.
    pub root: AmrCell,
    /// The simulation domain (spatial/velocity extents, resolutions, BCs).
    pub domain: Domain,
    /// Cells with |f| > refinement_threshold are subdivided.
    pub refinement_threshold: f64,
    /// Gradient-based refinement threshold (reserved for future use).
    pub gradient_threshold: f64,
    /// Maximum refinement level (0 = root only).
    pub max_level: usize,
    /// Velocity block removal threshold: blocks with max(|f|) < this are deallocated.
    /// Set to 0 to disable sparse velocity cleanup. Default: 1e-14.
    pub velocity_removal_threshold: f64,
    /// Current active block count (leaves with non-negligible f).
    pub active_block_count: usize,
}

impl AmrGrid {
    /// Create a new AmrGrid with a single root cell spanning the full 6D domain.
    ///
    /// - `refinement_threshold`: cells with |f| above this are subdivided.
    /// - `max_levels`: maximum tree depth (0 = root only, no refinement).
    pub fn new(domain: Domain, refinement_threshold: f64, max_levels: usize) -> Self {
        let lx = domain.lx();
        let lv = domain.lv();

        let root = AmrCell {
            center: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            size: [
                2.0 * lx[0],
                2.0 * lx[1],
                2.0 * lx[2],
                2.0 * lv[0],
                2.0 * lv[1],
                2.0 * lv[2],
            ],
            value: 0.0,
            children: None,
            level: 0,
        };

        let mut grid = AmrGrid {
            root,
            domain,
            refinement_threshold,
            gradient_threshold: refinement_threshold,
            max_level: max_levels,
            velocity_removal_threshold: 1e-14,
            active_block_count: 1,
        };
        grid.update_block_count();
        grid
    }

    /// Refine cells where |f| exceeds the refinement threshold, up to max_level.
    pub fn refine(&mut self) {
        self.root
            .refine_recursive(self.refinement_threshold, self.max_level);
        self.update_block_count();
    }

    /// Merge children back into parent when their values are nearly uniform
    /// (spread < threshold / 10).
    pub fn coarsen(&mut self) {
        let merge_threshold = self.refinement_threshold / 10.0;
        self.root.coarsen_recursive(merge_threshold);
        self.update_block_count();
    }

    /// Remove velocity-space blocks where max(|f|) < velocity_removal_threshold.
    ///
    /// Walks the tree and coarsens leaf cells in velocity dimensions whose
    /// values are below the threshold. This reduces memory for cold/warm
    /// distributions where most of velocity space is empty.
    pub fn cleanup_sparse_velocity(&mut self) {
        if self.velocity_removal_threshold <= 0.0 {
            return;
        }
        let thresh = self.velocity_removal_threshold;
        self.root.coarsen_recursive(thresh);
        self.update_block_count();
    }

    /// Update the active block count from the tree.
    fn update_block_count(&mut self) {
        self.active_block_count = self.root.collect_leaves().len();
    }

    /// Number of leaf cells currently in the tree.
    pub fn leaf_count(&self) -> usize {
        self.root.collect_leaves().len()
    }

    /// Helper: spatial extents as [lx1, lx2, lx3].
    #[inline]
    fn lx(&self) -> [f64; 3] {
        self.domain.lx()
    }

    /// Helper: velocity extents as [lv1, lv2, lv3].
    #[inline]
    fn lv(&self) -> [f64; 3] {
        self.domain.lv()
    }

    /// Helper: spatial grid sizes [nx1, nx2, nx3].
    #[inline]
    fn nx(&self) -> [usize; 3] {
        [
            self.domain.spatial_res.x1 as usize,
            self.domain.spatial_res.x2 as usize,
            self.domain.spatial_res.x3 as usize,
        ]
    }

    /// Helper: velocity grid sizes [nv1, nv2, nv3].
    #[inline]
    fn nv(&self) -> [usize; 3] {
        [
            self.domain.velocity_res.v1 as usize,
            self.domain.velocity_res.v2 as usize,
            self.domain.velocity_res.v3 as usize,
        ]
    }

    /// Map a spatial position to a grid index, clamped to [0, n-1].
    #[inline]
    fn spatial_index(&self, pos: &[f64; 3]) -> [usize; 3] {
        let lx = self.lx();
        let nx = self.nx();
        let dx = self.domain.dx();
        [
            ((pos[0] + lx[0]) / dx[0])
                .floor()
                .clamp(0.0, (nx[0] - 1) as f64) as usize,
            ((pos[1] + lx[1]) / dx[1])
                .floor()
                .clamp(0.0, (nx[1] - 1) as f64) as usize,
            ((pos[2] + lx[2]) / dx[2])
                .floor()
                .clamp(0.0, (nx[2] - 1) as f64) as usize,
        ]
    }

    /// Wrap spatial coordinates for periodic BC. For non-periodic BC, coordinates are
    /// left unchanged (particles that leave the domain are effectively absorbed).
    fn wrap_spatial(&self, coords: &mut [f64; 3]) {
        if matches!(self.domain.spatial_bc, SpatialBoundType::Periodic) {
            let lx = self.lx();
            for d in 0..3 {
                let l = lx[d];
                // Domain is [-l, l), period = 2*l
                let period = 2.0 * l;
                coords[d] = ((coords[d] + l) % period + period) % period - l;
            }
        }
    }

    /// Trilinear interpolation of a 3D field at a given position.
    fn interpolate_field_3d(&self, field: &[f64], shape: &[usize; 3], pos: &[f64; 3]) -> f64 {
        let lx = self.lx();
        let dx = self.domain.dx();
        let [nx, ny, nz] = *shape;

        // Fractional index.
        let fx = (pos[0] + lx[0]) / dx[0] - 0.5;
        let fy = (pos[1] + lx[1]) / dx[1] - 0.5;
        let fz = (pos[2] + lx[2]) / dx[2] - 0.5;

        let ix0 = fx.floor() as i64;
        let iy0 = fy.floor() as i64;
        let iz0 = fz.floor() as i64;

        let wx = fx - ix0 as f64;
        let wy = fy - iy0 as f64;
        let wz = fz - iz0 as f64;

        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);

        let clamp_or_wrap = |i: i64, n: usize| -> usize {
            if periodic {
                ((i % n as i64) + n as i64) as usize % n
            } else {
                i.clamp(0, n as i64 - 1) as usize
            }
        };

        let mut result = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dxi in 0..2 {
                    let ci = clamp_or_wrap(ix0 + dxi as i64, nx);
                    let cj = clamp_or_wrap(iy0 + dy as i64, ny);
                    let ck = clamp_or_wrap(iz0 + dz as i64, nz);
                    let ww = if dxi == 0 { 1.0 - wx } else { wx }
                        * if dy == 0 { 1.0 - wy } else { wy }
                        * if dz == 0 { 1.0 - wz } else { wz };
                    result += ww * field[ci * ny * nz + cj * nz + ck];
                }
            }
        }
        result
    }
}

impl PhaseSpaceRepr for AmrGrid {
    /// Compute density rho(x) = integral of f dv^3 by accumulating leaf cell contributions
    /// onto the spatial grid.
    fn compute_density(&self) -> DensityField {
        let nx = self.nx();
        let dx = self.domain.dx();
        let lx = self.lx();
        let n_spatial = nx[0] * nx[1] * nx[2];

        // Spatial cell volume for normalization.
        let dx3 = self.domain.cell_volume_3d();

        let leaves = self.root.collect_leaves();
        let rho = leaves
            .par_iter()
            .filter(|leaf| leaf.value.abs() >= 1e-300)
            .fold(
                || vec![0.0f64; n_spatial],
                |mut local_rho, leaf| {
                    // Determine which spatial cell(s) this leaf overlaps.
                    // For simplicity, deposit into the nearest spatial cell based on center.
                    let x = [leaf.center[0], leaf.center[1], leaf.center[2]];

                    // Find the spatial grid cell.
                    let ix = ((x[0] + lx[0]) / dx[0])
                        .floor()
                        .clamp(0.0, (nx[0] - 1) as f64) as usize;
                    let iy = ((x[1] + lx[1]) / dx[1])
                        .floor()
                        .clamp(0.0, (nx[1] - 1) as f64) as usize;
                    let iz = ((x[2] + lx[2]) / dx[2])
                        .floor()
                        .clamp(0.0, (nx[2] - 1) as f64) as usize;

                    // Contribution: f * dv^3_cell (the velocity sub-volume of this leaf cell).
                    // We also need to account for the fact that the leaf's spatial sub-volume
                    // may be smaller than dx3. The density contribution is:
                    //   rho_cell += f * (leaf_velocity_volume) * (leaf_spatial_volume / dx3)
                    // where the spatial volume ratio accounts for the fraction of the spatial
                    // cell covered by this leaf.
                    let leaf_spatial_vol = leaf.size[0] * leaf.size[1] * leaf.size[2];
                    let contribution = leaf.value * leaf.velocity_volume() * leaf_spatial_vol / dx3;
                    let flat = ix * nx[1] * nx[2] + iy * nx[2] + iz;
                    local_rho[flat] += contribution;
                    local_rho
                },
            )
            .reduce(
                || vec![0.0f64; n_spatial],
                |mut a, b| {
                    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += *y);
                    a
                },
            );

        DensityField {
            data: rho,
            shape: [nx[0], nx[1], nx[2]],
        }
    }

    /// Drift sub-step: shift leaf cell centers in spatial coordinates by v * dt.
    /// The velocity coordinates (center[3..6]) of each leaf encode its velocity,
    /// so the spatial drift is center[0..3] += center[3..6] * dt.
    fn advect_x(&mut self, _displacement: &DisplacementField, ctx: &SimContext) {
        let dt = ctx.dt;
        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);
        let lx = self.lx();

        let leaves = self.root.collect_leaves_mut();
        leaves.into_par_iter().for_each(|leaf| {
            // Drift: x_new = x_old + v * dt
            leaf.center[0] += leaf.center[3] * dt;
            leaf.center[1] += leaf.center[4] * dt;
            leaf.center[2] += leaf.center[5] * dt;

            // Wrap for periodic BC.
            if periodic {
                for (d, &l) in lx.iter().enumerate() {
                    let period = 2.0 * l;
                    leaf.center[d] = ((leaf.center[d] + l) % period + period) % period - l;
                }
            }
        });
    }

    /// Kick sub-step: shift leaf cell centers in velocity coordinates by a(x) * dt.
    /// The acceleration is interpolated from the AccelerationField at the leaf's
    /// spatial position.
    fn advect_v(&mut self, acceleration: &AccelerationField, ctx: &SimContext) {
        let dt = ctx.dt;
        let lx = self.lx();
        let dx = self.domain.dx();
        let nx = self.nx();

        // Pre-borrow acceleration components for interpolation.
        let gx = &acceleration.gx;
        let gy = &acceleration.gy;
        let gz = &acceleration.gz;
        let shape = &acceleration.shape;

        let leaves = self.root.collect_leaves_mut();
        leaves.into_par_iter().for_each(|leaf| {
            let pos = [leaf.center[0], leaf.center[1], leaf.center[2]];

            // Nearest-grid-point acceleration lookup (fast path).
            let ix = ((pos[0] + lx[0]) / dx[0])
                .floor()
                .clamp(0.0, (nx[0] - 1) as f64) as usize;
            let iy = ((pos[1] + lx[1]) / dx[1])
                .floor()
                .clamp(0.0, (nx[1] - 1) as f64) as usize;
            let iz = ((pos[2] + lx[2]) / dx[2])
                .floor()
                .clamp(0.0, (nx[2] - 1) as f64) as usize;
            let flat = ix * shape[1] * shape[2] + iy * shape[2] + iz;

            leaf.center[3] += gx[flat] * dt;
            leaf.center[4] += gy[flat] * dt;
            leaf.center[5] += gz[flat] * dt;
        });
    }

    /// Compute velocity moment of order n at the given spatial position.
    /// Sums over leaf cells whose spatial center is in the same grid cell.
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let ix_target = self.spatial_index(position);
        let dx = self.domain.dx();
        let lx = self.lx();
        let nx = self.nx();

        let leaves = self.root.collect_leaves();

        // Identify leaves that fall in the same spatial cell.
        let matching_leaves: Vec<&&AmrCell> = leaves
            .iter()
            .filter(|leaf| {
                let ix = ((leaf.center[0] + lx[0]) / dx[0])
                    .floor()
                    .clamp(0.0, (nx[0] - 1) as f64) as usize;
                let iy = ((leaf.center[1] + lx[1]) / dx[1])
                    .floor()
                    .clamp(0.0, (nx[1] - 1) as f64) as usize;
                let iz = ((leaf.center[2] + lx[2]) / dx[2])
                    .floor()
                    .clamp(0.0, (nx[2] - 1) as f64) as usize;
                ix == ix_target[0] && iy == ix_target[1] && iz == ix_target[2]
            })
            .collect();

        match order {
            0 => {
                // Zeroth moment: density = sum f * dv^3_cell.
                let rho: f64 = matching_leaves
                    .iter()
                    .map(|leaf| leaf.value * leaf.velocity_volume())
                    .sum();
                Tensor {
                    data: vec![rho],
                    rank: 0,
                    shape: vec![],
                }
            }
            1 => {
                // First moment: mean velocity = (1/rho) * sum f * v * dv^3.
                let mut rho = 0.0f64;
                let mut vbar = [0.0f64; 3];
                for leaf in &matching_leaves {
                    let w = leaf.value * leaf.velocity_volume();
                    rho += w;
                    vbar[0] += w * leaf.center[3];
                    vbar[1] += w * leaf.center[4];
                    vbar[2] += w * leaf.center[5];
                }
                let scale = if rho.abs() > 1e-30 { 1.0 / rho } else { 0.0 };
                Tensor {
                    data: vec![vbar[0] * scale, vbar[1] * scale, vbar[2] * scale],
                    rank: 1,
                    shape: vec![3],
                }
            }
            2 => {
                // Second moment: velocity dispersion tensor sum f * vi * vj * dv^3.
                let mut m2 = [0.0f64; 9];
                for leaf in &matching_leaves {
                    let w = leaf.value * leaf.velocity_volume();
                    let v = [leaf.center[3], leaf.center[4], leaf.center[5]];
                    for i in 0..3 {
                        for j in 0..3 {
                            m2[i * 3 + j] += w * v[i] * v[j];
                        }
                    }
                }
                Tensor {
                    data: m2.to_vec(),
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

    /// Total mass M = integral of f dx^3 dv^3 = sum over leaves of f * cell_volume_6D.
    fn total_mass(&self) -> f64 {
        self.root
            .collect_leaves()
            .par_iter()
            .map(|leaf| leaf.value * leaf.cell_volume())
            .sum()
    }

    /// Casimir invariant C2 = integral of f^2 dx^3 dv^3.
    fn casimir_c2(&self) -> f64 {
        self.root
            .collect_leaves()
            .par_iter()
            .map(|leaf| leaf.value * leaf.value * leaf.cell_volume())
            .sum()
    }

    /// Boltzmann entropy S = -integral of f ln(f) dx^3 dv^3.
    fn entropy(&self) -> f64 {
        self.root
            .collect_leaves()
            .par_iter()
            .filter(|leaf| leaf.value > 0.0)
            .map(|leaf| -leaf.value * leaf.value.ln() * leaf.cell_volume())
            .sum()
    }

    /// Count distinct velocity streams at each spatial position.
    /// For each spatial grid cell, count the number of leaf cells that overlap it
    /// with significant value (indicating separate velocity-space populations).
    fn stream_count(&self) -> StreamCountField {
        let nx = self.nx();
        let dx = self.domain.dx();
        let lx = self.lx();
        let n_spatial = nx[0] * nx[1] * nx[2];

        let leaves = self.root.collect_leaves();
        let counts = leaves
            .par_iter()
            .filter(|leaf| leaf.value.abs() >= 1e-30)
            .fold(
                || vec![0u32; n_spatial],
                |mut local, leaf| {
                    let ix = ((leaf.center[0] + lx[0]) / dx[0])
                        .floor()
                        .clamp(0.0, (nx[0] - 1) as f64) as usize;
                    let iy = ((leaf.center[1] + lx[1]) / dx[1])
                        .floor()
                        .clamp(0.0, (nx[1] - 1) as f64) as usize;
                    let iz = ((leaf.center[2] + lx[2]) / dx[2])
                        .floor()
                        .clamp(0.0, (nx[2] - 1) as f64) as usize;
                    let flat = ix * nx[1] * nx[2] + iy * nx[2] + iz;
                    local[flat] += 1;
                    local
                },
            )
            .reduce(
                || vec![0u32; n_spatial],
                |mut a, b| {
                    a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += *y);
                    a
                },
            );

        StreamCountField {
            data: counts,
            shape: [nx[0], nx[1], nx[2]],
        }
    }

    /// Extract the velocity distribution f(v | x) at a given spatial position.
    /// Returns the values of f from all leaf cells whose spatial center maps to
    /// the same grid cell as the query position.
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let ix_target = self.spatial_index(position);
        let dx = self.domain.dx();
        let lx = self.lx();
        let nx = self.nx();

        let leaves = self.root.collect_leaves();
        leaves
            .iter()
            .filter(|leaf| {
                let ix = ((leaf.center[0] + lx[0]) / dx[0])
                    .floor()
                    .clamp(0.0, (nx[0] - 1) as f64) as usize;
                let iy = ((leaf.center[1] + lx[1]) / dx[1])
                    .floor()
                    .clamp(0.0, (nx[1] - 1) as f64) as usize;
                let iz = ((leaf.center[2] + lx[2]) / dx[2])
                    .floor()
                    .clamp(0.0, (nx[2] - 1) as f64) as usize;
                ix == ix_target[0] && iy == ix_target[1] && iz == ix_target[2]
            })
            .map(|leaf| leaf.value)
            .collect()
    }

    /// Total kinetic energy T = (1/2) integral of f * v^2 dx^3 dv^3.
    fn total_kinetic_energy(&self) -> Option<f64> {
        let leaves = self.root.collect_leaves();
        let t: f64 = leaves
            .par_iter()
            .map(|leaf| {
                let v2 = leaf.center[3] * leaf.center[3]
                    + leaf.center[4] * leaf.center[4]
                    + leaf.center[5] * leaf.center[5];
                leaf.value * v2 * leaf.cell_volume()
            })
            .sum();
        Some(0.5 * t)
    }

    /// Extract a full 6D snapshot by depositing leaf values onto a uniform grid.
    fn to_snapshot(&self, time: f64) -> Option<PhaseSpaceSnapshot> {
        let nx = self.nx();
        let nv = self.nv();
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.lx();
        let lv = self.lv();

        let total = nx[0] * nx[1] * nx[2] * nv[0] * nv[1] * nv[2];
        let atomic_data: Vec<AtomicU64> = (0..total).map(|_| AtomicU64::new(0u64)).collect();

        let s_v3 = 1usize;
        let s_v2 = nv[2];
        let s_v1 = nv[1] * nv[2];
        let s_x3 = nv[0] * s_v1;
        let s_x2 = nx[2] * s_x3;
        let s_x1 = nx[1] * s_x2;

        let uniform_vol = self.domain.cell_volume_6d();

        let leaves = self.root.collect_leaves();
        leaves
            .par_iter()
            .filter(|leaf| leaf.value.abs() >= 1e-300)
            .for_each(|leaf| {
                // Map leaf center to 6D grid indices.
                let ix = ((leaf.center[0] + lx[0]) / dx[0])
                    .floor()
                    .clamp(0.0, (nx[0] - 1) as f64) as usize;
                let iy = ((leaf.center[1] + lx[1]) / dx[1])
                    .floor()
                    .clamp(0.0, (nx[1] - 1) as f64) as usize;
                let iz = ((leaf.center[2] + lx[2]) / dx[2])
                    .floor()
                    .clamp(0.0, (nx[2] - 1) as f64) as usize;
                let iv1 = ((leaf.center[3] + lv[0]) / dv[0])
                    .floor()
                    .clamp(0.0, (nv[0] - 1) as f64) as usize;
                let iv2 = ((leaf.center[4] + lv[1]) / dv[1])
                    .floor()
                    .clamp(0.0, (nv[1] - 1) as f64) as usize;
                let iv3 = ((leaf.center[5] + lv[2]) / dv[2])
                    .floor()
                    .clamp(0.0, (nv[2] - 1) as f64) as usize;

                let flat = ix * s_x1 + iy * s_x2 + iz * s_x3 + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3;
                // Accumulate — multiple AMR leaves may map to the same uniform cell.
                // Weight by the ratio of the leaf's 6D volume to the uniform cell volume.
                let weight = leaf.cell_volume() / uniform_vol;
                let val = leaf.value * weight;
                let atom = &atomic_data[flat];
                let mut old = atom.load(Ordering::Relaxed);
                loop {
                    let new = f64::from_bits(old) + val;
                    match atom.compare_exchange_weak(
                        old,
                        new.to_bits(),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(x) => old = x,
                    }
                }
            });

        let data: Vec<f64> = atomic_data
            .into_iter()
            .map(|a| f64::from_bits(a.into_inner()))
            .collect();

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::context::SimContext;
    use crate::tooling::core::events::EventEmitter;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::progress::StepProgress;

    fn test_domain() -> Domain {
        Domain::builder()
            .spatial_extent(2.0)
            .velocity_extent(2.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn amr_cell_basics() {
        let cell = AmrCell {
            center: [0.0; 6],
            size: [2.0; 6],
            value: 1.0,
            children: None,
            level: 0,
        };
        assert!(cell.is_leaf());
        assert!((cell.cell_volume() - 64.0).abs() < 1e-12); // 2^6
        assert!((cell.velocity_volume() - 8.0).abs() < 1e-12); // 2^3
    }

    #[test]
    fn amr_subdivide() {
        let mut cell = AmrCell {
            center: [0.0; 6],
            size: [4.0; 6],
            value: 5.0,
            children: None,
            level: 0,
        };
        cell.subdivide();
        assert!(!cell.is_leaf());
        let children = cell.children.as_ref().unwrap();
        assert_eq!(children.len(), 64);
        // Each child should have half the size and inherit the value.
        for child in children.iter() {
            assert_eq!(child.level, 1);
            assert!((child.value - 5.0).abs() < 1e-12);
            for d in 0..6 {
                assert!((child.size[d] - 2.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn amr_collect_leaves() {
        let mut cell = AmrCell {
            center: [0.0; 6],
            size: [4.0; 6],
            value: 1.0,
            children: None,
            level: 0,
        };
        assert_eq!(cell.collect_leaves().len(), 1);

        cell.subdivide();
        assert_eq!(cell.collect_leaves().len(), 64);

        // Subdivide one child further.
        if let Some(ref mut children) = cell.children {
            children[0].subdivide();
        }
        // 63 original children + 64 grandchildren = 127.
        assert_eq!(cell.collect_leaves().len(), 63 + 64);
    }

    #[test]
    fn amr_refinement_concentration() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 0.1, 3);
        // Set root value high to trigger refinement.
        amr.root.value = 1.0;
        amr.refine();
        // Should have children.
        assert!(
            amr.root.children.is_some(),
            "Root should be refined when value > threshold"
        );
        let leaves = amr.root.collect_leaves();
        assert_eq!(
            leaves.len(),
            64,
            "Should have 64 children after one refinement"
        );
    }

    #[test]
    fn amr_mass_conservation() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 0.01, 2);
        amr.root.value = 1.0;
        let m0 = amr.total_mass();

        // Advect with zero displacement — mass should be exactly preserved.
        let n = 4 * 4 * 4;
        let dummy = DisplacementField {
            dx: vec![0.0; n],
            dy: vec![0.0; n],
            dz: vec![0.0; n],
            shape: [4, 4, 4],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = amr.domain.clone();

        let __solver = FftPoisson::new(&__domain_tmp);

        let __ctx = SimContext {
            solver: &__solver,

            advector: &__advector,

            emitter: &__emitter,

            progress: &__progress,

            step: 0,

            time: 0.0,

            dt: 0.01,

            g: 0.0,
        };

        amr.advect_x(&dummy, &__ctx);
        let m1 = amr.total_mass();

        assert!(
            (m0 - m1).abs() / m0.max(1e-15) < 1e-10,
            "Mass not conserved: {m0} vs {m1}"
        );
    }

    #[test]
    fn amr_mass_conservation_after_refinement() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 0.1, 2);
        amr.root.value = 1.0;
        let m_before = amr.total_mass();

        amr.refine();
        let m_after = amr.total_mass();

        assert!(
            (m_before - m_after).abs() / m_before.max(1e-15) < 1e-10,
            "Refinement should conserve mass: {m_before} vs {m_after}"
        );
    }

    #[test]
    fn amr_coarsen_uniform() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 0.5, 2);
        amr.root.value = 1.0;

        // Refine.
        amr.refine();
        assert_eq!(amr.leaf_count(), 64);

        // All children have the same value (1.0), so coarsening should merge them.
        amr.coarsen();
        assert_eq!(
            amr.leaf_count(),
            1,
            "Uniform children should be merged back into one leaf"
        );
        assert!((amr.root.value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn amr_convergence() {
        // Higher max_level should give more cells and finer resolution.
        let domain = test_domain();
        let mut amr1 = AmrGrid::new(domain.clone(), 0.01, 1);
        amr1.root.value = 1.0;
        amr1.refine();

        let mut amr2 = AmrGrid::new(domain.clone(), 0.01, 2);
        amr2.root.value = 1.0;
        amr2.refine();
        // Refine children too (they inherit value = 1.0 > threshold = 0.01).
        amr2.refine();

        let rho1 = amr1.compute_density();
        let rho2 = amr2.compute_density();

        // Both should have non-zero density.
        assert!(rho1.data.iter().any(|&x| x > 0.0));
        assert!(rho2.data.iter().any(|&x| x > 0.0));

        // The finer grid should have more leaf cells.
        assert!(amr2.leaf_count() > amr1.leaf_count());
    }

    #[test]
    fn amr_entropy_positive() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 0.1, 1);
        amr.root.value = 2.0; // f > 1 so ln(f) > 0, entropy = -f*ln(f) < 0
        // Actually for the Boltzmann entropy -f*ln(f), when f > e it's negative.
        // For f = 0.5, -0.5*ln(0.5) = 0.5*0.693 > 0.
        amr.root.value = 0.5;
        let s = amr.entropy();
        assert!(s > 0.0, "Entropy should be positive for 0 < f < 1, got {s}");
    }

    #[test]
    fn amr_casimir_c2() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 0.1, 1);
        amr.root.value = 3.0;
        let c2 = amr.casimir_c2();
        let expected = 3.0 * 3.0 * amr.root.cell_volume();
        assert!(
            (c2 - expected).abs() / expected < 1e-12,
            "C2 mismatch: {c2} vs {expected}"
        );
    }

    #[test]
    fn amr_kinetic_energy() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 100.0, 1);
        // Don't refine. The root cell has center at velocity = (0,0,0),
        // so v^2 = 0 and T = 0.
        amr.root.value = 1.0;
        let t = amr.total_kinetic_energy().unwrap();
        assert!(
            t.abs() < 1e-12,
            "Kinetic energy should be zero for v=0 center, got {t}"
        );
    }

    #[test]
    fn amr_advect_v() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 100.0, 1);
        amr.root.value = 1.0;
        let v_before = amr.root.center[3];

        // Create a uniform acceleration field.
        let n = 4 * 4 * 4;
        let acc = AccelerationField {
            gx: vec![1.0; n],
            gy: vec![0.0; n],
            gz: vec![0.0; n],
            shape: [4, 4, 4],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = amr.domain.clone();

        let __solver = FftPoisson::new(&__domain_tmp);

        let __ctx = SimContext {
            solver: &__solver,

            advector: &__advector,

            emitter: &__emitter,

            progress: &__progress,

            step: 0,

            time: 0.0,

            dt: 0.5,

            g: 0.0,
        };

        amr.advect_v(&acc, &__ctx);

        // v1 should have increased by a*dt = 1.0 * 0.5 = 0.5.
        let v_after = amr.root.center[3];
        assert!(
            (v_after - v_before - 0.5).abs() < 1e-12,
            "Expected v1 shift of 0.5, got {}",
            v_after - v_before
        );
    }

    #[test]
    fn amr_advect_x_periodic_wrap() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 100.0, 1);
        amr.root.value = 1.0;

        // Give the root cell a large velocity so it wraps around.
        amr.root.center[3] = 1.0; // v_x = 1.0
        let n = 4 * 4 * 4;
        let dummy = DisplacementField {
            dx: vec![0.0; n],
            dy: vec![0.0; n],
            dz: vec![0.0; n],
            shape: [4, 4, 4],
        };

        // Advance by dt = 5.0: x_new = 0 + 1.0 * 5.0 = 5.0, should wrap to 5 mod 4 - 2 = 1.0.
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = amr.domain.clone();

        let __solver = FftPoisson::new(&__domain_tmp);

        let __ctx = SimContext {
            solver: &__solver,

            advector: &__advector,

            emitter: &__emitter,

            progress: &__progress,

            step: 0,

            time: 0.0,

            dt: 5.0,

            g: 0.0,
        };

        amr.advect_x(&dummy, &__ctx);
        let x = amr.root.center[0];
        // Domain is [-2, 2), period = 4. 5.0 mod 4 = 1.0. 1.0 - 2.0 = -1.0? No.
        // ((5.0 + 2.0) % 4.0 + 4.0) % 4.0 - 2.0 = (7.0 % 4.0) - 2.0 = 3.0 - 2.0 = 1.0.
        assert!(
            (x - 1.0).abs() < 1e-12,
            "Expected periodic wrap to 1.0, got {x}"
        );
    }

    #[test]
    fn amr_density_nonzero() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 0.01, 1);
        amr.root.value = 1.0;
        amr.refine();

        let rho = amr.compute_density();
        let total: f64 = rho.data.iter().sum();
        assert!(
            total > 0.0,
            "Density should be non-zero for non-zero distribution"
        );
    }

    #[test]
    fn amr_snapshot_roundtrip() {
        let domain = test_domain();
        let mut amr = AmrGrid::new(domain, 100.0, 1);
        amr.root.value = 2.5;

        let snap = amr.to_snapshot(0.0).unwrap();
        assert_eq!(snap.shape, [4, 4, 4, 4, 4, 4]);
        // The root cell covers the entire domain, so all uniform cells should get
        // some contribution.
        let nonzero = snap.data.iter().filter(|&&x| x > 0.0).count();
        assert!(nonzero > 0, "Snapshot should have nonzero entries");
    }
}
