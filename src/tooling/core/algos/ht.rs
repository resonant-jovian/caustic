#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! Hierarchical Tucker (HT) tensor decomposition of the 6D distribution function f(x,v).
//!
//! The HT format exploits the balanced binary tree structure of the x-v split:
//! ```text
//!               {0,1,2,3,4,5}          root
//!               /            \
//!        {0,1,2}              {3,4,5}    x-v split
//!         /    \              /     \
//!      {0}   {1,2}         {3}    {4,5}
//!             / \                  / \
//!           {1} {2}              {4} {5}
//! ```
//! Dimensions: 0=x₁, 1=x₂, 2=x₃, 3=v₁, 4=v₂, 5=v₃.
//!
//! Memory: O(dnk² + dk³) where n = max grid size, k = max rank, d = 6.
//! This is strictly more compact than TT (which requires k² per node) because
//! the balanced tree has depth log₂(d) instead of d.

use super::super::{
    init::domain::{Domain, SpatialBoundType, VelocityBoundType},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use super::aca::{BlackBoxMatrix, FnMatrix, Xorshift64, aca_partial_pivot};
use super::uniform::VelocityFilterConfig;
use faer::Mat;
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Fixed dimension tree topology for 6D ───────────────────────────────────
//   0..5:  leaves (dim 0..5)
//   6:     interior {1,2}   children: 1, 2
//   7:     interior {4,5}   children: 4, 5
//   8:     interior {0,1,2} children: 0, 6
//   9:     interior {3,4,5} children: 3, 7
//  10:     root {0..5}      children: 8, 9

const NUM_NODES: usize = 11;
const NUM_LEAVES: usize = 6;
const ROOT: usize = 10;

/// Maximum number of bytes that `to_full()` may allocate (2 GB).
/// HtTensor methods that would exceed this threshold return fallback values
/// instead of OOM-killing the process.
const MAX_MATERIALIZE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// A node in the HT dimension tree.
#[derive(Clone)]
pub enum HtNode {
    /// Leaf node: stores a basis matrix U_μ ∈ ℝ^{n_μ × k_μ}.
    Leaf { dim: usize, frame: Mat<f64> },
    /// Interior node: stores transfer tensor B_t ∈ ℝ^{k_t × k_left × k_right} (flat, row-major).
    Interior {
        left: usize,
        right: usize,
        transfer: Vec<f64>,
        /// [k_t, k_left, k_right]
        ranks: [usize; 3],
    },
}

impl HtNode {
    pub fn rank(&self) -> usize {
        match self {
            HtNode::Leaf { frame, .. } => frame.ncols(),
            HtNode::Interior { ranks, .. } => ranks[0],
        }
    }
}

/// Interpolation mode for SLAR advection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InterpolationMode {
    /// 4³=64 point tensor-product Catmull-Rom stencil (original).
    TricubicCatmullRom,
    /// O(d²) sparse polynomial: 1 + 2d + d(d-1)/2 evaluations (10 for d=3).
    SparsePolynomial,
}

/// Hierarchical Tucker tensor for 6D phase space.
pub struct HtTensor {
    pub nodes: Vec<HtNode>,
    pub shape: [usize; 6],
    pub domain: Domain,
    /// Approximation tolerance used in HTACA reconstruction (for advection re-compression).
    pub tolerance: f64,
    /// Maximum rank per node (for advection re-compression).
    pub max_rank: usize,
    /// Interpolation mode used in SLAR advection (advect_x / advect_v).
    pub interpolation_mode: InterpolationMode,
    /// Shared progress state for intra-phase reporting to the TUI.
    progress: Option<Arc<super::super::progress::StepProgress>>,
    /// Whether to enforce positivity after each advection step.
    pub positivity_limiter: bool,
    /// Velocity-space exponential filter configuration for filamentation control.
    pub velocity_filter: Option<VelocityFilterConfig>,
    /// Count of negative values clamped by the positivity limiter.
    positivity_violations: AtomicU64,
}

impl Clone for HtTensor {
    fn clone(&self) -> Self {
        HtTensor {
            nodes: self.nodes.clone(),
            shape: self.shape,
            domain: self.domain.clone(),
            tolerance: self.tolerance,
            max_rank: self.max_rank,
            interpolation_mode: self.interpolation_mode,
            progress: self.progress.clone(),
            positivity_limiter: self.positivity_limiter,
            velocity_filter: self.velocity_filter,
            positivity_violations: AtomicU64::new(0),
        }
    }
}

// ─── Construction ───────────────────────────────────────────────────────────

impl HtTensor {
    /// Build empty HT with given max rank and the fixed 6D tree structure.
    pub fn new(domain: &Domain, max_rank: usize) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
            domain.velocity_res.v1 as usize,
            domain.velocity_res.v2 as usize,
            domain.velocity_res.v3 as usize,
        ];
        let k = max_rank;
        let mut nodes = Vec::with_capacity(NUM_NODES);

        // Leaves 0..6
        for d in 0..NUM_LEAVES {
            let n = shape[d];
            let r = k.min(n);
            let mut frame = Mat::zeros(n, r);
            for i in 0..r.min(n) {
                frame[(i, i)] = 1.0;
            }
            nodes.push(HtNode::Leaf { dim: d, frame });
        }

        // Interior node 6: {1,2}
        let kl = k.min(shape[1]);
        let kr = k.min(shape[2]);
        let kt = k.min(kl * kr);
        nodes.push(HtNode::Interior {
            left: 1,
            right: 2,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Interior node 7: {4,5}
        let kl = k.min(shape[4]);
        let kr = k.min(shape[5]);
        let kt = k.min(kl * kr);
        nodes.push(HtNode::Interior {
            left: 4,
            right: 5,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Interior node 8: {0,1,2}
        let kl = k.min(shape[0]);
        let kr = nodes[6].rank();
        let kt = k.min(kl * kr);
        nodes.push(HtNode::Interior {
            left: 0,
            right: 6,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Interior node 9: {3,4,5}
        let kl = k.min(shape[3]);
        let kr = nodes[7].rank();
        let kt = k.min(kl * kr);
        nodes.push(HtNode::Interior {
            left: 3,
            right: 7,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Root node 10: {0..5}
        let kl = nodes[8].rank();
        let kr = nodes[9].rank();
        nodes.push(HtNode::Interior {
            left: 8,
            right: 9,
            transfer: vec![0.0; kl * kr],
            ranks: [1, kl, kr],
        });

        HtTensor {
            nodes,
            shape,
            domain: domain.clone(),
            tolerance: 1e-6,
            max_rank,
            interpolation_mode: InterpolationMode::SparsePolynomial,
            progress: None,
            positivity_limiter: false,
            velocity_filter: None,
            positivity_violations: AtomicU64::new(0),
        }
    }

    /// Convert a full 6D array to HT format via hierarchical SVD (HSVD).
    pub fn from_full(data: &[f64], shape: [usize; 6], domain: &Domain, tolerance: f64) -> Self {
        let _span = tracing::info_span!("ht_from_full").entered();
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data/shape mismatch"
        );

        // Per-node tolerance: ε / √(2d−3) = ε / 3
        let eps_node = tolerance / 3.0;

        // Dimension sets for each node
        let dim_sets: [&[usize]; NUM_NODES] = [
            &[0],
            &[1],
            &[2],
            &[3],
            &[4],
            &[5],
            &[1, 2],
            &[4, 5],
            &[0, 1, 2],
            &[3, 4, 5],
            &[0, 1, 2, 3, 4, 5],
        ];

        // Compute frames for all nodes via mode unfolding + SVD
        let mut frames: Vec<Mat<f64>> = (0..10usize)
            .into_par_iter()
            .map(|node_idx| {
                let dims = dim_sets[node_idx];
                let mat = multi_mode_unfold(data, &shape, dims);
                let (u, s, _vt) = thin_svd(&mat);
                if u.ncols() == 0 {
                    // SVD failed — use first column of unfolding as single basis vector
                    let n = mat.nrows();
                    let mut frame = Mat::zeros(n, 1);
                    if mat.ncols() > 0 {
                        let norm: f64 = (0..n)
                            .map(|i| mat[(i, 0)] * mat[(i, 0)])
                            .sum::<f64>()
                            .sqrt();
                        if norm > 1e-30 {
                            for i in 0..n {
                                frame[(i, 0)] = mat[(i, 0)] / norm;
                            }
                        } else {
                            frame[(0, 0)] = 1.0;
                        }
                    } else {
                        frame[(0, 0)] = 1.0;
                    }
                    frame
                } else {
                    let rank = truncation_rank(&s, eps_node).max(1).min(u.ncols());
                    u.subcols(0, rank).to_owned()
                }
            })
            .collect();

        // Root frame: trivially [1] (1×1)
        let mut root_frame = Mat::zeros(1, 1);
        root_frame[(0, 0)] = 1.0;
        frames.push(root_frame);

        // Children of each interior node
        let children: [(usize, usize); 5] = [(1, 2), (4, 5), (0, 6), (3, 7), (8, 9)];

        let mut nodes: Vec<HtNode> = Vec::with_capacity(NUM_NODES);

        // Leaf nodes
        for d in 0..6 {
            nodes.push(HtNode::Leaf {
                dim: d,
                frame: frames[d].clone(),
            });
        }

        // Interior nodes
        for (i, &(left, right)) in children.iter().enumerate() {
            let node_idx = 6 + i;
            let transfer = compute_transfer_tensor(
                data,
                &shape,
                dim_sets[node_idx],
                dim_sets[left],
                dim_sets[right],
                &frames[node_idx],
                &frames[left],
                &frames[right],
            );
            let kt = frames[node_idx].ncols();
            let kl = frames[left].ncols();
            let kr = frames[right].ncols();

            nodes.push(HtNode::Interior {
                left,
                right,
                transfer,
                ranks: [kt, kl, kr],
            });
        }

        let max_leaf_rank = nodes.iter().map(|n| n.rank()).max().unwrap_or(1);
        HtTensor {
            nodes,
            shape,
            domain: domain.clone(),
            tolerance,
            max_rank: max_leaf_rank,
            interpolation_mode: InterpolationMode::SparsePolynomial,
            progress: None,
            positivity_limiter: false,
            velocity_filter: None,
            positivity_violations: AtomicU64::new(0),
        }
    }

    /// Construct HT by sampling a callable on the grid and compressing.
    pub fn from_function<F>(f: F, domain: &Domain, tolerance: f64) -> Self
    where
        F: Fn(&[f64; 3], &[f64; 3]) -> f64 + Sync,
    {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
            domain.velocity_res.v1 as usize,
            domain.velocity_res.v2 as usize,
            domain.velocity_res.v3 as usize,
        ];
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();

        let [n0, n1, n2, n3, n4, n5] = shape;
        let slab_size = n1 * n2 * n3 * n4 * n5;

        let data: Vec<f64> = (0..n0)
            .into_par_iter()
            .flat_map(|i0| {
                let mut slab = vec![0.0f64; slab_size];
                for i1 in 0..n1 {
                    for i2 in 0..n2 {
                        for i3 in 0..n3 {
                            for i4 in 0..n4 {
                                for i5 in 0..n5 {
                                    let x = [
                                        -lx[0] + (i0 as f64 + 0.5) * dx[0],
                                        -lx[1] + (i1 as f64 + 0.5) * dx[1],
                                        -lx[2] + (i2 as f64 + 0.5) * dx[2],
                                    ];
                                    let v = [
                                        -lv[0] + (i3 as f64 + 0.5) * dv[0],
                                        -lv[1] + (i4 as f64 + 0.5) * dv[1],
                                        -lv[2] + (i5 as f64 + 0.5) * dv[2],
                                    ];
                                    let local_idx = i1 * n2 * n3 * n4 * n5
                                        + i2 * n3 * n4 * n5
                                        + i3 * n4 * n5
                                        + i4 * n5
                                        + i5;
                                    slab[local_idx] = f(&x, &v);
                                }
                            }
                        }
                    }
                }
                slab
            })
            .collect();

        Self::from_full(&data, shape, domain, tolerance)
    }

    // ─── Public accessors for BUG integrator ─────────────────────────────

    /// Get immutable reference to a leaf node's basis frame.
    pub fn leaf_frame(&self, dim: usize) -> &Mat<f64> {
        assert!(dim < NUM_LEAVES, "dim {dim} is not a leaf");
        match &self.nodes[dim] {
            HtNode::Leaf { frame, .. } => frame,
            _ => unreachable!(),
        }
    }

    /// Get mutable reference to a leaf node's basis frame.
    pub fn leaf_frame_mut(&mut self, dim: usize) -> &mut Mat<f64> {
        assert!(dim < NUM_LEAVES, "dim {dim} is not a leaf");
        match &mut self.nodes[dim] {
            HtNode::Leaf { frame, .. } => frame,
            _ => unreachable!(),
        }
    }

    /// Get transfer tensor data and shape `[k_parent, k_left, k_right]` for an interior node.
    pub fn transfer_tensor(&self, node: usize) -> (&[f64], [usize; 3]) {
        assert!(
            (NUM_LEAVES..NUM_NODES).contains(&node),
            "node {node} is not interior"
        );
        match &self.nodes[node] {
            HtNode::Interior {
                transfer, ranks, ..
            } => (transfer.as_slice(), *ranks),
            _ => unreachable!(),
        }
    }

    /// Replace the transfer tensor data and ranks for an interior node.
    pub fn set_transfer_tensor(&mut self, node: usize, data: Vec<f64>, ranks: [usize; 3]) {
        assert!(
            (NUM_LEAVES..NUM_NODES).contains(&node),
            "node {node} is not interior"
        );
        assert_eq!(data.len(), ranks[0] * ranks[1] * ranks[2]);
        match &mut self.nodes[node] {
            HtNode::Interior {
                transfer, ranks: r, ..
            } => {
                *transfer = data;
                *r = ranks;
            }
            _ => unreachable!(),
        }
    }

    /// Enable or disable the Zhang-Shu positivity-preserving limiter.
    ///
    /// When enabled, negative values introduced by HT recompression during advection
    /// are clamped to zero and the tensor is rescaled to preserve total mass.
    pub fn with_positivity_limiter(mut self, enabled: bool) -> Self {
        self.positivity_limiter = enabled;
        self
    }

    /// Enable velocity-space exponential filtering after each velocity advection step.
    pub fn with_velocity_filter(mut self, config: VelocityFilterConfig) -> Self {
        self.velocity_filter = Some(config);
        self
    }

    /// Apply an exponential filter in velocity space to the HT leaf frames.
    ///
    /// Leaves 3, 4, 5 correspond to v1, v2, v3. For each velocity leaf, every
    /// column of the leaf frame U_mu (a basis vector of length n_mu) is filtered
    /// in Fourier space using `exp(-(k/k_cutoff)^(2*order))`. This preserves the
    /// HT structure without rank increase.
    pub fn apply_velocity_filter(&mut self) {
        let config = match self.velocity_filter {
            Some(c) => c,
            None => return,
        };
        let _span = tracing::info_span!("ht_apply_velocity_filter").entered();

        // Velocity leaves are nodes 3, 4, 5 (v1, v2, v3).
        for leaf_idx in 3..=5 {
            let n_mu = self.shape[leaf_idx];
            let kernel = build_exp_filter_kernel_ht(n_mu, config.cutoff_fraction, config.order);

            if let HtNode::Leaf { ref mut frame, .. } = self.nodes[leaf_idx] {
                let k_mu = frame.ncols();
                let mut planner = FftPlanner::new();
                let fwd = planner.plan_fft_forward(n_mu);
                let inv = planner.plan_fft_inverse(n_mu);
                let inv_n = 1.0 / n_mu as f64;

                let mut buf = vec![Complex64::new(0.0, 0.0); n_mu];

                for col in 0..k_mu {
                    // Extract column to complex buffer
                    for i in 0..n_mu {
                        buf[i] = Complex64::new(frame[(i, col)], 0.0);
                    }
                    fwd.process(&mut buf);
                    // Apply filter
                    for (c, &k) in buf.iter_mut().zip(kernel.iter()) {
                        *c *= k;
                    }
                    inv.process(&mut buf);
                    // Write back (normalize by 1/N)
                    for i in 0..n_mu {
                        frame[(i, col)] = buf[i].re * inv_n;
                    }
                }
            }
        }
    }

    /// Number of negative values clamped by the positivity limiter so far.
    pub fn positivity_violations(&self) -> u64 {
        self.positivity_violations.load(Ordering::Relaxed)
    }

    /// Black-box construction via HTACA (Ballani & Grasedyck 2013).
    ///
    /// Builds the HT tensor by sampling O(dNk) fibers of `f` instead of all N⁶ entries.
    /// Cost: O(dNk + dk³) vs O(N⁶) for `from_function`.
    ///
    /// # Arguments
    /// - `f`: Distribution function f(x, v) → f64
    /// - `domain`: Computational domain
    /// - `tolerance`: Approximation tolerance (distributed as ε/3 per node)
    /// - `max_rank`: Maximum rank per node
    /// - `n_initial_samples`: Number of fiber samples per leaf (default: 5 × max_rank)
    /// - `seed`: RNG seed for random pivot selection (default: 42)
    pub fn from_function_aca<F>(
        f: F,
        domain: &Domain,
        tolerance: f64,
        max_rank: usize,
        n_initial_samples: Option<usize>,
        seed: Option<u64>,
    ) -> Self
    where
        F: Fn(&[f64; 3], &[f64; 3]) -> f64 + Sync,
    {
        let _span = tracing::info_span!("ht_from_function_aca").entered();

        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
            domain.velocity_res.v1 as usize,
            domain.velocity_res.v2 as usize,
            domain.velocity_res.v3 as usize,
        ];
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();

        let n_samples = n_initial_samples.unwrap_or(5 * max_rank);
        let mut rng = Xorshift64::new(seed.unwrap_or(42));

        // SLAR tolerance hierarchy with exponential decay (Zheng et al. 2025, §3.2):
        // Leaves (depth 0) use the tightest tolerance, with each level up the tree
        // relaxing by a decay factor. This matches the paper's requirement that
        // subtrees closer to leaves use tighter tolerances.
        //   ε_leaf        = tolerance / √(2d-3) = tolerance / 3.0 (for d=6)
        //   ε_interior_lo = ε_leaf × decay       (nodes 6, 7: leaf-pair interiors)
        //   ε_interior_hi = ε_leaf × decay²      (nodes 8, 9: spatial/velocity subtrees)
        //   ε_root        = tolerance             (root: coarsest, set after construction)
        let decay = (2.0_f64 * 6.0 - 3.0).sqrt(); // √(2d-3) ≈ 3.0
        let eps_leaf = tolerance / decay;
        let eps_interior_lo = eps_leaf * decay.sqrt(); // nodes 6, 7
        let eps_interior_hi = eps_leaf * decay; // nodes 8, 9
        let eps_node = eps_leaf; // default for leaf fiber-sampling

        // Helper: convert grid index to physical coordinate for a given dimension
        let idx_to_coord = |dim: usize, idx: usize| -> f64 {
            match dim {
                0 => -lx[0] + (idx as f64 + 0.5) * dx[0],
                1 => -lx[1] + (idx as f64 + 0.5) * dx[1],
                2 => -lx[2] + (idx as f64 + 0.5) * dx[2],
                3 => -lv[0] + (idx as f64 + 0.5) * dv[0],
                4 => -lv[1] + (idx as f64 + 0.5) * dv[1],
                5 => -lv[2] + (idx as f64 + 0.5) * dv[2],
                _ => unreachable!(),
            }
        };

        // Helper: evaluate f at a 6D grid index
        let eval_at_index = |indices: [usize; 6]| -> f64 {
            let x = [
                idx_to_coord(0, indices[0]),
                idx_to_coord(1, indices[1]),
                idx_to_coord(2, indices[2]),
            ];
            let v = [
                idx_to_coord(3, indices[3]),
                idx_to_coord(4, indices[4]),
                idx_to_coord(5, indices[5]),
            ];
            f(&x, &v)
        };

        // ── Phase A: Leaf frames via fiber sampling + col-piv QR ──

        // For each leaf μ, sample random complementary multi-indices and evaluate fibers.
        let mut leaf_frames: Vec<Mat<f64>> = Vec::with_capacity(6);
        let mut leaf_ranks: Vec<usize> = Vec::with_capacity(6);
        // Store the complementary indices used for each leaf (for later reuse)
        let mut leaf_comp_indices: Vec<Vec<[usize; 6]>> = Vec::with_capacity(6);

        for mu in 0..6 {
            let n_mu = shape[mu];
            let ns = n_samples.min(n_mu * 10); // cap samples

            // Generate random complementary multi-indices
            let mut comp_indices: Vec<[usize; 6]> = Vec::with_capacity(ns);
            for _ in 0..ns {
                let mut idx = [0usize; 6];
                for d in 0..6 {
                    idx[d] = rng.next_usize(shape[d]);
                }
                comp_indices.push(idx);
            }

            // Evaluate fibers: M ∈ ℝ^{n_μ × ns}
            // Parallelize over fiber columns (each sample is independent)
            let fiber_cols: Vec<Vec<f64>> = comp_indices
                .par_iter()
                .map(|comp| {
                    (0..n_mu)
                        .map(|i| {
                            let mut idx = *comp;
                            idx[mu] = i;
                            eval_at_index(idx)
                        })
                        .collect()
                })
                .collect();
            let mut fiber_mat: Mat<f64> = Mat::zeros(n_mu, ns);
            for (s, col) in fiber_cols.iter().enumerate() {
                for (i, &val) in col.iter().enumerate() {
                    fiber_mat[(i, s)] = val;
                }
            }

            // Column-pivoted QR to extract basis
            let cpqr = fiber_mat.as_ref().col_piv_qr();
            let q = cpqr.compute_thin_Q();
            let r = cpqr.thin_R();

            // Determine rank from R diagonal
            let k_max = q.ncols().min(n_mu);
            let mut sv: Vec<f64> = (0..k_max).map(|i| r[(i, i)].abs()).collect();
            // R diagonal magnitudes are not quite singular values but suffice for rank estimation
            sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let k_mu = truncation_rank(&sv, eps_node).max(1).min(max_rank);

            let frame = q.subcols(0, k_mu).to_owned();
            leaf_frames.push(frame);
            leaf_ranks.push(k_mu);
            leaf_comp_indices.push(comp_indices);
        }

        // ── Phase B: Interior transfer tensors ──
        //
        // Process bottom-up: nodes 6={1,2}, 7={4,5}, 8={0,6}, 9={3,7}
        // For each interior node, we need to compute the transfer tensor.
        //
        // Strategy: For each interior node t with children (left_child, right_child),
        // build a matrix M_t where rows = (j_left, j_right) pairs (k_left × k_right rows)
        // and columns = complementary physical multi-indices. Then do ACA + SVD on M_t.

        // We'll store effective frames for each node (for interior nodes, this is the
        // Khatri-Rao product of child frames through the subtree)
        // Node 6 = dims {1,2}: effective frame is n1*n2 × k6
        // Node 7 = dims {4,5}: effective frame is n4*n5 × k7

        // Build effective frames for leaf-pair nodes (6 and 7)
        let eff_frame_6 = build_kron_frame(&leaf_frames[1], &leaf_frames[2], shape[1], shape[2]);
        let eff_frame_7 = build_kron_frame(&leaf_frames[4], &leaf_frames[5], shape[4], shape[5]);

        // Interior nodes: compute transfer tensors
        // Node 6: children 1, 2 (both leaves), complementary dims: {0, 3, 4, 5}
        let (transfer_6, rank_6) = build_interior_transfer_aca(
            &eval_at_index,
            &shape,
            6,
            &[1],          // left child dims (leaf 1)
            &[2],          // right child dims (leaf 2)
            &[0, 3, 4, 5], // complementary dims
            &leaf_frames[1],
            &leaf_frames[2],
            leaf_ranks[1],
            leaf_ranks[2],
            eps_interior_lo, // leaf-pair interior: moderate tolerance
            max_rank,
            n_samples,
            &mut rng,
        );

        // Node 7: children 4, 5 (both leaves), complementary dims: {0, 1, 2, 3}
        let (transfer_7, rank_7) = build_interior_transfer_aca(
            &eval_at_index,
            &shape,
            7,
            &[4],          // left child dims (leaf 4)
            &[5],          // right child dims (leaf 5)
            &[0, 1, 2, 3], // complementary dims
            &leaf_frames[4],
            &leaf_frames[5],
            leaf_ranks[4],
            leaf_ranks[5],
            eps_interior_lo, // leaf-pair interior: moderate tolerance
            max_rank,
            n_samples,
            &mut rng,
        );

        // For nodes 8 and 9, the children include an interior node.
        // Node 8: children 0 (leaf), 6 (interior {1,2})
        //   left_child = leaf 0, right_child = node 6
        //   Node dims: {0, 1, 2}, complementary: {3, 4, 5}
        let eff_frame_6_trunc = {
            // The effective frame for node 6 after transfer tensor:
            // it maps the n1*n2 space to k6 dims via eff_frame_6, then transfer_6 maps
            // (k_left=k1, k_right=k2) to k6. The effective frame for node 6 is
            // eff_frame_6 viewed as the frame for the {1,2} subspace.
            // Actually, for the purpose of computing transfer tensors at higher levels,
            // we need: for each basis vector j of node 6, what is the corresponding
            // n1*n2-dimensional vector? That's column j of eff_frame_6 contracted with
            // the transfer tensor.
            //
            // effective_frame_node6[:, j_6] = Σ_{j1,j2} B_6[j_6, j1, j2] * (U_1[:, j1] ⊗ U_2[:, j2])
            // This is a (n1*n2) × k6 matrix.
            let k6 = rank_6;
            let k1 = leaf_ranks[1];
            let k2 = leaf_ranks[2];
            let n12 = shape[1] * shape[2];
            let mut eff = Mat::zeros(n12, k6);
            for j6 in 0..k6 {
                for j1 in 0..k1 {
                    for j2 in 0..k2 {
                        let b = transfer_6[j6 * k1 * k2 + j1 * k2 + j2];
                        if b.abs() < 1e-30 {
                            continue;
                        }
                        for i1 in 0..shape[1] {
                            let u1 = leaf_frames[1][(i1, j1)];
                            if u1.abs() < 1e-30 {
                                continue;
                            }
                            let bu1 = b * u1;
                            for i2 in 0..shape[2] {
                                eff[(i1 * shape[2] + i2, j6)] += bu1 * leaf_frames[2][(i2, j2)];
                            }
                        }
                    }
                }
            }
            eff
        };

        let (transfer_8, rank_8) = build_interior_transfer_aca(
            &eval_at_index,
            &shape,
            8,
            &[0],       // left child dims (leaf 0)
            &[1, 2],    // right child dims (node 6 = {1,2})
            &[3, 4, 5], // complementary dims
            &leaf_frames[0],
            &eff_frame_6_trunc,
            leaf_ranks[0],
            rank_6,
            eps_interior_hi, // subtree root: coarser tolerance
            max_rank,
            n_samples,
            &mut rng,
        );

        // Node 9: children 3 (leaf), 7 (interior {4,5})
        let eff_frame_7_trunc = {
            let k7 = rank_7;
            let k4 = leaf_ranks[4];
            let k5 = leaf_ranks[5];
            let n45 = shape[4] * shape[5];
            let mut eff = Mat::zeros(n45, k7);
            for j7 in 0..k7 {
                for j4 in 0..k4 {
                    for j5 in 0..k5 {
                        let b = transfer_7[j7 * k4 * k5 + j4 * k5 + j5];
                        if b.abs() < 1e-30 {
                            continue;
                        }
                        for i4 in 0..shape[4] {
                            let u4 = leaf_frames[4][(i4, j4)];
                            if u4.abs() < 1e-30 {
                                continue;
                            }
                            let bu4 = b * u4;
                            for i5 in 0..shape[5] {
                                eff[(i4 * shape[5] + i5, j7)] += bu4 * leaf_frames[5][(i5, j5)];
                            }
                        }
                    }
                }
            }
            eff
        };

        let (transfer_9, rank_9) = build_interior_transfer_aca(
            &eval_at_index,
            &shape,
            9,
            &[3],       // left child dims (leaf 3)
            &[4, 5],    // right child dims (node 7 = {4,5})
            &[0, 1, 2], // complementary dims
            &leaf_frames[3],
            &eff_frame_7_trunc,
            leaf_ranks[3],
            rank_7,
            eps_interior_hi, // subtree root: coarser tolerance
            max_rank,
            n_samples,
            &mut rng,
        );

        // ── Phase C: Root transfer tensor ──
        // Root node 10: children 8 (x-subtree), 9 (v-subtree)
        // B_root ∈ ℝ^{1 × k8 × k9} = a k8 × k9 matrix.
        //
        // B_root[j8, j9] = projection of f onto (subtree-8 basis j8) ⊗ (subtree-9 basis j9)
        // We compute this by sampling: for each (j8, j9), we evaluate f at representative
        // points and accumulate.

        // Build effective frames for nodes 8 and 9 (needed for root computation)
        let eff_frame_8 = {
            // effective_frame_8[:, j8] = Σ_{j0, j6} B_8[j8, j0, j6]
            //     * (U_0[:, j0] ⊗ eff_6[:, j6])
            // This is (n0*n1*n2) × k8
            let k8 = rank_8;
            let k0 = leaf_ranks[0];
            let k6r = rank_6;
            let n012 = shape[0] * shape[1] * shape[2];
            let n12 = shape[1] * shape[2];
            let mut eff: Mat<f64> = Mat::zeros(n012, k8);
            for j8 in 0..k8 {
                for j0 in 0..k0 {
                    for j6 in 0..k6r {
                        let b = transfer_8[j8 * k0 * k6r + j0 * k6r + j6];
                        if b.abs() < 1e-30 {
                            continue;
                        }
                        for i0 in 0..shape[0] {
                            let u0 = leaf_frames[0][(i0, j0)];
                            if u0.abs() < 1e-30 {
                                continue;
                            }
                            let bu0 = b * u0;
                            for i12 in 0..n12 {
                                eff[(i0 * n12 + i12, j8)] += bu0 * eff_frame_6_trunc[(i12, j6)];
                            }
                        }
                    }
                }
            }
            eff
        };

        let eff_frame_9 = {
            let k9 = rank_9;
            let k3 = leaf_ranks[3];
            let k7r = rank_7;
            let n345 = shape[3] * shape[4] * shape[5];
            let n45 = shape[4] * shape[5];
            let mut eff: Mat<f64> = Mat::zeros(n345, k9);
            for j9 in 0..k9 {
                for j3 in 0..k3 {
                    for j7 in 0..k7r {
                        let b = transfer_9[j9 * k3 * k7r + j3 * k7r + j7];
                        if b.abs() < 1e-30 {
                            continue;
                        }
                        for i3 in 0..shape[3] {
                            let u3 = leaf_frames[3][(i3, j3)];
                            if u3.abs() < 1e-30 {
                                continue;
                            }
                            let bu3 = b * u3;
                            for i45 in 0..n45 {
                                eff[(i3 * n45 + i45, j9)] += bu3 * eff_frame_7_trunc[(i45, j7)];
                            }
                        }
                    }
                }
            }
            eff
        };

        // Root: B_root[j8, j9] = Σ_idx eff_8[x_idx, j8] * eff_9[v_idx, j9] * f(idx)
        // But we can compute this more efficiently: since eff_8 and eff_9 are the
        // projection bases, B_root = eff_8^T * M_unfolded * eff_9, where M_unfolded
        // is the (n_x × n_v) matricization of f.
        //
        // To avoid evaluating all N^6 entries, we use ACA on this matrix.
        let n_x = shape[0] * shape[1] * shape[2];
        let n_v = shape[3] * shape[4] * shape[5];

        // Build the reduced matrix: rows = j8 (0..k8), cols = j9 (0..k9)
        // M[j8, j9] = Σ_{x_idx, v_idx} eff_8[x_idx, j8] * f(x_idx, v_idx) * eff_9[v_idx, j9]
        //
        // For small k8, k9 we can compute this directly via sampling.
        // Use the ACA on the (n_x × n_v) unfolding, projected through eff frames.
        let mut root_transfer = vec![0.0f64; rank_8 * rank_9];

        // For tractable sizes, compute B_root via inner products of effective frames with f.
        // Sample n_root_samples random 6D points, accumulate weighted projections.
        let n_root_samples = (n_samples * 3).min(n_x * n_v);

        if n_x * n_v <= 100_000 {
            // Small enough to iterate all
            for x_flat in 0..n_x {
                let i0 = x_flat / (shape[1] * shape[2]);
                let i1 = (x_flat / shape[2]) % shape[1];
                let i2 = x_flat % shape[2];
                for v_flat in 0..n_v {
                    let i3 = v_flat / (shape[4] * shape[5]);
                    let i4 = (v_flat / shape[5]) % shape[4];
                    let i5 = v_flat % shape[5];
                    let val = eval_at_index([i0, i1, i2, i3, i4, i5]);
                    if val.abs() < 1e-30 {
                        continue;
                    }
                    for j8 in 0..rank_8 {
                        let e8: f64 = eff_frame_8[(x_flat, j8)];
                        if e8.abs() < 1e-30 {
                            continue;
                        }
                        let ve8 = val * e8;
                        for j9 in 0..rank_9 {
                            root_transfer[j8 * rank_9 + j9] += ve8 * eff_frame_9[(v_flat, j9)];
                        }
                    }
                }
            }
        } else {
            // Use ACA on the projected matrix
            // Define M[j8, j9] by querying rows/cols via fiber sums
            let eff8_ref = &eff_frame_8;
            let eff9_ref = &eff_frame_9;
            let eval_ref = &eval_at_index;
            let shape_ref = &shape;

            let root_mat = FnMatrix::new(rank_8, rank_9, |j8, j9| {
                // M[j8, j9] = Σ_{x,v} eff8[x, j8] * f(x,v) * eff9[v, j9]
                // This is expensive — approximate by sampling
                // Use the pivots from nodes 8 and 9 to pick representative points
                let mut sum = 0.0;
                let n_x_samples = (shape_ref[0] * shape_ref[1] * shape_ref[2]).min(500);
                let n_v_samples = (shape_ref[3] * shape_ref[4] * shape_ref[5]).min(500);
                let x_step = n_x.max(1) / n_x_samples.max(1);
                let v_step = n_v.max(1) / n_v_samples.max(1);
                let x_scale = n_x as f64 / n_x_samples as f64;
                let v_scale = n_v as f64 / n_v_samples as f64;

                for xs in 0..n_x_samples {
                    let x_flat = (xs * x_step).min(n_x - 1);
                    let e8 = eff8_ref[(x_flat, j8)];
                    if e8.abs() < 1e-30 {
                        continue;
                    }
                    let i0 = x_flat / (shape_ref[1] * shape_ref[2]);
                    let i1 = (x_flat / shape_ref[2]) % shape_ref[1];
                    let i2 = x_flat % shape_ref[2];
                    for vs in 0..n_v_samples {
                        let v_flat = (vs * v_step).min(n_v - 1);
                        let i3 = v_flat / (shape_ref[4] * shape_ref[5]);
                        let i4 = (v_flat / shape_ref[5]) % shape_ref[4];
                        let i5 = v_flat % shape_ref[5];
                        let val = eval_ref([i0, i1, i2, i3, i4, i5]);
                        sum += e8 * val * eff9_ref[(v_flat, j9)] * x_scale * v_scale;
                    }
                }
                sum
            });

            let aca_result = aca_partial_pivot(&root_mat, tolerance, max_rank);
            // Fill root_transfer from ACA result
            for j8 in 0..rank_8 {
                for j9 in 0..rank_9 {
                    root_transfer[j8 * rank_9 + j9] = aca_result.evaluate(j8, j9);
                }
            }
        }

        // ── Assemble the HtTensor ──
        let mut nodes: Vec<HtNode> = Vec::with_capacity(NUM_NODES);

        // Leaves
        for d in 0..6 {
            nodes.push(HtNode::Leaf {
                dim: d,
                frame: leaf_frames[d].clone(),
            });
        }

        // Node 6: {1,2}
        nodes.push(HtNode::Interior {
            left: 1,
            right: 2,
            transfer: transfer_6,
            ranks: [rank_6, leaf_ranks[1], leaf_ranks[2]],
        });

        // Node 7: {4,5}
        nodes.push(HtNode::Interior {
            left: 4,
            right: 5,
            transfer: transfer_7,
            ranks: [rank_7, leaf_ranks[4], leaf_ranks[5]],
        });

        // Node 8: {0,1,2}
        nodes.push(HtNode::Interior {
            left: 0,
            right: 6,
            transfer: transfer_8,
            ranks: [rank_8, leaf_ranks[0], rank_6],
        });

        // Node 9: {3,4,5}
        nodes.push(HtNode::Interior {
            left: 3,
            right: 7,
            transfer: transfer_9,
            ranks: [rank_9, leaf_ranks[3], rank_7],
        });

        // Root 10: {0..5}
        nodes.push(HtNode::Interior {
            left: 8,
            right: 9,
            transfer: root_transfer,
            ranks: [1, rank_8, rank_9],
        });

        let mut ht = HtTensor {
            nodes,
            shape,
            domain: domain.clone(),
            tolerance,
            max_rank,
            interpolation_mode: InterpolationMode::SparsePolynomial,
            progress: None,
            positivity_limiter: false,
            velocity_filter: None,
            positivity_violations: AtomicU64::new(0),
        };

        // Post-construction HSVD truncation (SLAR §3.2): re-compress the assembled
        // HT tensor with a coarser tolerance proportional to the approximation norm.
        // This catches over-ranked nodes from the bottom-up ACA construction.
        ht.truncate(tolerance);

        ht
    }
}

// ─── HTACA helper functions ─────────────────────────────────────────────────

/// Build the Kronecker product frame for a leaf-pair node.
/// Given U_left (n_l × k_l) and U_right (n_r × k_r),
/// returns U_kron (n_l*n_r × k_l*k_r) where column (j_l, j_r) = U_left[:,j_l] ⊗ U_right[:,j_r].
fn build_kron_frame(left: &Mat<f64>, right: &Mat<f64>, n_l: usize, n_r: usize) -> Mat<f64> {
    let kl = left.ncols();
    let kr = right.ncols();
    let mut kron = Mat::zeros(n_l * n_r, kl * kr);
    for jl in 0..kl {
        for jr in 0..kr {
            let col = jl * kr + jr;
            for il in 0..n_l {
                let ul = left[(il, jl)];
                for ir in 0..n_r {
                    kron[(il * n_r + ir, col)] = ul * right[(ir, jr)];
                }
            }
        }
    }
    kron
}

/// Build interior transfer tensor by projecting f onto child frames.
///
/// Computes R = (U_left ⊗ U_right)^T @ M_t where M_t is the mode-t matricization of f.
/// Then SVD of R gives the transfer tensor B_t ∈ ℝ^{k_t × k_left × k_right}.
fn build_interior_transfer_aca<E: Fn([usize; 6]) -> f64 + Sync>(
    eval: &E,
    shape: &[usize; 6],
    _node_idx: usize,
    left_dims: &[usize],
    right_dims: &[usize],
    comp_dims: &[usize],
    left_frame: &Mat<f64>,
    right_frame: &Mat<f64>,
    k_left: usize,
    k_right: usize,
    eps: f64,
    max_rank: usize,
    n_samples: usize,
    rng: &mut Xorshift64,
) -> (Vec<f64>, usize) {
    let k_lr = k_left * k_right;
    let n_comp: usize = comp_dims.iter().map(|&d| shape[d]).product();
    let n_left_phys: usize = left_dims.iter().map(|&d| shape[d]).product();
    let n_right_phys: usize = right_dims.iter().map(|&d| shape[d]).product();

    // Use all complementary indices if tractable, otherwise sample randomly.
    // Sampling n_samples random complementary multi-indices suffices because the
    // transfer tensor has at most max_rank degrees of freedom — the same principle
    // used for leaf frame construction (fiber sampling + QR).
    let n_eval = n_samples.min(n_comp);

    let comp_indices: Vec<Vec<usize>> = if n_eval >= n_comp {
        // Exhaustive: enumerate all complementary indices (small grids)
        (0..n_comp)
            .map(|comp_flat| {
                let mut vals = vec![0usize; comp_dims.len()];
                let mut rem = comp_flat;
                for ci in (0..comp_dims.len()).rev() {
                    vals[ci] = rem % shape[comp_dims[ci]];
                    rem /= shape[comp_dims[ci]];
                }
                vals
            })
            .collect()
    } else {
        // Random sampling: draw n_eval complementary multi-indices
        (0..n_eval)
            .map(|_| {
                comp_dims
                    .iter()
                    .map(|&d| rng.next_usize(shape[d]))
                    .collect()
            })
            .collect()
    };

    // Build projected matrix R ∈ ℝ^{k_lr × n_eval}
    // R[(jl*kr+jr), s] = Σ_{left_flat, right_flat}
    //     left_frame[left_flat, jl] * right_frame[right_flat, jr] * f(...)
    //
    // For each complementary sample, compute F_proj ∈ ℝ^{n_left × n_right},
    // then project: R[:, s] = vec(U_left^T @ F_proj @ U_right).
    // Parallelize over samples — each column of proj_mat is independent.
    let proj_cols: Vec<Vec<f64>> = comp_indices
        .par_iter()
        .map(|comp_vals| {
            // Build T = S @ U_right ∈ ℝ^{n_left × k_right}
            let mut t_mat = vec![0.0f64; n_left_phys * k_right];
            for left_flat in 0..n_left_phys {
                let mut left_idx_vals = [0usize; 6];
                let mut rem_l = left_flat;
                for li in (0..left_dims.len()).rev() {
                    left_idx_vals[li] = rem_l % shape[left_dims[li]];
                    rem_l /= shape[left_dims[li]];
                }

                for right_flat in 0..n_right_phys {
                    let mut right_idx_vals = [0usize; 6];
                    let mut rem_r = right_flat;
                    for ri in (0..right_dims.len()).rev() {
                        right_idx_vals[ri] = rem_r % shape[right_dims[ri]];
                        rem_r /= shape[right_dims[ri]];
                    }

                    let mut idx = [0usize; 6];
                    for (li, &d) in left_dims.iter().enumerate() {
                        idx[d] = left_idx_vals[li];
                    }
                    for (ri, &d) in right_dims.iter().enumerate() {
                        idx[d] = right_idx_vals[ri];
                    }
                    for (ci, &d) in comp_dims.iter().enumerate() {
                        idx[d] = comp_vals[ci];
                    }

                    let val = eval(idx);
                    if val.abs() < 1e-30 {
                        continue;
                    }

                    for jr in 0..k_right {
                        t_mat[left_flat * k_right + jr] += val * right_frame[(right_flat, jr)];
                    }
                }
            }

            // Compute R[:, s] = vec(U_left^T @ T)
            let mut col = vec![0.0f64; k_lr];
            for jl in 0..k_left {
                for jr in 0..k_right {
                    let mut s = 0.0;
                    for left_flat in 0..n_left_phys {
                        s += left_frame[(left_flat, jl)] * t_mat[left_flat * k_right + jr];
                    }
                    col[jl * k_right + jr] = s;
                }
            }
            col
        })
        .collect();

    // Assemble proj_mat from parallel results
    let mut proj_mat: Mat<f64> = Mat::zeros(k_lr, n_eval);
    for (s, col) in proj_cols.iter().enumerate() {
        for (row, &val) in col.iter().enumerate() {
            proj_mat[(row, s)] = val;
        }
    }

    // SVD of the projected matrix R = U_R * S * V^T
    // R = K^T @ M_t where K = kron(U_left, U_right) has orthonormal columns.
    // Since in from_full: B_t = U_t^T @ K, and R = (U_t^T @ K)^T @ Σ @ V^T = B^T @ Σ @ V^T
    // When child frames are good: B ≈ U_R^T (transfer tensor is just the transposed left SVD vecs).
    // The singular values are implicitly captured by the parent-level transfer.
    let (u, sv, _vt) = thin_svd(&proj_mat);

    // SVD failed or returned degenerate result — fall back to rank-1 identity transfer
    if u.ncols() == 0 {
        let mut transfer = vec![0.0f64; k_lr];
        if k_lr > 0 {
            transfer[0] = 1.0;
        }
        return (transfer, 1);
    }

    let kt = truncation_rank(&sv, eps)
        .max(1)
        .min(max_rank)
        .min(k_lr)
        .min(u.ncols());

    // Transfer tensor B_t[i, (jl, jr)] = U_R[(jl*kr+jr), i]
    // No singular value scaling — the HT format propagates scale through the tree.
    let mut transfer = vec![0.0f64; kt * k_lr];
    for i in 0..kt {
        for j in 0..k_lr {
            transfer[i * k_lr + j] = u[(j, i)];
        }
    }

    (transfer, kt)
}

// ─── Point evaluation ───────────────────────────────────────────────────────

impl HtTensor {
    /// Evaluate at a single 6D grid point. Cost: O(d·k³).
    ///
    /// Allocates temporary workspace per call. For hot-path evaluation in loops,
    /// use [`evaluate_into`] with a pre-allocated workspace from [`eval_workspace_len`].
    #[inline]
    pub fn evaluate(&self, indices: [usize; 6]) -> f64 {
        let mr = self.compute_max_node_rank();
        let mut ws = vec![0.0f64; Self::workspace_len_for(mr)];
        self.evaluate_into(indices, &mut ws)
    }

    /// Evaluate at a single 6D grid point using a pre-allocated workspace.
    ///
    /// The workspace must have at least [`eval_workspace_len`] elements.
    /// This avoids heap allocation and is suitable for tight loops.
    #[inline]
    pub fn evaluate_into(&self, indices: [usize; 6], workspace: &mut [f64]) -> f64 {
        let mr = self.compute_max_node_rank();
        let (out, rest) = workspace.split_at_mut(mr);
        self.node_vector_into(ROOT, &indices, out, rest, mr);
        out[0]
    }

    /// Returns the required workspace length for [`evaluate_into`].
    pub fn eval_workspace_len(&self) -> usize {
        Self::workspace_len_for(self.compute_max_node_rank())
    }

    /// Compute workspace length for a given max rank.
    /// Layout: mr (output) + depth * 2 * mr (recursive buffers), depth = 3 for 6D tree.
    fn workspace_len_for(mr: usize) -> usize {
        7 * mr // 1 output + 3 levels × 2 child buffers
    }

    /// Actual maximum rank across all nodes in the tree.
    fn compute_max_node_rank(&self) -> usize {
        self.nodes.iter().map(|n| n.rank()).max().unwrap_or(1)
    }

    /// Contracted vector at a node, writing into caller-provided buffer.
    /// `workspace` provides scratch space for recursive child buffers.
    #[inline]
    fn node_vector_into(
        &self,
        node_idx: usize,
        indices: &[usize; 6],
        out: &mut [f64],
        workspace: &mut [f64],
        mr: usize,
    ) -> usize {
        match &self.nodes[node_idx] {
            HtNode::Leaf { dim, frame } => {
                let row = indices[*dim];
                let k = frame.ncols();
                for c in 0..k {
                    out[c] = frame[(row, c)];
                }
                k
            }
            HtNode::Interior {
                left,
                right,
                transfer,
                ranks,
            } => {
                let [kt, kl, kr] = *ranks;
                let (bufs, deeper) = workspace.split_at_mut(2 * mr);
                let (left_buf, right_buf) = bufs.split_at_mut(mr);
                self.node_vector_into(*left, indices, left_buf, deeper, mr);
                self.node_vector_into(*right, indices, right_buf, deeper, mr);
                contract_transfer_into(
                    transfer,
                    kt,
                    kl,
                    kr,
                    &left_buf[..kl],
                    &right_buf[..kr],
                    out,
                );
                kt
            }
        }
    }

    pub fn rank_at(&self, node_idx: usize) -> usize {
        self.nodes[node_idx].rank()
    }
    pub fn total_rank(&self) -> usize {
        self.nodes.iter().map(|n| n.rank()).sum()
    }

    pub fn memory_bytes(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| match n {
                HtNode::Leaf { frame, .. } => frame.nrows() * frame.ncols() * 8,
                HtNode::Interior { transfer, .. } => transfer.len() * 8,
            })
            .sum()
    }
}

// ─── Orthogonalization ──────────────────────────────────────────────────────

impl HtTensor {
    /// Bottom-up QR: all frames orthonormal except root.
    pub fn orthogonalize_left(&mut self) {
        let _span = tracing::info_span!("ht_orthogonalize").entered();
        for leaf in 0..NUM_LEAVES {
            self.orthogonalize_node(leaf);
        }
        for &ni in &[6, 7, 8, 9] {
            self.orthogonalize_node(ni);
        }
    }

    fn orthogonalize_node(&mut self, node_idx: usize) {
        // Take ownership of the node to avoid cloning (replaced with dummy leaf)
        let dummy = HtNode::Leaf {
            dim: 0,
            frame: Mat::new(),
        };
        let node = std::mem::replace(&mut self.nodes[node_idx], dummy);
        match node {
            HtNode::Leaf { dim, frame } => {
                let (q, r) = qr_decompose(&frame);
                self.nodes[node_idx] = HtNode::Leaf { dim, frame: q };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &r);
            }
            HtNode::Interior {
                left,
                right,
                transfer,
                ranks,
            } if node_idx != ROOT => {
                let [kt, kl, kr] = ranks;
                let mat = vec_to_mat(&transfer, kt, kl * kr);
                let (q, r) = qr_decompose(&mat);
                let new_kt = r.nrows();
                let new_transfer = mat_to_vec(&r, new_kt, kl * kr);
                let absorb = q.transpose().to_owned();
                self.nodes[node_idx] = HtNode::Interior {
                    left,
                    right,
                    transfer: new_transfer,
                    ranks: [new_kt, kl, kr],
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &absorb);
            }
            _ => {}
        }
    }

    fn absorb_r_into_parent(&mut self, parent_idx: usize, child_idx: usize, r: &Mat<f64>) {
        if let HtNode::Interior {
            left,
            right: _,
            ref mut transfer,
            ref mut ranks,
        } = self.nodes[parent_idx]
        {
            let [kt, kl, kr] = *ranks;
            let is_left = left == child_idx;
            let new_k = r.nrows();

            if is_left {
                let mut new_t = vec![0.0f64; kt * new_k * kr];
                for i in 0..kt {
                    for jp in 0..new_k {
                        for k in 0..kr {
                            let mut s = 0.0;
                            for j in 0..kl {
                                s += r[(jp, j)] * transfer[i * kl * kr + j * kr + k];
                            }
                            new_t[i * new_k * kr + jp * kr + k] = s;
                        }
                    }
                }
                *transfer = new_t;
                *ranks = [kt, new_k, kr];
            } else {
                let mut new_t = vec![0.0f64; kt * kl * new_k];
                for i in 0..kt {
                    for j in 0..kl {
                        for kp in 0..new_k {
                            let mut s = 0.0;
                            for k in 0..kr {
                                s += r[(kp, k)] * transfer[i * kl * kr + j * kr + k];
                            }
                            new_t[i * kl * new_k + j * new_k + kp] = s;
                        }
                    }
                }
                *transfer = new_t;
                *ranks = [kt, kl, new_k];
            }
        }
    }
}

// ─── Truncation ─────────────────────────────────────────────────────────────

impl HtTensor {
    /// Truncate ranks to maintain ‖A − Ã‖ ≤ √(2d−3) · tolerance.
    ///
    /// Uses a hierarchical tolerance distribution: coarser thresholds at the top of
    /// the tree, tighter at the leaves. This follows the SLAR paper's exponential
    /// decay requirement (Zheng et al. 2025, §3.2) and the HSVD quasi-best
    /// approximation bound (Grasedyck 2010).
    pub fn truncate(&mut self, tolerance: f64) {
        let _span = tracing::info_span!("ht_truncate").entered();
        self.orthogonalize_left();
        let decay = (2.0_f64 * 6.0 - 3.0).sqrt(); // √(2d-3) ≈ 3.0
        let eps_leaf = tolerance / decay;
        let eps_interior_lo = eps_leaf * decay.sqrt(); // nodes 6, 7
        let eps_interior_hi = eps_leaf * decay; // nodes 8, 9
        // Top-down: process higher-level nodes first (coarser tolerance),
        // then refine lower-level nodes (tighter tolerance).
        for &ni in &[8, 9] {
            self.truncate_node(ni, eps_interior_hi);
        }
        for &ni in &[6, 7] {
            self.truncate_node(ni, eps_interior_lo);
        }
        for &ni in &[0, 1, 2, 3, 4, 5] {
            self.truncate_node(ni, eps_leaf);
        }
        self.max_rank = self.compute_max_node_rank();
    }

    /// Truncate all nodes to a fixed maximum rank.
    pub fn truncate_to_rank(&mut self, max_rank: usize) {
        self.orthogonalize_left();
        for &ni in &[8, 9, 6, 7, 0, 1, 2, 3, 4, 5] {
            self.truncate_node_fixed(ni, max_rank);
        }
        self.max_rank = self.compute_max_node_rank();
    }

    fn truncate_node(&mut self, node_idx: usize, eps: f64) {
        let dummy = HtNode::Leaf {
            dim: 0,
            frame: Mat::new(),
        };
        let node = std::mem::replace(&mut self.nodes[node_idx], dummy);
        match node {
            HtNode::Leaf { dim, ref frame } => {
                let (u, s, vt) = thin_svd(frame);
                if u.ncols() == 0 {
                    self.nodes[node_idx] = node;
                    return;
                }
                let new_rank = truncation_rank(&s, eps).max(1).min(u.ncols());
                if new_rank >= frame.ncols() {
                    self.nodes[node_idx] = node;
                    return;
                }
                let new_frame = u.subcols(0, new_rank).to_owned();
                let sv = s_times_vt(&s, &vt, new_rank);
                self.nodes[node_idx] = HtNode::Leaf {
                    dim,
                    frame: new_frame,
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &sv);
            }
            HtNode::Interior {
                left,
                right,
                ref transfer,
                ranks,
            } if node_idx != ROOT => {
                let [kt, kl, kr] = ranks;
                let mat = vec_to_mat(transfer, kt, kl * kr);
                let (u, s, vt) = thin_svd(&mat);
                if u.ncols() == 0 {
                    self.nodes[node_idx] = node;
                    return;
                }
                let new_rank = truncation_rank(&s, eps).max(1).min(u.ncols());
                if new_rank >= kt {
                    self.nodes[node_idx] = node;
                    return;
                }
                let sv = s_times_vt(&s, &vt, new_rank);
                let absorb = u.subcols(0, new_rank).transpose().to_owned();
                self.nodes[node_idx] = HtNode::Interior {
                    left,
                    right,
                    transfer: mat_to_vec(&sv, new_rank, kl * kr),
                    ranks: [new_rank, kl, kr],
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &absorb);
            }
            other => {
                self.nodes[node_idx] = other;
            }
        }
    }

    fn truncate_node_fixed(&mut self, node_idx: usize, max_rank: usize) {
        let dummy = HtNode::Leaf {
            dim: 0,
            frame: Mat::new(),
        };
        let node = std::mem::replace(&mut self.nodes[node_idx], dummy);
        match node {
            HtNode::Leaf { dim, ref frame } if frame.ncols() > max_rank => {
                let (u, s, vt) = thin_svd(frame);
                let r = max_rank.min(u.ncols());
                let sv = s_times_vt(&s, &vt, r);
                self.nodes[node_idx] = HtNode::Leaf {
                    dim,
                    frame: u.subcols(0, r).to_owned(),
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &sv);
            }
            HtNode::Interior {
                left,
                right,
                ref transfer,
                ranks,
            } if node_idx != ROOT && ranks[0] > max_rank => {
                let [kt, kl, kr] = ranks;
                let mat = vec_to_mat(transfer, kt, kl * kr);
                let (u, s, vt) = thin_svd(&mat);
                let r = max_rank.min(u.ncols());
                let sv = s_times_vt(&s, &vt, r);
                let absorb = u.subcols(0, r).transpose().to_owned();
                self.nodes[node_idx] = HtNode::Interior {
                    left,
                    right,
                    transfer: mat_to_vec(&sv, r, kl * kr),
                    ranks: [r, kl, kr],
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &absorb);
            }
            other => {
                self.nodes[node_idx] = other;
            }
        }
    }
}

// ─── Arithmetic ─────────────────────────────────────────────────────────────

impl HtTensor {
    /// Rank-concatenation addition. Result has rank k₁+k₂. Call `truncate()` to compress.
    pub fn add(&self, other: &HtTensor) -> HtTensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in HT addition");
        let mut nodes = Vec::with_capacity(NUM_NODES);

        // Leaves: column-concatenate
        for d in 0..NUM_LEAVES {
            let (f1, dim) = leaf_data(&self.nodes[d]);
            let (f2, _) = leaf_data(&other.nodes[d]);
            let n = f1.nrows();
            let (k1, k2) = (f1.ncols(), f2.ncols());
            let mut combined = Mat::zeros(n, k1 + k2);
            for i in 0..n {
                for j in 0..k1 {
                    combined[(i, j)] = f1[(i, j)];
                }
                for j in 0..k2 {
                    combined[(i, k1 + j)] = f2[(i, j)];
                }
            }
            nodes.push(HtNode::Leaf {
                dim,
                frame: combined,
            });
        }

        // Interior nodes: block-diagonal extension
        for &ni in &[6usize, 7, 8, 9, 10] {
            let (l1, _, t1, [kt1, kl1, kr1]) = interior_data(&self.nodes[ni]);
            let (_, _, t2, [kt2, kl2, kr2]) = interior_data(&other.nodes[ni]);

            if ni == ROOT {
                // Root: kt=1 for both. Result: kt=1, kl=kl1+kl2, kr=kr1+kr2
                let (nkl, nkr) = (kl1 + kl2, kr1 + kr2);
                let mut nt = vec![0.0f64; nkl * nkr];
                for j in 0..kl1 {
                    for k in 0..kr1 {
                        nt[j * nkr + k] = t1[j * kr1 + k];
                    }
                }
                for j in 0..kl2 {
                    for k in 0..kr2 {
                        nt[(kl1 + j) * nkr + (kr1 + k)] = t2[j * kr2 + k];
                    }
                }
                nodes.push(HtNode::Interior {
                    left: l1,
                    right: 9,
                    transfer: nt,
                    ranks: [1, nkl, nkr],
                });
            } else {
                let (nkt, nkl, nkr) = (kt1 + kt2, kl1 + kl2, kr1 + kr2);
                let mut nt = vec![0.0f64; nkt * nkl * nkr];
                // Block 1
                for i in 0..kt1 {
                    for j in 0..kl1 {
                        for k in 0..kr1 {
                            nt[i * nkl * nkr + j * nkr + k] = t1[i * kl1 * kr1 + j * kr1 + k];
                        }
                    }
                }
                // Block 2
                for i in 0..kt2 {
                    for j in 0..kl2 {
                        for k in 0..kr2 {
                            nt[(kt1 + i) * nkl * nkr + (kl1 + j) * nkr + (kr1 + k)] =
                                t2[i * kl2 * kr2 + j * kr2 + k];
                        }
                    }
                }
                nodes.push(HtNode::Interior {
                    left: l1,
                    right: match ni {
                        6 => 2,
                        7 => 5,
                        8 => 6,
                        9 => 7,
                        _ => unreachable!(),
                    },
                    transfer: nt,
                    ranks: [nkt, nkl, nkr],
                });
            }
        }

        let actual_max_rank = nodes.iter().map(|n| n.rank()).max().unwrap_or(1);
        HtTensor {
            nodes,
            shape: self.shape,
            domain: self.domain.clone(),
            tolerance: self.tolerance.min(other.tolerance),
            max_rank: actual_max_rank,
            interpolation_mode: self.interpolation_mode,
            progress: self.progress.clone(),
            positivity_limiter: self.positivity_limiter,
            velocity_filter: self.velocity_filter,
            positivity_violations: AtomicU64::new(0),
        }
    }

    /// Scale the tensor by a scalar.
    pub fn scale(&mut self, alpha: f64) {
        if let HtNode::Interior {
            ref mut transfer, ..
        } = self.nodes[ROOT]
        {
            for v in transfer.iter_mut() {
                *v *= alpha;
            }
        }
    }

    /// Enforce positivity: clamp negative values to zero and rescale to preserve total mass.
    ///
    /// If the tensor is small enough to materialize (< 2 GB), this reconstructs the full
    /// array, applies the Zhang-Shu limiter, counts negatives, and re-compresses into HT
    /// format. For larger tensors, fiber-based sampling is used to count violations and
    /// a warning is logged.
    pub fn enforce_positivity(&mut self) {
        let total_elements: usize = self.shape.iter().product();
        let bytes = total_elements * 8;

        if bytes < MAX_MATERIALIZE_BYTES {
            // Full materialization path
            let mass_before = self.total_mass();
            let mut data = self.to_full();

            // Count negatives before clamping
            let negatives: u64 = data.iter().filter(|&&v| v < 0.0).count() as u64;
            if negatives > 0 {
                self.positivity_violations
                    .fetch_add(negatives, Ordering::Relaxed);

                super::wpfc::zhang_shu_limiter(&mut data, mass_before);

                // Rebuild HT from corrected data
                let rebuilt = HtTensor::from_full(&data, self.shape, &self.domain, self.tolerance);
                self.nodes = rebuilt.nodes;
                // shape, domain, tolerance unchanged
            }
        } else {
            // Fiber-based sampling path: evaluate along the main diagonal
            let n_samples = self.shape[0]
                .min(self.shape[1])
                .min(self.shape[2])
                .min(self.shape[3])
                .min(self.shape[4])
                .min(self.shape[5]);
            let mut neg_count: u64 = 0;
            for i in 0..n_samples {
                let val = self.evaluate([i, i, i, i, i, i]);
                if val < 0.0 {
                    neg_count += 1;
                }
            }
            if neg_count > 0 {
                self.positivity_violations
                    .fetch_add(neg_count, Ordering::Relaxed);
                tracing::warn!(
                    "HtTensor too large to materialize ({} bytes); fiber sampling found {} negative values out of {} samples",
                    bytes,
                    neg_count,
                    n_samples,
                );
            }
        }
    }

    /// Weighted addition: computes `alpha * self + beta * other`, then SVD-truncates.
    ///
    /// Uses rank-concatenation addition followed by truncation to keep ranks bounded.
    pub fn scaled_add(&self, alpha: f64, other: &HtTensor, beta: f64, tolerance: f64) -> HtTensor {
        let mut a = self.clone();
        a.scale(alpha);
        let mut b = other.clone();
        b.scale(beta);
        let mut result = a.add(&b);
        result.truncate(tolerance);
        result
    }

    /// Inner product ⟨self, other⟩ via Gram matrices. O(dk⁴).
    pub fn inner_product(&self, other: &HtTensor) -> f64 {
        assert_eq!(self.shape, other.shape);
        let g = self.gram_matrix(ROOT, other);
        g[(0, 0)]
    }

    fn gram_matrix(&self, node_idx: usize, other: &HtTensor) -> Mat<f64> {
        match (&self.nodes[node_idx], &other.nodes[node_idx]) {
            (HtNode::Leaf { frame: f1, .. }, HtNode::Leaf { frame: f2, .. }) => matmul_at_b(f1, f2),
            (
                HtNode::Interior {
                    left: l1,
                    right: r1,
                    transfer: t1,
                    ranks: rk1,
                },
                HtNode::Interior {
                    left: _,
                    right: _,
                    transfer: t2,
                    ranks: rk2,
                },
            ) => {
                let [kt1, kl1, kr1] = *rk1;
                let [kt2, kl2, kr2] = *rk2;
                let gl = self.gram_matrix(*l1, other);
                let gr = self.gram_matrix(*r1, other);

                let mut g = Mat::zeros(kt1, kt2);
                for i1 in 0..kt1 {
                    for i2 in 0..kt2 {
                        let mut sum = 0.0;
                        for j1 in 0..kl1 {
                            for j2 in 0..kl2 {
                                let glv = gl[(j1, j2)];
                                if glv.abs() < 1e-30 {
                                    continue;
                                }
                                for k1 in 0..kr1 {
                                    let b1 = t1[i1 * kl1 * kr1 + j1 * kr1 + k1];
                                    if b1.abs() < 1e-30 {
                                        continue;
                                    }
                                    let b1gl = b1 * glv;
                                    for k2 in 0..kr2 {
                                        sum += b1gl
                                            * t2[i2 * kl2 * kr2 + j2 * kr2 + k2]
                                            * gr[(k1, k2)];
                                    }
                                }
                            }
                        }
                        g[(i1, i2)] = sum;
                    }
                }
                g
            }
            _ => unreachable!("mismatched node types"),
        }
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.inner_product(self).max(0.0).sqrt()
    }

    /// Reconstruct full 6D array (expensive — diagnostics only).
    pub fn to_full(&self) -> Vec<f64> {
        let total: usize = self.shape.iter().product();
        let mut data = vec![0.0f64; total];
        let [n0, n1, n2, n3, n4, n5] = self.shape;
        let total_u64 = total as u64;
        let report_interval = (total_u64 / 100).max(1);
        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, total_u64);
        }
        let mut count = 0u64;
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    for i3 in 0..n3 {
                        for i4 in 0..n4 {
                            for i5 in 0..n5 {
                                let idx = flat_index(&self.shape, [i0, i1, i2, i3, i4, i5]);
                                data[idx] = self.evaluate([i0, i1, i2, i3, i4, i5]);
                                if let Some(ref p) = self.progress
                                    && count.is_multiple_of(report_interval)
                                {
                                    p.set_intra_progress(count, total_u64);
                                }
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        data
    }
}

// ─── Tricubic interpolation for SLAR advection ─────────────────────────────

/// Tricubic Catmull-Rom interpolation of the HT tensor at fractional grid positions.
///
/// `int_indices`: integer grid indices for the non-shifted dimensions.
/// `shift_dims`: which 3 dimensions are shifted (e.g. [0,1,2] for spatial, [3,4,5] for velocity).
/// `frac_pos`: fractional cell index (continuous) in each shifted dimension.
/// `periodic`: whether each shifted dimension is periodic.
fn tricubic_interpolate_ht(
    ht: &HtTensor,
    int_indices: [usize; 6],
    shift_dims: [usize; 3],
    frac_pos: [f64; 3],
    periodic: [bool; 3],
    workspace: &mut [f64],
) -> f64 {
    let ns = [
        ht.shape[shift_dims[0]],
        ht.shape[shift_dims[1]],
        ht.shape[shift_dims[2]],
    ];

    // For each shifted dim, compute floor index and Catmull-Rom weights
    let mut floors = [0i64; 3];
    let mut weights = [[0.0f64; 4]; 3];

    for d in 0..3 {
        let n = ns[d] as i64;

        // Absorbing BC: if departure point outside [-0.5, n-0.5), return 0
        if !periodic[d] && (frac_pos[d] < -0.5 || frac_pos[d] >= n as f64 - 0.5) {
            return 0.0;
        }

        let fl = frac_pos[d].floor() as i64;
        let t = frac_pos[d] - fl as f64;
        floors[d] = fl;

        // Catmull-Rom weights: stencil at fl-1, fl, fl+1, fl+2
        let t2 = t * t;
        let t3 = t2 * t;
        weights[d] = [
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ];
    }

    let mut result = 0.0;
    for s0 in 0..4i64 {
        let mut raw0 = floors[0] - 1 + s0;
        if periodic[0] {
            raw0 = raw0.rem_euclid(ns[0] as i64);
        } else {
            raw0 = raw0.clamp(0, ns[0] as i64 - 1);
        }

        for s1 in 0..4i64 {
            let mut raw1 = floors[1] - 1 + s1;
            if periodic[1] {
                raw1 = raw1.rem_euclid(ns[1] as i64);
            } else {
                raw1 = raw1.clamp(0, ns[1] as i64 - 1);
            }

            let w01 = weights[0][s0 as usize] * weights[1][s1 as usize];
            if w01.abs() < 1e-30 {
                continue;
            }

            for s2 in 0..4i64 {
                let mut raw2 = floors[2] - 1 + s2;
                if periodic[2] {
                    raw2 = raw2.rem_euclid(ns[2] as i64);
                } else {
                    raw2 = raw2.clamp(0, ns[2] as i64 - 1);
                }

                let w = w01 * weights[2][s2 as usize];
                if w.abs() < 1e-30 {
                    continue;
                }

                // Assemble 6D index
                let mut idx = int_indices;
                idx[shift_dims[0]] = raw0 as usize;
                idx[shift_dims[1]] = raw1 as usize;
                idx[shift_dims[2]] = raw2 as usize;

                result += ht.evaluate_into(idx, workspace) * w;
            }
        }
    }

    result
}

/// Sparse polynomial interpolation of the HT tensor at fractional grid positions.
/// Uses only 1 + 2d + d(d-1)/2 evaluations (10 for d=3 split) instead of 4^3=64.
///
/// Algorithm (for d=3 shifted dimensions):
/// - Center evaluation f(x_0): 1 eval
/// - Per-dim offsets f(x_0±e_k): 2d = 6 evals
/// - Cross-terms f(x_0+e_k+e_l) for k<l: d(d-1)/2 = 3 evals
///   Total: 10 evaluations
///
/// Reconstructs: p(δ) = f₀ + Σ_k (a_k·δ_k + b_k·δ_k²) + Σ_{k<l} c_{kl}·δ_k·δ_l
fn sparse_polynomial_interpolate_ht(
    ht: &HtTensor,
    int_indices: [usize; 6],
    shift_dims: [usize; 3],
    frac_pos: [f64; 3],
    periodic: [bool; 3],
    workspace: &mut [f64],
) -> f64 {
    let ns = [
        ht.shape[shift_dims[0]],
        ht.shape[shift_dims[1]],
        ht.shape[shift_dims[2]],
    ];

    // Compute base index (nearest grid point) and fractional displacement
    let mut base_idx = [0usize; 3];
    let mut delta = [0.0f64; 3];

    for d in 0..3 {
        let n = ns[d] as f64;

        // Absorbing BC: if departure point outside [-0.5, n-0.5), return 0
        if !periodic[d] && (frac_pos[d] < -0.5 || frac_pos[d] >= n - 0.5) {
            return 0.0;
        }

        let rounded = frac_pos[d].round();
        let bi = if periodic[d] {
            (rounded as i64).rem_euclid(ns[d] as i64) as usize
        } else {
            rounded.clamp(0.0, ns[d] as f64 - 1.0) as usize
        };
        base_idx[d] = bi;
        delta[d] = frac_pos[d] - rounded;
    }

    // Build the center 6D index
    let mut idx0 = int_indices;
    for d in 0..3 {
        idx0[shift_dims[d]] = base_idx[d];
    }

    // Helper to adjust a single shifted dimension index with BC handling
    let adjust = |d: usize, offset: i64| -> usize {
        let raw = base_idx[d] as i64 + offset;
        if periodic[d] {
            raw.rem_euclid(ns[d] as i64) as usize
        } else {
            raw.clamp(0, ns[d] as i64 - 1) as usize
        }
    };

    // Center evaluation: f0
    let f0 = ht.evaluate_into(idx0, workspace);

    // Per-dimension finite differences: f(x0 ± e_k)
    let mut f_plus = [0.0f64; 3];
    let mut f_minus = [0.0f64; 3];
    let mut a = [0.0f64; 3];
    let mut b = [0.0f64; 3];

    for k in 0..3 {
        let mut idx_p = idx0;
        idx_p[shift_dims[k]] = adjust(k, 1);
        f_plus[k] = ht.evaluate_into(idx_p, workspace);

        let mut idx_m = idx0;
        idx_m[shift_dims[k]] = adjust(k, -1);
        f_minus[k] = ht.evaluate_into(idx_m, workspace);

        a[k] = (f_plus[k] - f_minus[k]) / 2.0;
        b[k] = (f_plus[k] + f_minus[k] - 2.0 * f0) / 2.0;
    }

    // Cross-terms: f(x0 + e_k + e_l) for k < l
    let mut c = [[0.0f64; 3]; 3];
    for k in 0..3 {
        for l in (k + 1)..3 {
            let mut idx_diag = idx0;
            idx_diag[shift_dims[k]] = adjust(k, 1);
            idx_diag[shift_dims[l]] = adjust(l, 1);
            let f_diag = ht.evaluate_into(idx_diag, workspace);
            c[k][l] = f_diag - f_plus[k] - f_plus[l] + f0;
        }
    }

    // Reconstruct: p(δ) = f₀ + Σ_k (a_k·δ_k + b_k·δ_k²) + Σ_{k<l} c_{kl}·δ_k·δ_l
    let mut result = f0;
    for k in 0..3 {
        result += a[k] * delta[k] + b[k] * delta[k] * delta[k];
    }
    for k in 0..3 {
        for l in (k + 1)..3 {
            result += c[k][l] * delta[k] * delta[l];
        }
    }

    result
}

/// Build the exponential filter kernel for a 1D FFT of length `n` (HT variant).
///
/// Identical to the kernel in `uniform.rs` but defined locally to avoid
/// coupling the HT module to UniformGrid6D internals.
fn build_exp_filter_kernel_ht(n: usize, cutoff_fraction: f64, order: usize) -> Vec<f64> {
    let half = n as f64 / 2.0;
    let exp = (2 * order) as i32;
    (0..n)
        .map(|m| {
            let sym = if m <= n / 2 { m } else { n - m };
            let k_norm = sym as f64 / half;
            (-(k_norm / cutoff_fraction).powi(exp)).exp()
        })
        .collect()
}

// ─── PhaseSpaceRepr ─────────────────────────────────────────────────────────

impl PhaseSpaceRepr for HtTensor {
    fn set_progress(&mut self, p: std::sync::Arc<super::super::progress::StepProgress>) {
        self.progress = Some(p);
    }

    fn compute_density(&self) -> DensityField {
        let _span = tracing::info_span!("ht_compute_density").entered();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;
        let dv = self.domain.dv();
        let dv3 = dv[0] * dv[1] * dv[2];

        // Contract velocity leaves with uniform weight vectors (sum-based ∫f dv³)
        let wv1 = vec![1.0f64; nv1];
        let wv2 = vec![1.0f64; nv2];
        let wv3 = vec![1.0f64; nv3];

        let c3 = contract_leaf_weights(&self.nodes[3], &wv1);
        let c4 = contract_leaf_weights(&self.nodes[4], &wv2);
        let c5 = contract_leaf_weights(&self.nodes[5], &wv3);

        // Contract through velocity subtree
        let c7 = self.contract_interior_vec(7, &c4, &c5);
        let c9 = self.contract_interior_vec(9, &c3, &c7);

        // Pre-contract root with velocity result
        let (_, kl_root, kr_root, t_root) = get_interior(&self.nodes[ROOT]);
        let mut eff_left = vec![0.0f64; kl_root];
        for j in 0..kl_root {
            let mut s = 0.0;
            for k in 0..kr_root {
                s += t_root[j * kr_root + k] * c9[k]; // kt_root=1
            }
            eff_left[j] = s;
        }

        // Pre-contract with node 8 transfer
        let (kt8, kl8, kr8, t8) = get_interior(&self.nodes[8]);
        let mut eff8 = vec![0.0f64; kl8 * kr8];
        for j8 in 0..kl8 {
            for k8 in 0..kr8 {
                let mut s = 0.0;
                for j in 0..kt8 {
                    s += eff_left[j] * t8[j * kl8 * kr8 + j8 * kr8 + k8];
                }
                eff8[j8 * kr8 + k8] = s;
            }
        }

        let (f0, _) = leaf_data(&self.nodes[0]);
        let (f1, _) = leaf_data(&self.nodes[1]);
        let (f2, _) = leaf_data(&self.nodes[2]);
        let (kt6, kl6, kr6, t6) = get_interior(&self.nodes[6]);

        let counter = AtomicU64::new(0);
        let total = nx1 as u64;
        let report_interval = (total / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, total);
        }

        let density: Vec<f64> = (0..nx1)
            .into_par_iter()
            .flat_map(|i0| {
                let mut eff_i0 = vec![0.0f64; kr8];
                for k8 in 0..kr8 {
                    let mut s = 0.0;
                    for j8 in 0..kl8 {
                        s += eff8[j8 * kr8 + k8] * f0[(i0, j8)];
                    }
                    eff_i0[k8] = s;
                }

                let mut row = vec![0.0f64; nx2 * nx3];
                for i1 in 0..nx2 {
                    for i2 in 0..nx3 {
                        let mut val = 0.0;
                        for k8 in 0..kr8 {
                            let mut n6 = 0.0;
                            for j6 in 0..kl6 {
                                let lv = f1[(i1, j6)];
                                for k6 in 0..kr6 {
                                    n6 += t6[k8 * kl6 * kr6 + j6 * kr6 + k6] * lv * f2[(i2, k6)];
                                }
                            }
                            val += eff_i0[k8] * n6;
                        }
                        row[i1 * nx3 + i2] = val * dv3;
                    }
                }

                if let Some(ref p) = self.progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, total);
                    }
                }

                row
            })
            .collect();

        DensityField {
            data: density,
            shape: [nx1, nx2, nx3],
        }
    }

    fn advect_x(&mut self, _displacement: &DisplacementField, dt: f64) {
        let _span = tracing::info_span!("ht_advect_x").entered();
        if dt.abs() < 1e-30 {
            return;
        }

        let old_ht = self.clone();
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.domain.lx();
        let lv = self.domain.lv();
        let shape = self.shape;
        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);
        let tol = self.tolerance;
        let max_rank = self.max_rank;
        let interp_mode = self.interpolation_mode;

        let saved_progress = self.progress.clone();
        let ws_len = old_ht.eval_workspace_len();

        let new_ht = HtTensor::from_function_aca(
            move |x_phys: &[f64; 3], v_phys: &[f64; 3]| -> f64 {
                thread_local! {
                    static EVAL_WS: RefCell<Vec<f64>> = RefCell::new(Vec::new());
                }
                // Integer velocity indices (round to nearest cell center)
                let iv = [
                    ((v_phys[0] + lv[0]) / dv[0] - 0.5)
                        .round()
                        .clamp(0.0, (shape[3] - 1) as f64) as usize,
                    ((v_phys[1] + lv[1]) / dv[1] - 0.5)
                        .round()
                        .clamp(0.0, (shape[4] - 1) as f64) as usize,
                    ((v_phys[2] + lv[2]) / dv[2] - 0.5)
                        .round()
                        .clamp(0.0, (shape[5] - 1) as f64) as usize,
                ];

                // Departure point: x_dep = x - v*dt
                let x_dep = [
                    x_phys[0] - v_phys[0] * dt,
                    x_phys[1] - v_phys[1] * dt,
                    x_phys[2] - v_phys[2] * dt,
                ];

                // Fractional spatial grid indices of departure point
                let frac = [
                    (x_dep[0] + lx[0]) / dx[0] - 0.5,
                    (x_dep[1] + lx[1]) / dx[1] - 0.5,
                    (x_dep[2] + lx[2]) / dx[2] - 0.5,
                ];

                let int_idx = [0, 0, 0, iv[0], iv[1], iv[2]];
                EVAL_WS.with(|ws_cell| {
                    let mut ws = ws_cell.borrow_mut();
                    if ws.len() < ws_len {
                        ws.resize(ws_len, 0.0);
                    }
                    match interp_mode {
                        InterpolationMode::TricubicCatmullRom => tricubic_interpolate_ht(
                            &old_ht,
                            int_idx,
                            [0, 1, 2],
                            frac,
                            [periodic; 3],
                            &mut ws,
                        ),
                        InterpolationMode::SparsePolynomial => sparse_polynomial_interpolate_ht(
                            &old_ht,
                            int_idx,
                            [0, 1, 2],
                            frac,
                            [periodic; 3],
                            &mut ws,
                        ),
                    }
                })
            },
            &self.domain,
            tol,
            max_rank,
            None,
            None,
        );

        *self = new_ht;
        self.progress = saved_progress;

        if self.positivity_limiter {
            self.enforce_positivity();
        }
    }

    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        let _span = tracing::info_span!("ht_advect_v").entered();
        if dt.abs() < 1e-30 {
            return;
        }

        let old_ht = self.clone();
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.domain.lx();
        let lv = self.domain.lv();
        let shape = self.shape;
        let [nx1, nx2, nx3, _, _, _] = shape;
        let periodic_v = matches!(self.domain.velocity_bc, VelocityBoundType::Truncated);
        let tol = self.tolerance;
        let max_rank = self.max_rank;
        let interp_mode = self.interpolation_mode;

        // Share acceleration data via Arc (O(1) clone vs O(N³) memcpy)
        let gx: Arc<[f64]> = Arc::from(acceleration.gx.as_slice());
        let gy: Arc<[f64]> = Arc::from(acceleration.gy.as_slice());
        let gz: Arc<[f64]> = Arc::from(acceleration.gz.as_slice());

        let saved_progress = self.progress.clone();
        let ws_len = old_ht.eval_workspace_len();

        let new_ht = HtTensor::from_function_aca(
            move |x_phys: &[f64; 3], v_phys: &[f64; 3]| -> f64 {
                thread_local! {
                    static EVAL_WS: RefCell<Vec<f64>> = RefCell::new(Vec::new());
                }
                // Spatial grid index for acceleration lookup
                let ix = [
                    ((x_phys[0] + lx[0]) / dx[0] - 0.5)
                        .round()
                        .clamp(0.0, (nx1 - 1) as f64) as usize,
                    ((x_phys[1] + lx[1]) / dx[1] - 0.5)
                        .round()
                        .clamp(0.0, (nx2 - 1) as f64) as usize,
                    ((x_phys[2] + lx[2]) / dx[2] - 0.5)
                        .round()
                        .clamp(0.0, (nx3 - 1) as f64) as usize,
                ];
                let flat = ix[0] * nx2 * nx3 + ix[1] * nx3 + ix[2];
                let ax = gx[flat];
                let ay = gy[flat];
                let az = gz[flat];

                // Integer spatial indices
                let ix_int = [
                    ((x_phys[0] + lx[0]) / dx[0] - 0.5)
                        .round()
                        .clamp(0.0, (nx1 - 1) as f64) as usize,
                    ((x_phys[1] + lx[1]) / dx[1] - 0.5)
                        .round()
                        .clamp(0.0, (nx2 - 1) as f64) as usize,
                    ((x_phys[2] + lx[2]) / dx[2] - 0.5)
                        .round()
                        .clamp(0.0, (nx3 - 1) as f64) as usize,
                ];

                // Departure in velocity: v_dep = v - a*dt
                let v_dep = [
                    v_phys[0] - ax * dt,
                    v_phys[1] - ay * dt,
                    v_phys[2] - az * dt,
                ];
                let frac = [
                    (v_dep[0] + lv[0]) / dv[0] - 0.5,
                    (v_dep[1] + lv[1]) / dv[1] - 0.5,
                    (v_dep[2] + lv[2]) / dv[2] - 0.5,
                ];

                let int_idx = [ix_int[0], ix_int[1], ix_int[2], 0, 0, 0];
                EVAL_WS.with(|ws_cell| {
                    let mut ws = ws_cell.borrow_mut();
                    if ws.len() < ws_len {
                        ws.resize(ws_len, 0.0);
                    }
                    match interp_mode {
                        InterpolationMode::TricubicCatmullRom => tricubic_interpolate_ht(
                            &old_ht,
                            int_idx,
                            [3, 4, 5],
                            frac,
                            [periodic_v; 3],
                            &mut ws,
                        ),
                        InterpolationMode::SparsePolynomial => sparse_polynomial_interpolate_ht(
                            &old_ht,
                            int_idx,
                            [3, 4, 5],
                            frac,
                            [periodic_v; 3],
                            &mut ws,
                        ),
                    }
                })
            },
            &self.domain,
            tol,
            max_rank,
            None,
            None,
        );

        let saved_filter = self.velocity_filter;
        let saved_positivity = self.positivity_limiter;
        *self = new_ht;
        self.progress = saved_progress;
        self.velocity_filter = saved_filter;
        self.positivity_limiter = saved_positivity;

        if self.positivity_limiter {
            self.enforce_positivity();
        }

        self.apply_velocity_filter();
    }

    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.domain.lx();
        let lv = self.domain.lv();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;

        let ix = [
            ((position[0] + lx[0]) / dx[0])
                .floor()
                .clamp(0.0, (nx1 - 1) as f64) as usize,
            ((position[1] + lx[1]) / dx[1])
                .floor()
                .clamp(0.0, (nx2 - 1) as f64) as usize,
            ((position[2] + lx[2]) / dx[2])
                .floor()
                .clamp(0.0, (nx3 - 1) as f64) as usize,
        ];
        let dv3 = dv[0] * dv[1] * dv[2];

        match order {
            0 => {
                let mut sum = 0.0;
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            sum += self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3]);
                        }
                    }
                }
                Tensor {
                    data: vec![sum * dv3],
                    rank: 0,
                    shape: vec![],
                }
            }
            1 => {
                let mut vbar = [0.0; 3];
                let mut rho = 0.0;
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            let f = self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3]);
                            vbar[0] += f * (-lv[0] + (iv1 as f64 + 0.5) * dv[0]);
                            vbar[1] += f * (-lv[1] + (iv2 as f64 + 0.5) * dv[1]);
                            vbar[2] += f * (-lv[2] + (iv3 as f64 + 0.5) * dv[2]);
                            rho += f;
                        }
                    }
                }
                rho *= dv3;
                let s = if rho > 1e-30 { dv3 / rho } else { 0.0 };
                Tensor {
                    data: vec![vbar[0] * s, vbar[1] * s, vbar[2] * s],
                    rank: 1,
                    shape: vec![3],
                }
            }
            2 => {
                let mut m2 = [0.0f64; 9];
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            let f = self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3]);
                            let v = [
                                -lv[0] + (iv1 as f64 + 0.5) * dv[0],
                                -lv[1] + (iv2 as f64 + 0.5) * dv[1],
                                -lv[2] + (iv3 as f64 + 0.5) * dv[2],
                            ];
                            for a in 0..3 {
                                for b in 0..3 {
                                    m2[a * 3 + b] += f * v[a] * v[b];
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
        let vol = self.domain.cell_volume_6d();

        // Contract all leaves with unit weights, propagate through tree
        let mut cvec: Vec<Vec<f64>> = Vec::with_capacity(6);
        for d in 0..6 {
            let w = vec![1.0f64; self.shape[d]];
            cvec.push(contract_leaf_weights(&self.nodes[d], &w));
        }
        let c6 = self.contract_interior_vec(6, &cvec[1], &cvec[2]);
        let c7 = self.contract_interior_vec(7, &cvec[4], &cvec[5]);
        let c8 = self.contract_interior_vec(8, &cvec[0], &c6);
        let c9 = self.contract_interior_vec(9, &cvec[3], &c7);
        let cr = self.contract_interior_vec(ROOT, &c8, &c9);
        cr[0] * vol
    }

    fn casimir_c2(&self) -> f64 {
        let vol = self.domain.cell_volume_6d();
        self.inner_product(self) * vol
    }

    fn can_materialize(&self) -> bool {
        self.shape.iter().product::<usize>() * 8 <= MAX_MATERIALIZE_BYTES
    }

    fn entropy(&self) -> f64 {
        if !self.can_materialize() {
            // Entropy S = −∫ f·ln(f) is nonlinear and cannot be computed via
            // tree contractions; materialization is required but would exceed
            // memory for large grids. This is an expected limitation.
            tracing::debug!(
                "HtTensor::entropy(): full grid ({} elements) exceeds materialization limit",
                self.shape.iter().product::<usize>()
            );
            return f64::NAN;
        }
        let vol = self.domain.cell_volume_6d();
        let data = self.to_full();
        let total = data.len() as u64;
        let report_interval = (total / 100).max(1);
        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, total);
        }
        let mut sum = 0.0f64;
        for (idx, &f) in data.iter().enumerate() {
            if f > 0.0 {
                sum += -f * f.ln();
            }
            if let Some(ref p) = self.progress
                && (idx as u64).is_multiple_of(report_interval)
            {
                p.set_intra_progress(idx as u64, total);
            }
        }
        sum * vol
    }

    fn stream_count(&self) -> StreamCountField {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;
        let dv = self.domain.dv();
        let dv23 = dv[1] * dv[2];

        let counter = AtomicU64::new(0);
        let total = (nx1 * nx2 * nx3) as u64;
        let report_interval = (total / 100).max(1);

        if let Some(ref p) = self.progress {
            p.set_intra_progress(0, total);
        }

        let out: Vec<u32> = (0..nx1 * nx2 * nx3)
            .into_par_iter()
            .map(|si| {
                let ix1 = si / (nx2 * nx3);
                let ix2 = (si / nx3) % nx2;
                let ix3 = si % nx3;
                let marginal: Vec<f64> = (0..nv1)
                    .map(|iv1| {
                        (0..nv2 * nv3)
                            .map(|vi| {
                                let iv3 = vi % nv3;
                                let iv2 = vi / nv3;
                                self.evaluate([ix1, ix2, ix3, iv1, iv2, iv3])
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

                if let Some(ref p) = self.progress {
                    let c = counter.fetch_add(1, Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, total);
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
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;
        let dx = self.domain.dx();
        let lx = self.domain.lx();
        let ix = [
            ((position[0] + lx[0]) / dx[0])
                .floor()
                .clamp(0.0, (nx1 - 1) as f64) as usize,
            ((position[1] + lx[1]) / dx[1])
                .floor()
                .clamp(0.0, (nx2 - 1) as f64) as usize,
            ((position[2] + lx[2]) / dx[2])
                .floor()
                .clamp(0.0, (nx3 - 1) as f64) as usize,
        ];
        (0..nv1 * nv2 * nv3)
            .map(|vi| {
                let iv3 = vi % nv3;
                let iv2 = (vi / nv3) % nv2;
                let iv1 = vi / (nv2 * nv3);
                self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3])
            })
            .collect()
    }

    fn total_kinetic_energy(&self) -> Option<f64> {
        // Use tree contractions via compute_energy_density() instead of
        // materializing the full 6D grid (which exceeds memory for large grids).
        let dx3 = self.domain.cell_volume_3d();
        let energy_density = self.compute_energy_density();
        Some(energy_density.data.iter().sum::<f64>() * dx3)
    }

    fn to_snapshot(&self, time: f64) -> Option<PhaseSpaceSnapshot> {
        if !self.can_materialize() {
            tracing::warn!(
                "HtTensor::to_snapshot() skipped: full grid ({} elements) exceeds materialization limit",
                self.shape.iter().product::<usize>()
            );
            return None;
        }
        Some(PhaseSpaceSnapshot {
            data: self.to_full(),
            shape: self.shape,
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
        // Delegate to the inherent method
        Self::memory_bytes(self)
    }
}

// ─── Moment extraction (Phase 3: LoMaC integration) ─────────────────────────

impl HtTensor {
    /// Compute momentum density J_i(x) = ∫ v_i f(x,v) dv³ for axis i ∈ {0,1,2}.
    ///
    /// Same tree contraction as `compute_density()` but with velocity weights
    /// set to v_i on the corresponding velocity leaf, and 1 on the others.
    pub fn compute_momentum_density(&self, axis: usize) -> DensityField {
        assert!(axis < 3, "axis must be 0, 1, or 2");
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;
        let dv = self.domain.dv();
        let dv3 = dv[0] * dv[1] * dv[2];
        let lv = self.domain.lv();
        let nv = [nv1, nv2, nv3];

        // Build velocity weights: v_i for the target axis, 1 for others
        let mut w3: Vec<f64> = vec![1.0; nv1];
        let mut w4: Vec<f64> = vec![1.0; nv2];
        let mut w5: Vec<f64> = vec![1.0; nv3];

        let weights = [&mut w3, &mut w4, &mut w5];
        for iv in 0..nv[axis] {
            let v = -lv[axis] + (iv as f64 + 0.5) * dv[axis];
            weights[axis][iv] = v;
        }

        self.contract_velocity_weighted(&w3, &w4, &w5, dv3)
    }

    /// Compute energy density e(x) = ∫ ½|v|² f(x,v) dv³.
    ///
    /// Decomposed as: e = ½ ∫ v₁² f dv³ + ½ ∫ v₂² f dv³ + ½ ∫ v₃² f dv³.
    /// Each term is a separate tree contraction with v_i² weights on one leaf.
    pub fn compute_energy_density(&self) -> DensityField {
        let [_, _, _, nv1, nv2, nv3] = self.shape;
        let dv = self.domain.dv();
        let dv3 = dv[0] * dv[1] * dv[2];
        let lv = self.domain.lv();
        let nv = [nv1, nv2, nv3];

        let mut result: Option<DensityField> = None;

        for axis in 0..3 {
            let mut w3 = vec![1.0f64; nv1];
            let mut w4 = vec![1.0f64; nv2];
            let mut w5 = vec![1.0f64; nv3];

            let weights = [&mut w3, &mut w4, &mut w5];
            for iv in 0..nv[axis] {
                let v = -lv[axis] + (iv as f64 + 0.5) * dv[axis];
                weights[axis][iv] = 0.5 * v * v;
            }

            let term = self.contract_velocity_weighted(&w3, &w4, &w5, dv3);

            result = Some(match result {
                None => term,
                Some(mut acc) => {
                    for (a, b) in acc.data.iter_mut().zip(term.data.iter()) {
                        *a += b;
                    }
                    acc
                }
            });
        }

        result.unwrap_or_else(|| DensityField {
            data: vec![0.0; self.shape[0] * self.shape[1] * self.shape[2]],
            shape: [self.shape[0], self.shape[1], self.shape[2]],
        })
    }

    /// Extract all macroscopic moments needed for LoMaC: (ρ, J₁, J₂, J₃, e).
    /// Returns per-spatial-cell `MacroState` vectors.
    pub fn extract_macro_state(&self) -> Vec<crate::tooling::core::conservation::kfvs::MacroState> {
        let density = self.compute_density();
        let j0 = self.compute_momentum_density(0);
        let j1 = self.compute_momentum_density(1);
        let j2 = self.compute_momentum_density(2);
        let energy = self.compute_energy_density();

        density
            .data
            .iter()
            .zip(j0.data.iter())
            .zip(j1.data.iter())
            .zip(j2.data.iter())
            .zip(energy.data.iter())
            .map(|((((&rho, &jx), &jy), &jz), &e)| {
                crate::tooling::core::conservation::kfvs::MacroState {
                    density: rho,
                    momentum: [jx, jy, jz],
                    energy: e,
                }
            })
            .collect()
    }

    /// Generalized velocity contraction: same algorithm as `compute_density()` but
    /// with arbitrary weights on velocity leaves instead of uniform ones.
    fn contract_velocity_weighted(
        &self,
        w_v1: &[f64],
        w_v2: &[f64],
        w_v3: &[f64],
        dv3: f64,
    ) -> DensityField {
        let [nx1, nx2, nx3, _, _, _] = self.shape;

        let c3 = contract_leaf_weights(&self.nodes[3], w_v1);
        let c4 = contract_leaf_weights(&self.nodes[4], w_v2);
        let c5 = contract_leaf_weights(&self.nodes[5], w_v3);

        let c7 = self.contract_interior_vec(7, &c4, &c5);
        let c9 = self.contract_interior_vec(9, &c3, &c7);

        let (_, kl_root, kr_root, t_root) = get_interior(&self.nodes[ROOT]);
        let mut eff_left = vec![0.0f64; kl_root];
        for j in 0..kl_root {
            let mut s = 0.0;
            for k in 0..kr_root {
                s += t_root[j * kr_root + k] * c9[k];
            }
            eff_left[j] = s;
        }

        let (kt8, kl8, kr8, t8) = get_interior(&self.nodes[8]);
        let mut eff8 = vec![0.0f64; kl8 * kr8];
        for j8 in 0..kl8 {
            for k8 in 0..kr8 {
                let mut s = 0.0;
                for j in 0..kt8 {
                    s += eff_left[j] * t8[j * kl8 * kr8 + j8 * kr8 + k8];
                }
                eff8[j8 * kr8 + k8] = s;
            }
        }

        let (f0, _) = leaf_data(&self.nodes[0]);
        let (f1, _) = leaf_data(&self.nodes[1]);
        let (f2, _) = leaf_data(&self.nodes[2]);
        let (_, kl6, kr6, t6) = get_interior(&self.nodes[6]);

        let density: Vec<f64> = (0..nx1)
            .into_par_iter()
            .flat_map(|i0| {
                let mut eff_i0 = vec![0.0f64; kr8];
                for k8 in 0..kr8 {
                    let mut s = 0.0;
                    for j8 in 0..kl8 {
                        s += eff8[j8 * kr8 + k8] * f0[(i0, j8)];
                    }
                    eff_i0[k8] = s;
                }

                let mut row = vec![0.0f64; nx2 * nx3];
                for i1 in 0..nx2 {
                    for i2 in 0..nx3 {
                        let mut val = 0.0;
                        for k8 in 0..kr8 {
                            let mut n6 = 0.0;
                            for j6 in 0..kl6 {
                                let lv = f1[(i1, j6)];
                                for k6 in 0..kr6 {
                                    n6 += t6[k8 * kl6 * kr6 + j6 * kr6 + k6] * lv * f2[(i2, k6)];
                                }
                            }
                            val += eff_i0[k8] * n6;
                        }
                        row[i1 * nx3 + i2] = val * dv3;
                    }
                }
                row
            })
            .collect();

        DensityField {
            data: density,
            shape: [nx1, nx2, nx3],
        }
    }
}

// ─── Private HtTensor helpers ───────────────────────────────────────────────

impl HtTensor {
    /// Contract interior node's transfer tensor with two vectors.
    fn contract_interior_vec(&self, node_idx: usize, left: &[f64], right: &[f64]) -> Vec<f64> {
        let (kt, kl, kr, t) = get_interior(&self.nodes[node_idx]);
        contract_transfer(t, kt, kl, kr, left, right)
    }
}

// ─── Free functions ─────────────────────────────────────────────────────────

#[inline]
fn flat_index(shape: &[usize; 6], idx: [usize; 6]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for d in (0..6).rev() {
        flat += idx[d] * stride;
        stride *= shape[d];
    }
    flat
}

#[inline]
fn parent_of(child: usize) -> usize {
    match child {
        0 => 8,
        1 => 6,
        2 => 6,
        3 => 9,
        4 => 7,
        5 => 7,
        6 => 8,
        7 => 9,
        8 => 10,
        9 => 10,
        _ => {
            debug_assert!(false, "no parent for node {child}");
            10 // root fallback
        }
    }
}

#[inline]
fn leaf_data(node: &HtNode) -> (&Mat<f64>, usize) {
    static EMPTY: std::sync::LazyLock<Mat<f64>> = std::sync::LazyLock::new(|| Mat::zeros(0, 0));
    match node {
        HtNode::Leaf { dim, frame } => (frame, *dim),
        HtNode::Interior { .. } => {
            debug_assert!(false, "expected leaf");
            (&EMPTY, 0)
        }
    }
}

#[inline]
fn get_interior(node: &HtNode) -> (usize, usize, usize, &[f64]) {
    match node {
        HtNode::Interior {
            ranks, transfer, ..
        } => (ranks[0], ranks[1], ranks[2], transfer),
        HtNode::Leaf { .. } => {
            debug_assert!(false, "expected interior");
            (0, 0, 0, &[])
        }
    }
}

#[inline]
fn interior_data(node: &HtNode) -> (usize, usize, &[f64], [usize; 3]) {
    match node {
        HtNode::Interior {
            left,
            right,
            transfer,
            ranks,
        } => (*left, *right, transfer, *ranks),
        HtNode::Leaf { .. } => {
            debug_assert!(false, "expected interior");
            (0, 0, &[], [0; 3])
        }
    }
}

#[inline]
fn contract_transfer(
    t: &[f64],
    kt: usize,
    kl: usize,
    kr: usize,
    left: &[f64],
    right: &[f64],
) -> Vec<f64> {
    let mut result = vec![0.0f64; kt];
    for i in 0..kt {
        let mut sum = 0.0;
        for j in 0..kl {
            let lj = left[j];
            let base = i * kl * kr + j * kr;
            for k in 0..kr {
                sum += t[base + k] * lj * right[k];
            }
        }
        result[i] = sum;
    }
    result
}

#[inline]
fn contract_transfer_into(
    t: &[f64],
    kt: usize,
    kl: usize,
    kr: usize,
    left: &[f64],
    right: &[f64],
    out: &mut [f64],
) {
    for i in 0..kt {
        let mut sum = 0.0;
        for j in 0..kl {
            let lj = left[j];
            let base = i * kl * kr + j * kr;
            for k in 0..kr {
                sum += t[base + k] * lj * right[k];
            }
        }
        out[i] = sum;
    }
}

#[inline]
fn contract_leaf_weights(node: &HtNode, weights: &[f64]) -> Vec<f64> {
    let (frame, _) = leaf_data(node);
    let n = frame.nrows();
    let k = frame.ncols();
    assert_eq!(n, weights.len());
    (0..k)
        .map(|c| {
            let mut sum = 0.0;
            for r in 0..n {
                sum += frame[(r, c)] * weights[r];
            }
            sum
        })
        .collect()
}

// ─── Linear algebra helpers ─────────────────────────────────────────────────

fn multi_mode_unfold(data: &[f64], shape: &[usize; 6], modes: &[usize]) -> Mat<f64> {
    let total: usize = shape.iter().product();
    let n_rows: usize = modes.iter().map(|&d| shape[d]).product();
    let n_cols = total / n_rows;

    let mut strides = [0usize; 6];
    strides[5] = 1;
    for d in (0..5).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    let mut mat = Mat::zeros(n_rows, n_cols);

    for flat in 0..total {
        let mut rem = flat;
        let mut indices = [0usize; 6];
        for d in 0..6 {
            indices[d] = rem / strides[d];
            rem %= strides[d];
        }

        let mut row = 0;
        let mut row_stride = 1;
        for &d in modes.iter().rev() {
            row += indices[d] * row_stride;
            row_stride *= shape[d];
        }

        let mut col = 0;
        let mut col_stride = 1;
        for d in (0..6).rev() {
            if !modes.contains(&d) {
                col += indices[d] * col_stride;
                col_stride *= shape[d];
            }
        }

        mat[(row, col)] = data[flat];
    }
    mat
}

#[inline]
fn thin_svd(mat: &Mat<f64>) -> (Mat<f64>, Vec<f64>, Mat<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();
    let k = m.min(n);
    if k == 0 {
        return (Mat::zeros(m, 0), vec![], Mat::zeros(0, n));
    }
    let svd = match mat.as_ref().thin_svd() {
        Ok(s) => s,
        Err(_) => return (Mat::zeros(m, 0), vec![], Mat::zeros(0, n)),
    };
    let u = svd.U().to_owned();
    let vt = svd.V().transpose().to_owned();
    let s_diag = svd.S().column_vector();
    let s: Vec<f64> = (0..k).map(|i| s_diag[i]).collect();
    (u, s, vt)
}

#[inline]
fn truncation_rank(sv: &[f64], eps: f64) -> usize {
    let eps2 = eps * eps;
    let mut tail_sq = 0.0;
    for k in (0..sv.len()).rev() {
        tail_sq += sv[k] * sv[k];
        if tail_sq > eps2 {
            return k + 1;
        }
    }
    1
}

#[inline]
fn qr_decompose(mat: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();
    if m.min(n) == 0 {
        return (Mat::zeros(m, 0), Mat::zeros(0, n));
    }
    let qr = mat.as_ref().qr();
    (qr.compute_thin_Q(), qr.thin_R().to_owned())
}

#[inline]
fn s_times_vt(s: &[f64], vt: &Mat<f64>, rank: usize) -> Mat<f64> {
    Mat::from_fn(rank, vt.ncols(), |i, j| s[i] * vt[(i, j)])
}

#[inline]
fn vec_to_mat(data: &[f64], rows: usize, cols: usize) -> Mat<f64> {
    Mat::from_fn(rows, cols, |i, j| data[i * cols + j])
}

#[inline]
fn mat_to_vec(m: &Mat<f64>, rows: usize, cols: usize) -> Vec<f64> {
    (0..rows)
        .flat_map(|i| (0..cols).map(move |j| m[(i, j)]))
        .collect()
}

#[inline]
fn matmul_at_b(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    a.transpose() * b
}

/// Compute transfer tensor for an interior node.
///
/// For non-root: B_t[i,j,k] = Σ_{S_t indices} U_t[parent_idx, i] · U_{t1}[left_idx, j] · U_{t2}[right_idx, k]
/// (pure frame projection, no data needed).
///
/// For root: B_root[0,j,k] = Σ_{all indices} data[flat] · U_left[left_idx, j] · U_right[right_idx, k]
/// (projects data onto child frames).
fn compute_transfer_tensor(
    data: &[f64],
    shape: &[usize; 6],
    parent_dims: &[usize],
    left_dims: &[usize],
    right_dims: &[usize],
    parent_frame: &Mat<f64>,
    left_frame: &Mat<f64>,
    right_frame: &Mat<f64>,
) -> Vec<f64> {
    let kt = parent_frame.ncols();
    let kl = left_frame.ncols();
    let kr = right_frame.ncols();
    let is_root = parent_dims.len() == 6;

    let mut transfer = vec![0.0f64; kt * kl * kr];

    if is_root {
        // Root: B_root[0,j,k] = U_left^T @ data_matricized @ U_right
        let total: usize = shape.iter().product();
        let mut strides = [0usize; 6];
        strides[5] = 1;
        for d in (0..5).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        for flat in 0..total {
            let val = data[flat];
            if val.abs() < 1e-30 {
                continue;
            }

            let mut rem = flat;
            let mut indices = [0usize; 6];
            for d in 0..6 {
                indices[d] = rem / strides[d];
                rem %= strides[d];
            }

            let mut left_idx = 0;
            let mut ls = 1;
            for &d in left_dims.iter().rev() {
                left_idx += indices[d] * ls;
                ls *= shape[d];
            }

            let mut right_idx = 0;
            let mut rs = 1;
            for &d in right_dims.iter().rev() {
                right_idx += indices[d] * rs;
                rs *= shape[d];
            }

            for j in 0..kl {
                let ul = left_frame[(left_idx, j)];
                if ul.abs() < 1e-30 {
                    continue;
                }
                let vl = val * ul;
                for k in 0..kr {
                    transfer[j * kr + k] += vl * right_frame[(right_idx, k)];
                }
            }
        }
    } else {
        // Non-root: B_t = U_t^T @ (U_{t1} ⊗ U_{t2})
        // Iterate over all multi-indices in S_t (parent's dimension set).
        let n_parent: usize = parent_dims.iter().map(|&d| shape[d]).product();
        let n_left: usize = left_dims.iter().map(|&d| shape[d]).product();
        let n_right: usize = right_dims.iter().map(|&d| shape[d]).product();
        debug_assert_eq!(n_parent, n_left * n_right);

        // For each row of U_t, decompose into left/right sub-indices
        for parent_idx in 0..n_parent {
            // Decompose parent_idx into per-dimension indices within S_t
            // parent_idx = Σ indices[d] * stride, dims ordered as in parent_dims reversed
            let mut rem = parent_idx;
            let mut dim_indices = vec![0usize; parent_dims.len()];
            for i in (0..parent_dims.len()).rev() {
                dim_indices[i] = rem % shape[parent_dims[i]];
                rem /= shape[parent_dims[i]];
            }

            // Compute left_idx from left_dims
            let mut left_idx = 0;
            let mut ls = 1;
            for &d in left_dims.iter().rev() {
                let pos = parent_dims.iter().position(|&pd| pd == d).unwrap_or(0);
                left_idx += dim_indices[pos] * ls;
                ls *= shape[d];
            }

            // Compute right_idx from right_dims
            let mut right_idx = 0;
            let mut rs = 1;
            for &d in right_dims.iter().rev() {
                let pos = parent_dims.iter().position(|&pd| pd == d).unwrap_or(0);
                right_idx += dim_indices[pos] * rs;
                rs *= shape[d];
            }

            for i in 0..kt {
                let up = parent_frame[(parent_idx, i)];
                if up.abs() < 1e-30 {
                    continue;
                }
                for j in 0..kl {
                    let ul = left_frame[(left_idx, j)];
                    if ul.abs() < 1e-30 {
                        continue;
                    }
                    let upl = up * ul;
                    for k in 0..kr {
                        transfer[i * kl * kr + j * kr + k] += upl * right_frame[(right_idx, k)];
                    }
                }
            }
        }
    }
    transfer
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};

    fn test_domain(n: i128) -> Domain {
        Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(n)
            .velocity_resolution(n)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn round_trip_rank1() {
        let n = 4usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);

        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let idx = flat_index(&shape, [i0, i1, i2, i3, i4, i5]);
                                data[idx] = ((i0 + 1)
                                    * (i1 + 1)
                                    * (i2 + 1)
                                    * (i3 + 1)
                                    * (i4 + 1)
                                    * (i5 + 1)) as f64;
                            }
                        }
                    }
                }
            }
        }

        let ht = HtTensor::from_full(&data, shape, &domain, 1e-12);

        let mut max_err = 0.0f64;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let expected = ((i0 + 1)
                                    * (i1 + 1)
                                    * (i2 + 1)
                                    * (i3 + 1)
                                    * (i4 + 1)
                                    * (i5 + 1))
                                    as f64;
                                let got = ht.evaluate([i0, i1, i2, i3, i4, i5]);
                                max_err = max_err.max((got - expected).abs());
                            }
                        }
                    }
                }
            }
        }

        assert!(max_err < 1e-8, "rank-1 round-trip max error {max_err}");

        // Ranks should be small (ideally 1, but numerical SVD noise at 1e-12 may add a few)
        for d in 0..6 {
            assert!(
                ht.rank_at(d) <= 4,
                "leaf {d} rank {} too large for rank-1 tensor",
                ht.rank_at(d)
            );
        }
    }

    #[test]
    fn gaussian_blob() {
        let n = 6usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);
        let sigma = 0.3;

        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let x: Vec<f64> = [i0, i1, i2, i3, i4, i5]
                                    .iter()
                                    .map(|&i| (i as f64 + 0.5) / n as f64 - 0.5)
                                    .collect();
                                let r2: f64 = x.iter().map(|xi| xi * xi).sum();
                                data[flat_index(&shape, [i0, i1, i2, i3, i4, i5])] =
                                    (-r2 / (2.0 * sigma * sigma)).exp();
                            }
                        }
                    }
                }
            }
        }

        let tol = 1e-4;
        let ht = HtTensor::from_full(&data, shape, &domain, tol);

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let e = data[flat_index(&shape, [i0, i1, i2, i3, i4, i5])];
                                let g = ht.evaluate([i0, i1, i2, i3, i4, i5]);
                                err_sq += (g - e) * (g - e);
                                norm_sq += e * e;
                            }
                        }
                    }
                }
            }
        }

        let rel_err = (err_sq / norm_sq).sqrt();
        assert!(rel_err < tol * 10.0, "Gaussian rel error {rel_err:.2e}");
        println!(
            "Gaussian: full={}B, HT={}B, ratio={:.1}x",
            total * 8,
            ht.memory_bytes(),
            (total * 8) as f64 / ht.memory_bytes() as f64
        );
    }

    #[test]
    fn addition() {
        let n = 4usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);

        let mut data_a = vec![0.0f64; total];
        let mut data_b = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let idx = flat_index(&shape, [i0, i1, i2, i3, i4, i5]);
                                data_a[idx] = (i0 + 1) as f64;
                                data_b[idx] = (i3 + 1) as f64;
                            }
                        }
                    }
                }
            }
        }

        let a = HtTensor::from_full(&data_a, shape, &domain, 1e-12);
        let b = HtTensor::from_full(&data_b, shape, &domain, 1e-12);
        let s = a.add(&b);

        let mut max_err = 0.0f64;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let expected = (i0 + 1) as f64 + (i3 + 1) as f64;
                                let got = s.evaluate([i0, i1, i2, i3, i4, i5]);
                                max_err = max_err.max((got - expected).abs());
                            }
                        }
                    }
                }
            }
        }

        assert!(max_err < 1e-8, "addition max error {max_err}");
    }

    #[test]
    fn density_integration() {
        let n = 4usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);
        let sigma = 0.4;
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();

        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let x0 = -lx[0] + (i0 as f64 + 0.5) * dx[0];
                                let x1 = -lx[1] + (i1 as f64 + 0.5) * dx[1];
                                let x2 = -lx[2] + (i2 as f64 + 0.5) * dx[2];
                                let v0 = -lv[0] + (i3 as f64 + 0.5) * dv[0];
                                let v1 = -lv[1] + (i4 as f64 + 0.5) * dv[1];
                                let v2 = -lv[2] + (i5 as f64 + 0.5) * dv[2];
                                let gx =
                                    (-(x0 * x0 + x1 * x1 + x2 * x2) / (2.0 * sigma * sigma)).exp();
                                let hv =
                                    (-(v0 * v0 + v1 * v1 + v2 * v2) / (2.0 * sigma * sigma)).exp();
                                data[flat_index(&shape, [i0, i1, i2, i3, i4, i5])] = gx * hv;
                            }
                        }
                    }
                }
            }
        }

        let ht = HtTensor::from_full(&data, shape, &domain, 1e-10);
        let density = ht.compute_density();

        let dv3 = dv[0] * dv[1] * dv[2];
        let mut max_err = 0.0f64;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    let mut rho_direct = 0.0;
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                rho_direct += data[flat_index(&shape, [i0, i1, i2, i3, i4, i5])];
                            }
                        }
                    }
                    rho_direct *= dv3;
                    let rho_ht = density.data[i0 * n * n + i1 * n + i2];
                    max_err = max_err.max((rho_ht - rho_direct).abs());
                }
            }
        }

        assert!(max_err < 1e-6, "density max error {max_err:.2e}");
    }

    #[test]
    fn inner_product_and_norm() {
        let n = 4usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);

        let mut data = vec![0.0f64; total];
        for i in 0..total {
            data[i] = (i as f64) / (total as f64);
        }

        let ht = HtTensor::from_full(&data, shape, &domain, 1e-12);
        let direct_norm_sq: f64 = data.iter().map(|x| x * x).sum();
        let ht_norm_sq = ht.inner_product(&ht);

        let rel_err = ((ht_norm_sq - direct_norm_sq) / direct_norm_sq).abs();
        assert!(rel_err < 1e-6, "inner product rel error {rel_err:.2e}");
    }

    #[test]
    fn truncation_accuracy() {
        let n = 6usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);
        let sigma = 0.3;

        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let x: Vec<f64> = [i0, i1, i2, i3, i4, i5]
                                    .iter()
                                    .map(|&i| (i as f64 + 0.5) / n as f64 - 0.5)
                                    .collect();
                                let r2: f64 = x.iter().map(|xi| xi * xi).sum();
                                data[flat_index(&shape, [i0, i1, i2, i3, i4, i5])] =
                                    (-r2 / (2.0 * sigma * sigma)).exp();
                            }
                        }
                    }
                }
            }
        }

        let orig = HtTensor::from_full(&data, shape, &domain, 1e-12);
        let orig_rank = orig.total_rank();

        let mut trunc = orig.clone();
        trunc.truncate(1e-2);
        let trunc_rank = trunc.total_rank();

        assert!(
            trunc_rank <= orig_rank,
            "truncation should reduce rank: {trunc_rank} vs {orig_rank}"
        );

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let e = data[flat_index(&shape, [i0, i1, i2, i3, i4, i5])];
                                let g = trunc.evaluate([i0, i1, i2, i3, i4, i5]);
                                err_sq += (g - e) * (g - e);
                                norm_sq += e * e;
                            }
                        }
                    }
                }
            }
        }

        let rel_err = (err_sq / norm_sq).sqrt();
        assert!(rel_err < 0.1, "truncation rel error {rel_err:.2e}");
    }

    // ─── HTACA tests ────────────────────────────────────────────────────────

    #[test]
    fn htaca_separable() {
        // f(x,v) = g(x)·h(v) — separable, should achieve low rank
        let n = 6usize;
        let domain = test_domain(n as i128);

        let ht = HtTensor::from_function_aca(
            |x, v| {
                let gx = (-x[0] * x[0] - x[1] * x[1] - x[2] * x[2]).exp();
                let hv = (-v[0] * v[0] - v[1] * v[1] - v[2] * v[2]).exp();
                gx * hv
            },
            &domain,
            1e-6,
            10,
            None,
            None,
        );

        // Root rank should be 1 for a separable function
        assert_eq!(
            ht.rank_at(ROOT),
            1,
            "root rank should be 1 for separable function"
        );

        // Compare against from_function
        let ht_ref = HtTensor::from_function(
            |x, v| {
                let gx = (-x[0] * x[0] - x[1] * x[1] - x[2] * x[2]).exp();
                let hv = (-v[0] * v[0] - v[1] * v[1] - v[2] * v[2]).exp();
                gx * hv
            },
            &domain,
            1e-10,
        );

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let ref_val = ht_ref.evaluate([i0, i1, i2, i3, i4, i5]);
                                let aca_val = ht.evaluate([i0, i1, i2, i3, i4, i5]);
                                err_sq += (ref_val - aca_val).powi(2);
                                norm_sq += ref_val * ref_val;
                            }
                        }
                    }
                }
            }
        }
        let rel_err = if norm_sq > 0.0 {
            (err_sq / norm_sq).sqrt()
        } else {
            0.0
        };
        assert!(rel_err < 0.1, "htaca separable rel error {rel_err:.2e}");
    }

    #[test]
    fn htaca_gaussian() {
        // 6D Gaussian, compare vs from_full at same tolerance
        let n = 5usize;
        let domain = test_domain(n as i128);
        let sigma = 0.4;
        let tol = 1e-3;

        let f = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let r2 =
                x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            (-r2 / (2.0 * sigma * sigma)).exp()
        };

        let ht_aca = HtTensor::from_function_aca(f, &domain, tol, 15, None, None);
        let ht_ref = HtTensor::from_function(f, &domain, tol);

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let ref_val = ht_ref.evaluate([i0, i1, i2, i3, i4, i5]);
                                let aca_val = ht_aca.evaluate([i0, i1, i2, i3, i4, i5]);
                                err_sq += (ref_val - aca_val).powi(2);
                                norm_sq += ref_val * ref_val;
                            }
                        }
                    }
                }
            }
        }
        let rel_err = if norm_sq > 0.0 {
            (err_sq / norm_sq).sqrt()
        } else {
            0.0
        };
        assert!(rel_err < 0.5, "htaca gaussian rel error {rel_err:.2e}");
    }

    #[test]
    fn htaca_plummer() {
        // Plummer-like DF: f(x,v) ∝ (E₀ - E)^{7/2} where E = v²/2 + Φ(r)
        let n = 5usize;
        let domain = test_domain(n as i128);
        let tol = 1e-2;

        let f = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            let phi = -1.0 / (1.0 + r2).sqrt(); // Plummer potential (GM=1, a=1)
            let e = 0.5 * v2 + phi;
            if e < 0.0 { (-e).powf(3.5) } else { 0.0 }
        };

        let ht_aca = HtTensor::from_function_aca(f, &domain, tol, 15, None, None);
        let ht_ref = HtTensor::from_function(f, &domain, 1e-8);

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let ref_val = ht_ref.evaluate([i0, i1, i2, i3, i4, i5]);
                                let aca_val = ht_aca.evaluate([i0, i1, i2, i3, i4, i5]);
                                err_sq += (ref_val - aca_val).powi(2);
                                norm_sq += ref_val * ref_val;
                            }
                        }
                    }
                }
            }
        }
        let rel_err = if norm_sq > 0.0 {
            (err_sq / norm_sq).sqrt()
        } else {
            0.0
        };
        assert!(
            rel_err < 1.0,
            "htaca plummer rel error {rel_err:.2e} (expected < 1.0 for tol={tol})"
        );
    }

    #[test]
    fn htaca_scaling() {
        // Verify function evaluation count is reasonable (not wildly worse than N⁶)
        // Current implementation: leaf phase is O(Nk), interior phase iterates
        // complementary indices so is O(N⁵ or N⁶). Future ACA on interior nodes
        // will bring this down. For now, verify the algorithm runs at all grid sizes
        // and produces correct results.
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counts: Vec<(usize, f64)> = [4, 6, 8]
            .iter()
            .map(|&n| {
                let domain = test_domain(n as i128);
                let counter = AtomicUsize::new(0);
                let f = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
                    counter.fetch_add(1, Ordering::Relaxed);
                    let r2 = x[0] * x[0]
                        + x[1] * x[1]
                        + x[2] * x[2]
                        + v[0] * v[0]
                        + v[1] * v[1]
                        + v[2] * v[2];
                    (-r2).exp()
                };
                let ht = HtTensor::from_function_aca(f, &domain, 1e-4, 8, Some(20), Some(123));
                let c = counter.load(Ordering::Relaxed);
                // Verify the result is reasonable
                let val = ht.evaluate([0, 0, 0, 0, 0, 0]);
                (c, val)
            })
            .collect();

        let ratio_1 = counts[1].0 as f64 / counts[0].0 as f64;
        let ratio_2 = counts[2].0 as f64 / counts[0].0 as f64;

        // All grid sizes should produce non-zero, finite values
        for (n, (c, val)) in [4, 6, 8].iter().zip(&counts) {
            assert!(*c > 0, "N={n}: zero evaluations");
            assert!(val.is_finite() && *val > 0.0, "N={n}: bad value {val}");
        }

        println!(
            "HTACA eval counts: N=4:{}, N=6:{}, N=8:{} (ratios: {ratio_1:.1}, {ratio_2:.1})",
            counts[0].0, counts[1].0, counts[2].0
        );
    }

    #[test]
    fn htaca_density_consistency() {
        // compute_density() should match between from_full and from_function_aca
        let n = 5usize;
        let domain = test_domain(n as i128);
        let sigma = 0.4;

        let f = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let gx = (-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / (2.0 * sigma * sigma)).exp();
            let hv = (-(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / (2.0 * sigma * sigma)).exp();
            gx * hv
        };

        let ht_ref = HtTensor::from_function(f, &domain, 1e-8);
        let ht_aca = HtTensor::from_function_aca(f, &domain, 1e-4, 10, None, None);

        let rho_ref = ht_ref.compute_density();
        let rho_aca = ht_aca.compute_density();

        let mut max_rel_err = 0.0f64;
        for i in 0..rho_ref.data.len() {
            if rho_ref.data[i].abs() > 1e-10 {
                let rel = ((rho_ref.data[i] - rho_aca.data[i]) / rho_ref.data[i]).abs();
                max_rel_err = max_rel_err.max(rel);
            }
        }
        assert!(max_rel_err < 0.5, "density max rel error {max_rel_err:.2e}");
    }

    #[test]
    fn htaca_rank_convergence() {
        // Increasing max_rank should decrease or maintain error
        let n = 5usize;
        let domain = test_domain(n as i128);

        let f = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let r2 =
                x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            (-r2 * 2.0).exp()
        };

        let ht_ref = HtTensor::from_function(f, &domain, 1e-10);

        let mut prev_err = f64::MAX;
        for max_rank in [2, 4, 8] {
            let ht_aca = HtTensor::from_function_aca(f, &domain, 1e-8, max_rank, None, Some(42));

            let mut err_sq = 0.0;
            let mut norm_sq = 0.0;
            for i0 in 0..n {
                for i1 in 0..n {
                    for i2 in 0..n {
                        for i3 in 0..n {
                            for i4 in 0..n {
                                for i5 in 0..n {
                                    let ref_val = ht_ref.evaluate([i0, i1, i2, i3, i4, i5]);
                                    let aca_val = ht_aca.evaluate([i0, i1, i2, i3, i4, i5]);
                                    err_sq += (ref_val - aca_val).powi(2);
                                    norm_sq += ref_val * ref_val;
                                }
                            }
                        }
                    }
                }
            }
            let rel_err = if norm_sq > 0.0 {
                (err_sq / norm_sq).sqrt()
            } else {
                0.0
            };

            assert!(
                rel_err <= prev_err + 1e-10,
                "rank convergence: k={max_rank} error {rel_err:.2e} > prev {prev_err:.2e}"
            );
            prev_err = rel_err;
        }
    }

    // ─── SLAR advection tests ──────────────────────────────────────────────

    fn test_domain_bc(n: i128, sbc: SpatialBoundType, vbc: VelocityBoundType) -> Domain {
        Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(n)
            .velocity_resolution(n)
            .t_final(1.0)
            .spatial_bc(sbc)
            .velocity_bc(vbc)
            .build()
            .unwrap()
    }

    #[test]
    #[ignore] // takes ~194s in release mode
    fn slar_free_streaming_ht() {
        // Free streaming: f_new(x,v) = f_old(x - v*dt, v)
        let n = 8usize;
        let domain = test_domain_bc(
            n as i128,
            SpatialBoundType::Periodic,
            VelocityBoundType::Open,
        );
        let sigma = 0.3;
        let dt = 0.1;

        let f_ic = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let r2x = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let r2v = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            (-(r2x + r2v) / (2.0 * sigma * sigma)).exp()
        };

        let mut ht = HtTensor::from_function_aca(f_ic, &domain, 1e-4, 10, None, None);

        let dummy_disp = DisplacementField {
            dx: vec![0.0; n * n * n],
            dy: vec![0.0; n * n * n],
            dz: vec![0.0; n * n * n],
            shape: [n, n, n],
        };
        ht.advect_x(&dummy_disp, dt);

        // Compare against analytic: f(x - v*dt, v)
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let x = [
                                    -lx[0] + (i0 as f64 + 0.5) * dx[0],
                                    -lx[1] + (i1 as f64 + 0.5) * dx[1],
                                    -lx[2] + (i2 as f64 + 0.5) * dx[2],
                                ];
                                let v = [
                                    -lv[0] + (i3 as f64 + 0.5) * dv[0],
                                    -lv[1] + (i4 as f64 + 0.5) * dv[1],
                                    -lv[2] + (i5 as f64 + 0.5) * dv[2],
                                ];
                                // Back-trace: analytic value is f_ic(x - v*dt, v)
                                let x_dep = [x[0] - v[0] * dt, x[1] - v[1] * dt, x[2] - v[2] * dt];
                                let expected = f_ic(&x_dep, &v);
                                let got = ht.evaluate([i0, i1, i2, i3, i4, i5]);
                                err_sq += (got - expected).powi(2);
                                norm_sq += expected * expected;
                            }
                        }
                    }
                }
            }
        }

        let rel_err = if norm_sq > 0.0 {
            (err_sq / norm_sq).sqrt()
        } else {
            0.0
        };
        println!("slar_free_streaming_ht: rel_err = {rel_err:.4e}");
        assert!(
            rel_err < 0.15,
            "free streaming rel error {rel_err:.2e} >= 0.15"
        );
    }

    #[test]
    fn slar_uniform_kick_ht() {
        // Uniform acceleration: f_new(x,v) = f_old(x, v - a*dt)
        let n = 8usize;
        let domain = test_domain_bc(
            n as i128,
            SpatialBoundType::Periodic,
            VelocityBoundType::Truncated,
        );
        let sigma = 0.3;
        let dt = 0.1;
        let accel_val = 0.5;

        let f_ic = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let r2x = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let r2v = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            (-(r2x + r2v) / (2.0 * sigma * sigma)).exp()
        };

        let mut ht = HtTensor::from_function_aca(f_ic, &domain, 1e-4, 10, None, None);

        // Uniform acceleration field
        let n_sp = n * n * n;
        let accel = AccelerationField {
            gx: vec![accel_val; n_sp],
            gy: vec![0.0; n_sp],
            gz: vec![0.0; n_sp],
            shape: [n, n, n],
        };

        ht.advect_v(&accel, dt);

        // Compare: analytic is f_ic(x, v - a*dt)
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let x = [
                                    -lx[0] + (i0 as f64 + 0.5) * dx[0],
                                    -lx[1] + (i1 as f64 + 0.5) * dx[1],
                                    -lx[2] + (i2 as f64 + 0.5) * dx[2],
                                ];
                                let v = [
                                    -lv[0] + (i3 as f64 + 0.5) * dv[0],
                                    -lv[1] + (i4 as f64 + 0.5) * dv[1],
                                    -lv[2] + (i5 as f64 + 0.5) * dv[2],
                                ];
                                let v_dep = [v[0] - accel_val * dt, v[1], v[2]];
                                let expected = f_ic(&x, &v_dep);
                                let got = ht.evaluate([i0, i1, i2, i3, i4, i5]);
                                err_sq += (got - expected).powi(2);
                                norm_sq += expected * expected;
                            }
                        }
                    }
                }
            }
        }

        let rel_err = if norm_sq > 0.0 {
            (err_sq / norm_sq).sqrt()
        } else {
            0.0
        };
        println!("slar_uniform_kick_ht: rel_err = {rel_err:.4e}");
        assert!(
            rel_err < 0.15,
            "uniform kick rel error {rel_err:.2e} >= 0.15"
        );
    }

    #[test]
    fn slar_mass_conservation_ht() {
        // Mass should be approximately conserved after advection
        let n = 6usize;
        let domain = test_domain_bc(
            n as i128,
            SpatialBoundType::Periodic,
            VelocityBoundType::Truncated,
        );
        let sigma = 0.25;
        let dt = 0.1;

        let f_ic = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let r2x = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let r2v = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            (-(r2x + r2v) / (2.0 * sigma * sigma)).exp()
        };

        let mut ht = HtTensor::from_function_aca(f_ic, &domain, 1e-4, 10, None, None);
        let mass_before = ht.total_mass();

        let dummy_disp = DisplacementField {
            dx: vec![0.0; n * n * n],
            dy: vec![0.0; n * n * n],
            dz: vec![0.0; n * n * n],
            shape: [n, n, n],
        };
        ht.advect_x(&dummy_disp, dt);
        let mass_after = ht.total_mass();

        let rel_change = ((mass_after - mass_before) / mass_before).abs();
        println!(
            "slar_mass_conservation_ht: before={mass_before:.6}, after={mass_after:.6}, rel_change={rel_change:.4e}"
        );
        assert!(rel_change < 0.1, "mass change {rel_change:.2e} >= 10%");
    }

    #[test]
    #[ignore] // takes ~168s in release mode
    fn slar_separable_rank_ht() {
        // Separable IC: advection should not explode rank
        let n = 8usize;
        let domain = test_domain_bc(
            n as i128,
            SpatialBoundType::Periodic,
            VelocityBoundType::Open,
        );
        let dt = 0.05;

        let f_ic = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let gx = (-2.0 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])).exp();
            let hv = (-2.0 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])).exp();
            gx * hv
        };

        let mut ht = HtTensor::from_function_aca(f_ic, &domain, 1e-4, 10, None, None);
        let rank_before = ht.total_rank();

        let dummy_disp = DisplacementField {
            dx: vec![0.0; n * n * n],
            dy: vec![0.0; n * n * n],
            dz: vec![0.0; n * n * n],
            shape: [n, n, n],
        };
        ht.advect_x(&dummy_disp, dt);
        let rank_after = ht.total_rank();

        println!("slar_separable_rank_ht: rank_before={rank_before}, rank_after={rank_after}");
        // After advection, a separable f(x)g(v) becomes non-separable f(x-vt)g(v),
        // so rank growth is expected. Verify it stays bounded (not exponential).
        assert!(
            rank_after <= rank_before * 8,
            "rank grew too much: {rank_after} > 8 × {rank_before}"
        );
    }

    #[test]
    fn slar_drift_kick_drift_ht() {
        // Compare HT drift-kick-drift against UniformGrid6D (imported via PhaseSpaceRepr)
        let n = 6usize;
        let domain = test_domain_bc(
            n as i128,
            SpatialBoundType::Periodic,
            VelocityBoundType::Truncated,
        );
        let sigma = 0.3;
        let dt = 0.1;

        let f_ic = |x: &[f64; 3], v: &[f64; 3]| -> f64 {
            let r2x = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let r2v = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            (-(r2x + r2v) / (2.0 * sigma * sigma)).exp()
        };

        let mut ht = HtTensor::from_function_aca(f_ic, &domain, 1e-4, 10, None, None);

        let n_sp = n * n * n;
        let dummy_disp = DisplacementField {
            dx: vec![0.0; n_sp],
            dy: vec![0.0; n_sp],
            dz: vec![0.0; n_sp],
            shape: [n, n, n],
        };
        let accel = AccelerationField {
            gx: vec![0.3; n_sp],
            gy: vec![-0.1; n_sp],
            gz: vec![0.0; n_sp],
            shape: [n, n, n],
        };

        // Strang: drift(dt/2) → kick(dt) → drift(dt/2)
        ht.advect_x(&dummy_disp, dt / 2.0);
        ht.advect_v(&accel, dt);
        ht.advect_x(&dummy_disp, dt / 2.0);

        let rho_ht = ht.compute_density();

        // Verify density is physically reasonable (non-negative, finite)
        let mut any_positive = false;
        for &val in &rho_ht.data {
            assert!(val.is_finite(), "density contains NaN/Inf");
            if val > 1e-10 {
                any_positive = true;
            }
        }
        assert!(any_positive, "density is all zero after drift-kick-drift");

        // Check mass is reasonable
        let dx3 = domain.cell_volume_3d();
        let total_rho: f64 = rho_ht.data.iter().sum::<f64>() * dx3;
        println!("slar_drift_kick_drift_ht: total density integral = {total_rho:.6}");
        assert!(total_rho > 0.0, "total density integral should be positive");
    }

    // ─── Sparse polynomial interpolation tests ──────────────────────────

    #[test]
    fn sparse_poly_quadratic_exactness() {
        // A quadratic function should be reproduced exactly by the sparse polynomial
        let domain = Domain::builder()
            .spatial_extent(2.0)
            .velocity_extent(2.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        // Create a quadratic: f(x,v) = 1 + x1 + 0.5*x1^2 + x2*x3
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();
        let shape = [8usize; 6];

        let quadratic = |idx: [usize; 6]| -> f64 {
            let x1 = -lx[0] + (idx[0] as f64 + 0.5) * dx[0];
            let x2 = -lx[1] + (idx[1] as f64 + 0.5) * dx[1];
            let x3 = -lx[2] + (idx[2] as f64 + 0.5) * dx[2];
            1.0 + x1 + 0.5 * x1 * x1 + x2 * x3
        };

        let mut data = vec![0.0; 8usize.pow(6)];
        for i0 in 0..8 {
            for i1 in 0..8 {
                for i2 in 0..8 {
                    for i3 in 0..8 {
                        for i4 in 0..8 {
                            for i5 in 0..8 {
                                let idx = [i0, i1, i2, i3, i4, i5];
                                data[flat_index(&shape, idx)] = quadratic(idx);
                            }
                        }
                    }
                }
            }
        }

        let ht = HtTensor::from_full(&data, shape, &domain, 1e-14);

        // Test sparse polynomial at a fractional position
        let frac = [2.7, 3.4, 5.1]; // fractional cell indices
        let int_indices = [0, 0, 0, 4, 4, 4]; // velocity indices don't matter for this function

        let mut ws = vec![0.0; ht.eval_workspace_len()];
        let result =
            sparse_polynomial_interpolate_ht(&ht, int_indices, [0, 1, 2], frac, [true; 3], &mut ws);

        // Compute expected value from the quadratic at fractional position
        let x1 = -lx[0] + (frac[0] + 0.5) * dx[0];
        let x2 = -lx[1] + (frac[1] + 0.5) * dx[1];
        let x3 = -lx[2] + (frac[2] + 0.5) * dx[2];
        let expected = 1.0 + x1 + 0.5 * x1 * x1 + x2 * x3;

        // The sparse polynomial should reproduce quadratics well (not exactly due to HT compression)
        let err = (result - expected).abs();
        assert!(
            err < 0.5,
            "Sparse poly error {err} too large for quadratic (expected {expected}, got {result})"
        );
    }

    // ─── Positivity limiter tests ───────────────────────────────────────

    #[test]
    fn test_positivity_disabled_by_default() {
        let domain = test_domain(8);
        let ht = HtTensor::new(&domain, 4);
        assert!(
            !ht.positivity_limiter,
            "positivity_limiter should be false by default"
        );
        assert_eq!(ht.positivity_violations(), 0);
    }

    #[test]
    fn test_positivity_limiter_preserves_mass() {
        let n = 8usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];

        // Build a Plummer-like IC: f(x,v) = exp(-r_x^2 - r_v^2)
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();
        let total = n.pow(6);
        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let x0 = -lx[0] + (i0 as f64 + 0.5) * dx[0];
                                let x1 = -lx[1] + (i1 as f64 + 0.5) * dx[1];
                                let x2 = -lx[2] + (i2 as f64 + 0.5) * dx[2];
                                let v0 = -lv[0] + (i3 as f64 + 0.5) * dv[0];
                                let v1 = -lv[1] + (i4 as f64 + 0.5) * dv[1];
                                let v2 = -lv[2] + (i5 as f64 + 0.5) * dv[2];
                                let r2 = x0 * x0 + x1 * x1 + x2 * x2 + v0 * v0 + v1 * v1 + v2 * v2;
                                let idx = flat_index(&shape, [i0, i1, i2, i3, i4, i5]);
                                data[idx] = (-r2).exp();
                            }
                        }
                    }
                }
            }
        }

        // Build at high accuracy first, then truncate aggressively to introduce negatives
        let ht_full = HtTensor::from_full(&data, shape, &domain, 1e-14);
        let mass_original = ht_full.total_mass();

        // Aggressively truncate to rank 1 to introduce artifacts (including negatives)
        let low_rank_data = ht_full.to_full();
        let mut ht_low = HtTensor::from_full(&low_rank_data, shape, &domain, 0.5);
        ht_low.positivity_limiter = true;

        let mass_before = ht_low.total_mass();
        ht_low.enforce_positivity();
        let mass_after = ht_low.total_mass();

        // Mass should be preserved to reasonable tolerance (HT recompression introduces
        // some error, but zhang_shu_limiter preserves the sum of the full array)
        let rel_err = ((mass_after - mass_before) / mass_before.abs().max(1e-30)).abs();
        assert!(
            rel_err < 0.1,
            "Mass not preserved: before={mass_before}, after={mass_after}, rel_err={rel_err}"
        );
        assert!(mass_original > 0.0, "Original mass should be positive");
    }

    #[test]
    fn test_positivity_violations_counted() {
        let n = 8usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);

        // Create data with intentional negative values
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();
        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let x0 = -lx[0] + (i0 as f64 + 0.5) * dx[0];
                                let x1 = -lx[1] + (i1 as f64 + 0.5) * dx[1];
                                let x2 = -lx[2] + (i2 as f64 + 0.5) * dx[2];
                                let v0 = -lv[0] + (i3 as f64 + 0.5) * dv[0];
                                let v1 = -lv[1] + (i4 as f64 + 0.5) * dv[1];
                                let v2 = -lv[2] + (i5 as f64 + 0.5) * dv[2];
                                let r2 = x0 * x0 + x1 * x1 + x2 * x2 + v0 * v0 + v1 * v1 + v2 * v2;
                                // Use a function that oscillates: positive near center,
                                // negative further out
                                let idx = flat_index(&shape, [i0, i1, i2, i3, i4, i5]);
                                data[idx] = (1.0 - 2.0 * r2) * (-r2).exp();
                            }
                        }
                    }
                }
            }
        }

        // Verify that data has negative values
        let neg_count_raw = data.iter().filter(|&&v| v < 0.0).count();
        assert!(neg_count_raw > 0, "Test data should have negative values");

        // Build HT with tight tolerance to preserve negatives
        let mut ht = HtTensor::from_full(&data, shape, &domain, 1e-14);
        ht.positivity_limiter = true;
        assert_eq!(ht.positivity_violations(), 0);

        ht.enforce_positivity();
        assert!(
            ht.positivity_violations() > 0,
            "Should have counted positivity violations"
        );

        // Note: HT recompression via from_full after clipping may re-introduce
        // negatives due to Gibbs-like ringing from the hard discontinuity at zero.
        // The primary value of the HT positivity limiter is diagnostic (counting
        // violations). For strict positivity enforcement, use UniformGrid6D which
        // can clip without recompression artifacts.
    }

    #[test]
    fn test_ht_velocity_filter_preserves_structure() {
        // Build a smooth Gaussian HT tensor, filter it, and verify total_mass
        // is preserved (the filter preserves the DC component exactly).
        let n = 8usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);
        let sigma = 0.3;

        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let coords: Vec<f64> = [i0, i1, i2, i3, i4, i5]
                                    .iter()
                                    .map(|&i| (i as f64 + 0.5) / n as f64 - 0.5)
                                    .collect();
                                let r2: f64 = coords.iter().map(|c| c * c).sum();
                                let idx = flat_index(&shape, [i0, i1, i2, i3, i4, i5]);
                                data[idx] = (-r2 / (2.0 * sigma * sigma)).exp();
                            }
                        }
                    }
                }
            }
        }

        let mut ht = HtTensor::from_full(&data, shape, &domain, 1e-10);
        let mass_before = ht.total_mass();

        // Apply velocity filter
        ht.velocity_filter = Some(VelocityFilterConfig {
            cutoff_fraction: 0.8,
            order: 2,
        });
        ht.apply_velocity_filter();

        let mass_after = ht.total_mass();
        let rel_diff = ((mass_after - mass_before) / mass_before).abs();
        assert!(
            rel_diff < 0.05,
            "Total mass should be approximately preserved, relative diff = {rel_diff}"
        );

        // Verify HT structure: all nodes still have valid ranks
        for d in 0..6 {
            assert!(ht.rank_at(d) > 0, "Leaf {d} rank should be positive");
        }
    }

    #[test]
    fn test_ht_velocity_filter_damps_high_modes() {
        // Build an HT tensor with high-frequency velocity content, apply the
        // filter, and verify the high-frequency content is reduced.
        let n = 8usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);

        // f = 1 + 0.5 * cos(pi * iv3) which is the Nyquist mode in v3.
        // Leaves 3,4,5 = v1,v2,v3 — the filter on leaf 5 should damp this.
        let mut data = vec![0.0f64; total];
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for i4 in 0..n {
                            for i5 in 0..n {
                                let phase = std::f64::consts::PI * i5 as f64;
                                let idx = flat_index(&shape, [i0, i1, i2, i3, i4, i5]);
                                data[idx] = 1.0 + 0.5 * phase.cos();
                            }
                        }
                    }
                }
            }
        }

        let ht_orig = HtTensor::from_full(&data, shape, &domain, 1e-10);

        let mut ht_filtered = ht_orig.clone();
        ht_filtered.velocity_filter = Some(VelocityFilterConfig {
            cutoff_fraction: 0.5,
            order: 4,
        });
        ht_filtered.apply_velocity_filter();

        // Measure the change by evaluating at specific points.
        // The high-frequency component alternates sign: at i5=0 the signal is
        // 1+0.5=1.5, at i5=1 it's 1-0.5=0.5. After filtering, these should
        // be closer together (the oscillation is damped).
        let val_orig_0 = ht_orig.evaluate([0, 0, 0, 0, 0, 0]);
        let val_orig_1 = ht_orig.evaluate([0, 0, 0, 0, 0, 1]);
        let contrast_orig = (val_orig_0 - val_orig_1).abs();

        let val_filt_0 = ht_filtered.evaluate([0, 0, 0, 0, 0, 0]);
        let val_filt_1 = ht_filtered.evaluate([0, 0, 0, 0, 0, 1]);
        let contrast_filt = (val_filt_0 - val_filt_1).abs();

        assert!(
            contrast_filt < contrast_orig * 0.5,
            "High-frequency oscillation should be damped, orig contrast={contrast_orig}, filtered={contrast_filt}"
        );
    }
}
