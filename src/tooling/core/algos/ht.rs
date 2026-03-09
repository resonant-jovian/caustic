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
    init::domain::Domain,
    phasespace::PhaseSpaceRepr,
    types::*,
};
use faer::Mat;
use rust_decimal::prelude::ToPrimitive;

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

/// A node in the HT dimension tree.
#[derive(Clone)]
pub enum HtNode {
    /// Leaf node: stores a basis matrix U_μ ∈ ℝ^{n_μ × k_μ}.
    Leaf {
        dim: usize,
        frame: Mat<f64>,
    },
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

/// Hierarchical Tucker tensor for 6D phase space.
#[derive(Clone)]
pub struct HtTensor {
    pub nodes: Vec<HtNode>,
    pub shape: [usize; 6],
    pub domain: Domain,
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
            left: 1, right: 2,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Interior node 7: {4,5}
        let kl = k.min(shape[4]);
        let kr = k.min(shape[5]);
        let kt = k.min(kl * kr);
        nodes.push(HtNode::Interior {
            left: 4, right: 5,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Interior node 8: {0,1,2}
        let kl = k.min(shape[0]);
        let kr = nodes[6].rank();
        let kt = k.min(kl * kr);
        nodes.push(HtNode::Interior {
            left: 0, right: 6,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Interior node 9: {3,4,5}
        let kl = k.min(shape[3]);
        let kr = nodes[7].rank();
        let kt = k.min(kl * kr);
        nodes.push(HtNode::Interior {
            left: 3, right: 7,
            transfer: vec![0.0; kt * kl * kr],
            ranks: [kt, kl, kr],
        });

        // Root node 10: {0..5}
        let kl = nodes[8].rank();
        let kr = nodes[9].rank();
        nodes.push(HtNode::Interior {
            left: 8, right: 9,
            transfer: vec![0.0; kl * kr],
            ranks: [1, kl, kr],
        });

        HtTensor { nodes, shape, domain: domain.clone() }
    }

    /// Convert a full 6D array to HT format via hierarchical SVD (HSVD).
    pub fn from_full(data: &[f64], shape: [usize; 6], domain: &Domain, tolerance: f64) -> Self {
        let _span = tracing::info_span!("ht_from_full").entered();
        assert_eq!(data.len(), shape.iter().product::<usize>(), "data/shape mismatch");

        // Per-node tolerance: ε / √(2d−3) = ε / 3
        let eps_node = tolerance / 3.0;

        // Dimension sets for each node
        let dim_sets: [&[usize]; NUM_NODES] = [
            &[0], &[1], &[2], &[3], &[4], &[5],
            &[1, 2],
            &[4, 5],
            &[0, 1, 2],
            &[3, 4, 5],
            &[0, 1, 2, 3, 4, 5],
        ];

        // Compute frames for all nodes via mode unfolding + SVD
        let mut frames: Vec<Mat<f64>> = Vec::with_capacity(NUM_NODES);

        for node_idx in 0..10 {
            let dims = dim_sets[node_idx];
            let mat = multi_mode_unfold(data, &shape, dims);
            let (u, s, _vt) = thin_svd(&mat);
            let rank = truncation_rank(&s, eps_node).max(1);
            frames.push(u.subcols(0, rank).to_owned());
        }

        // Root frame: trivially [1] (1×1)
        let mut root_frame = Mat::zeros(1, 1);
        root_frame[(0, 0)] = 1.0;
        frames.push(root_frame);

        // Children of each interior node
        let children: [(usize, usize); 5] = [
            (1, 2), (4, 5), (0, 6), (3, 7), (8, 9),
        ];

        let mut nodes: Vec<HtNode> = Vec::with_capacity(NUM_NODES);

        // Leaf nodes
        for d in 0..6 {
            nodes.push(HtNode::Leaf { dim: d, frame: frames[d].clone() });
        }

        // Interior nodes
        for (i, &(left, right)) in children.iter().enumerate() {
            let node_idx = 6 + i;
            let transfer = compute_transfer_tensor(
                data, &shape,
                dim_sets[node_idx], dim_sets[left], dim_sets[right],
                &frames[node_idx], &frames[left], &frames[right],
            );
            let kt = frames[node_idx].ncols();
            let kl = frames[left].ncols();
            let kr = frames[right].ncols();

            nodes.push(HtNode::Interior {
                left, right, transfer,
                ranks: [kt, kl, kr],
            });
        }

        HtTensor { nodes, shape, domain: domain.clone() }
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
        let lx = ext_f64(&domain.spatial);
        let lv = ext_f64_v(&domain.velocity);

        let total: usize = shape.iter().product();
        let mut data = vec![0.0f64; total];
        let [n0, n1, n2, n3, n4, n5] = shape;

        for i0 in 0..n0 {
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
                                data[flat_index(&shape, [i0, i1, i2, i3, i4, i5])] = f(&x, &v);
                            }
                        }
                    }
                }
            }
        }

        Self::from_full(&data, shape, domain, tolerance)
    }
}

// ─── Point evaluation ───────────────────────────────────────────────────────

impl HtTensor {
    /// Evaluate at a single 6D grid point. Cost: O(d·k³).
    pub fn evaluate(&self, indices: [usize; 6]) -> f64 {
        let root_vec = self.node_vector(ROOT, &indices);
        debug_assert_eq!(root_vec.len(), 1);
        root_vec[0]
    }

    /// Contracted vector at a node for given indices.
    fn node_vector(&self, node_idx: usize, indices: &[usize; 6]) -> Vec<f64> {
        match &self.nodes[node_idx] {
            HtNode::Leaf { dim, frame } => {
                let row = indices[*dim];
                (0..frame.ncols()).map(|c| frame[(row, c)]).collect()
            }
            HtNode::Interior { left, right, transfer, ranks } => {
                let [kt, kl, kr] = *ranks;
                let lv = self.node_vector(*left, indices);
                let rv = self.node_vector(*right, indices);
                contract_transfer(transfer, kt, kl, kr, &lv, &rv)
            }
        }
    }

    pub fn rank_at(&self, node_idx: usize) -> usize { self.nodes[node_idx].rank() }
    pub fn total_rank(&self) -> usize { self.nodes.iter().map(|n| n.rank()).sum() }

    pub fn memory_bytes(&self) -> usize {
        self.nodes.iter().map(|n| match n {
            HtNode::Leaf { frame, .. } => frame.nrows() * frame.ncols() * 8,
            HtNode::Interior { transfer, .. } => transfer.len() * 8,
        }).sum()
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
        let node = self.nodes[node_idx].clone();
        match node {
            HtNode::Leaf { dim, frame } => {
                let (q, r) = qr_decompose(&frame);
                self.nodes[node_idx] = HtNode::Leaf { dim, frame: q };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &r);
            }
            HtNode::Interior { left, right, transfer, ranks } if node_idx != ROOT => {
                let [kt, kl, kr] = ranks;
                let mat = vec_to_mat(&transfer, kt, kl * kr);
                let (q, r) = qr_decompose(&mat);
                let new_kt = q.ncols();
                let new_transfer = mat_to_vec(&q, new_kt, kl * kr);
                self.nodes[node_idx] = HtNode::Interior {
                    left, right, transfer: new_transfer, ranks: [new_kt, kl, kr],
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &r);
            }
            _ => {}
        }
    }

    fn absorb_r_into_parent(&mut self, parent_idx: usize, child_idx: usize, r: &Mat<f64>) {
        if let HtNode::Interior { left, right: _, ref mut transfer, ref mut ranks } = self.nodes[parent_idx] {
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
    pub fn truncate(&mut self, tolerance: f64) {
        let _span = tracing::info_span!("ht_truncate").entered();
        self.orthogonalize_left();
        let eps = tolerance / 3.0;
        for &ni in &[8, 9, 6, 7, 0, 1, 2, 3, 4, 5] {
            self.truncate_node(ni, eps);
        }
    }

    /// Truncate all nodes to a fixed maximum rank.
    pub fn truncate_to_rank(&mut self, max_rank: usize) {
        self.orthogonalize_left();
        for &ni in &[8, 9, 6, 7, 0, 1, 2, 3, 4, 5] {
            self.truncate_node_fixed(ni, max_rank);
        }
    }

    fn truncate_node(&mut self, node_idx: usize, eps: f64) {
        let node = self.nodes[node_idx].clone();
        match node {
            HtNode::Leaf { dim, frame } => {
                let (u, s, vt) = thin_svd(&frame);
                let new_rank = truncation_rank(&s, eps).max(1);
                if new_rank >= frame.ncols() { return; }
                let new_frame = u.subcols(0, new_rank).to_owned();
                let sv = s_times_vt(&s, &vt, new_rank);
                self.nodes[node_idx] = HtNode::Leaf { dim, frame: new_frame };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &sv);
            }
            HtNode::Interior { left, right, transfer, ranks } if node_idx != ROOT => {
                let [kt, kl, kr] = ranks;
                let mat = vec_to_mat(&transfer, kt, kl * kr);
                let (u, s, vt) = thin_svd(&mat);
                let new_rank = truncation_rank(&s, eps).max(1);
                if new_rank >= kt { return; }
                let new_t = mat_to_vec(&u, new_rank, kl * kr);
                let sv = s_times_vt(&s, &vt, new_rank);
                self.nodes[node_idx] = HtNode::Interior {
                    left, right, transfer: new_t, ranks: [new_rank, kl, kr],
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &sv);
            }
            _ => {}
        }
    }

    fn truncate_node_fixed(&mut self, node_idx: usize, max_rank: usize) {
        let node = self.nodes[node_idx].clone();
        match node {
            HtNode::Leaf { dim, frame } => {
                if frame.ncols() <= max_rank { return; }
                let (u, s, vt) = thin_svd(&frame);
                let r = max_rank.min(u.ncols());
                let sv = s_times_vt(&s, &vt, r);
                self.nodes[node_idx] = HtNode::Leaf { dim, frame: u.subcols(0, r).to_owned() };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &sv);
            }
            HtNode::Interior { left, right, transfer, ranks } if node_idx != ROOT => {
                let [kt, kl, kr] = ranks;
                if kt <= max_rank { return; }
                let mat = vec_to_mat(&transfer, kt, kl * kr);
                let (u, s, vt) = thin_svd(&mat);
                let r = max_rank.min(u.ncols());
                let sv = s_times_vt(&s, &vt, r);
                self.nodes[node_idx] = HtNode::Interior {
                    left, right,
                    transfer: mat_to_vec(&u, r, kl * kr),
                    ranks: [r, kl, kr],
                };
                self.absorb_r_into_parent(parent_of(node_idx), node_idx, &sv);
            }
            _ => {}
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
                for j in 0..k1 { combined[(i, j)] = f1[(i, j)]; }
                for j in 0..k2 { combined[(i, k1 + j)] = f2[(i, j)]; }
            }
            nodes.push(HtNode::Leaf { dim, frame: combined });
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
                    left: l1, right: 9,
                    transfer: nt, ranks: [1, nkl, nkr],
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
                    left: l1, right: match ni { 6 => 2, 7 => 5, 8 => 6, 9 => 7, _ => unreachable!() },
                    transfer: nt, ranks: [nkt, nkl, nkr],
                });
            }
        }

        HtTensor { nodes, shape: self.shape, domain: self.domain.clone() }
    }

    /// Scale the tensor by a scalar.
    pub fn scale(&mut self, alpha: f64) {
        if let HtNode::Interior { ref mut transfer, .. } = self.nodes[ROOT] {
            for v in transfer.iter_mut() { *v *= alpha; }
        }
    }

    /// Inner product ⟨self, other⟩ via Gram matrices. O(dk⁴).
    pub fn inner_product(&self, other: &HtTensor) -> f64 {
        assert_eq!(self.shape, other.shape);
        let g = self.gram_matrix(ROOT, other);
        g[(0, 0)]
    }

    fn gram_matrix(&self, node_idx: usize, other: &HtTensor) -> Mat<f64> {
        match (&self.nodes[node_idx], &other.nodes[node_idx]) {
            (HtNode::Leaf { frame: f1, .. }, HtNode::Leaf { frame: f2, .. }) => {
                matmul_at_b(f1, f2)
            }
            (
                HtNode::Interior { left: l1, right: r1, transfer: t1, ranks: rk1 },
                HtNode::Interior { left: _, right: _, transfer: t2, ranks: rk2 },
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
                                if glv.abs() < 1e-30 { continue; }
                                for k1 in 0..kr1 {
                                    let b1 = t1[i1 * kl1 * kr1 + j1 * kr1 + k1];
                                    if b1.abs() < 1e-30 { continue; }
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
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    for i3 in 0..n3 {
                        for i4 in 0..n4 {
                            for i5 in 0..n5 {
                                let idx = flat_index(&self.shape, [i0, i1, i2, i3, i4, i5]);
                                data[idx] = self.evaluate([i0, i1, i2, i3, i4, i5]);
                            }
                        }
                    }
                }
            }
        }
        data
    }
}

// ─── PhaseSpaceRepr ─────────────────────────────────────────────────────────

impl PhaseSpaceRepr for HtTensor {
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

        let mut density = vec![0.0f64; nx1 * nx2 * nx3];

        for i0 in 0..nx1 {
            // eff_i0[k8] = sum_{j8} eff8[j8, k8] * f0[i0, j8]
            let mut eff_i0 = vec![0.0f64; kr8];
            for k8 in 0..kr8 {
                let mut s = 0.0;
                for j8 in 0..kl8 {
                    s += eff8[j8 * kr8 + k8] * f0[(i0, j8)];
                }
                eff_i0[k8] = s;
            }

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
                    density[i0 * nx2 * nx3 + i1 * nx3 + i2] = val * dv3;
                }
            }
        }

        DensityField { data: density, shape: [nx1, nx2, nx3] }
    }

    fn advect_x(&mut self, _displacement: &DisplacementField, _dt: f64) {
        todo!("HT advect_x requires SLAR method (Phase 2)")
    }

    fn advect_v(&mut self, _acceleration: &AccelerationField, _dt: f64) {
        todo!("HT advect_v requires SLAR method (Phase 2)")
    }

    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = ext_f64(&self.domain.spatial);
        let lv = ext_f64_v(&self.domain.velocity);
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;

        let ix = [
            ((position[0] + lx[0]) / dx[0]).floor().clamp(0.0, (nx1 - 1) as f64) as usize,
            ((position[1] + lx[1]) / dx[1]).floor().clamp(0.0, (nx2 - 1) as f64) as usize,
            ((position[2] + lx[2]) / dx[2]).floor().clamp(0.0, (nx3 - 1) as f64) as usize,
        ];
        let dv3 = dv[0] * dv[1] * dv[2];

        match order {
            0 => {
                let mut sum = 0.0;
                for iv1 in 0..nv1 { for iv2 in 0..nv2 { for iv3 in 0..nv3 {
                    sum += self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3]);
                }}}
                Tensor { data: vec![sum * dv3], rank: 0, shape: vec![] }
            }
            1 => {
                let mut vbar = [0.0; 3];
                let mut rho = 0.0;
                for iv1 in 0..nv1 { for iv2 in 0..nv2 { for iv3 in 0..nv3 {
                    let f = self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3]);
                    vbar[0] += f * (-lv[0] + (iv1 as f64 + 0.5) * dv[0]);
                    vbar[1] += f * (-lv[1] + (iv2 as f64 + 0.5) * dv[1]);
                    vbar[2] += f * (-lv[2] + (iv3 as f64 + 0.5) * dv[2]);
                    rho += f;
                }}}
                rho *= dv3;
                let s = if rho > 1e-30 { dv3 / rho } else { 0.0 };
                Tensor { data: vec![vbar[0]*s, vbar[1]*s, vbar[2]*s], rank: 1, shape: vec![3] }
            }
            2 => {
                let mut m2 = [0.0f64; 9];
                for iv1 in 0..nv1 { for iv2 in 0..nv2 { for iv3 in 0..nv3 {
                    let f = self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3]);
                    let v = [
                        -lv[0] + (iv1 as f64 + 0.5) * dv[0],
                        -lv[1] + (iv2 as f64 + 0.5) * dv[1],
                        -lv[2] + (iv3 as f64 + 0.5) * dv[2],
                    ];
                    for a in 0..3 { for b in 0..3 { m2[a*3+b] += f * v[a] * v[b]; } }
                }}}
                Tensor { data: m2.iter().map(|&x| x * dv3).collect(), rank: 2, shape: vec![3, 3] }
            }
            _ => Tensor { data: vec![], rank: order, shape: vec![] },
        }
    }

    fn total_mass(&self) -> f64 {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let vol = dx[0]*dx[1]*dx[2] * dv[0]*dv[1]*dv[2];

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
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let vol = dx[0]*dx[1]*dx[2] * dv[0]*dv[1]*dv[2];
        self.inner_product(self) * vol
    }

    fn entropy(&self) -> f64 {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let vol = dx[0]*dx[1]*dx[2] * dv[0]*dv[1]*dv[2];
        let data = self.to_full();
        data.iter().filter(|&&f| f > 0.0).map(|&f| -f * f.ln()).sum::<f64>() * vol
    }

    fn stream_count(&self) -> StreamCountField {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;
        let dv = self.domain.dv();
        let dv23 = dv[1] * dv[2];
        let mut out = vec![0u32; nx1 * nx2 * nx3];

        for ix1 in 0..nx1 { for ix2 in 0..nx2 { for ix3 in 0..nx3 {
            let marginal: Vec<f64> = (0..nv1).map(|iv1| {
                (0..nv2*nv3).map(|vi| {
                    let iv3 = vi % nv3;
                    let iv2 = vi / nv3;
                    self.evaluate([ix1, ix2, ix3, iv1, iv2, iv3])
                }).sum::<f64>() * dv23
            }).collect();
            let mut peaks = 0u32;
            for i in 1..nv1.saturating_sub(1) {
                if marginal[i] > marginal[i-1] && marginal[i] > marginal[i+1] { peaks += 1; }
            }
            out[ix1 * nx2 * nx3 + ix2 * nx3 + ix3] = peaks;
        }}}

        StreamCountField { data: out, shape: [nx1, nx2, nx3] }
    }

    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.shape;
        let dx = self.domain.dx();
        let lx = ext_f64(&self.domain.spatial);
        let ix = [
            ((position[0] + lx[0]) / dx[0]).floor().clamp(0.0, (nx1-1) as f64) as usize,
            ((position[1] + lx[1]) / dx[1]).floor().clamp(0.0, (nx2-1) as f64) as usize,
            ((position[2] + lx[2]) / dx[2]).floor().clamp(0.0, (nx3-1) as f64) as usize,
        ];
        (0..nv1*nv2*nv3).map(|vi| {
            let iv3 = vi % nv3;
            let iv2 = (vi / nv3) % nv2;
            let iv1 = vi / (nv2 * nv3);
            self.evaluate([ix[0], ix[1], ix[2], iv1, iv2, iv3])
        }).collect()
    }

    fn total_kinetic_energy(&self) -> f64 {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lv = ext_f64_v(&self.domain.velocity);
        let vol = dx[0]*dx[1]*dx[2] * dv[0]*dv[1]*dv[2];
        let [_, _, _, nv1, nv2, nv3] = self.shape;

        let data = self.to_full();
        let n_vel = nv1 * nv2 * nv3;
        let t: f64 = data.iter().enumerate().map(|(idx, &f)| {
            let vi = idx % n_vel;
            let iv3 = vi % nv3;
            let iv2 = (vi / nv3) % nv2;
            let iv1 = vi / (nv2 * nv3);
            let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
            let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
            let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
            f * (vx*vx + vy*vy + vz*vz)
        }).sum();
        0.5 * t * vol
    }

    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot {
        PhaseSpaceSnapshot { data: self.to_full(), shape: self.shape, time }
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

fn flat_index(shape: &[usize; 6], idx: [usize; 6]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for d in (0..6).rev() {
        flat += idx[d] * stride;
        stride *= shape[d];
    }
    flat
}

fn parent_of(child: usize) -> usize {
    match child {
        0 => 8, 1 => 6, 2 => 6,
        3 => 9, 4 => 7, 5 => 7,
        6 => 8, 7 => 9,
        8 => 10, 9 => 10,
        _ => panic!("no parent for node {child}"),
    }
}

fn leaf_data(node: &HtNode) -> (&Mat<f64>, usize) {
    match node {
        HtNode::Leaf { dim, frame } => (frame, *dim),
        _ => panic!("expected leaf"),
    }
}

fn get_interior(node: &HtNode) -> (usize, usize, usize, &[f64]) {
    match node {
        HtNode::Interior { ranks, transfer, .. } => (ranks[0], ranks[1], ranks[2], transfer),
        _ => panic!("expected interior"),
    }
}

fn interior_data(node: &HtNode) -> (usize, usize, &[f64], [usize; 3]) {
    match node {
        HtNode::Interior { left, right, transfer, ranks } => (*left, *right, transfer, *ranks),
        _ => panic!("expected interior"),
    }
}

fn contract_transfer(t: &[f64], kt: usize, kl: usize, kr: usize, left: &[f64], right: &[f64]) -> Vec<f64> {
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

fn contract_leaf_weights(node: &HtNode, weights: &[f64]) -> Vec<f64> {
    let (frame, _) = leaf_data(node);
    let (n, k) = (frame.nrows(), frame.ncols());
    assert_eq!(n, weights.len());
    let mut r = vec![0.0f64; k];
    for j in 0..k {
        for i in 0..n {
            r[j] += frame[(i, j)] * weights[i];
        }
    }
    r
}

fn ext_f64(s: &super::super::init::domain::SpatialDom) -> [f64; 3] {
    [s.x1.to_f64().unwrap(), s.x2.to_f64().unwrap(), s.x3.to_f64().unwrap()]
}

fn ext_f64_v(v: &super::super::init::domain::VelocityDom) -> [f64; 3] {
    [v.v1.to_f64().unwrap(), v.v2.to_f64().unwrap(), v.v3.to_f64().unwrap()]
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

fn thin_svd(mat: &Mat<f64>) -> (Mat<f64>, Vec<f64>, Mat<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();
    let k = m.min(n);
    if k == 0 {
        return (Mat::zeros(m, 0), vec![], Mat::zeros(0, n));
    }
    let svd = mat.as_ref().thin_svd().expect("SVD failed");
    let u = svd.U().to_owned();
    let vt = svd.V().transpose().to_owned();
    let s_diag = svd.S().column_vector();
    let s: Vec<f64> = (0..k).map(|i| s_diag[i]).collect();
    (u, s, vt)
}

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

fn qr_decompose(mat: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();
    if m.min(n) == 0 {
        return (Mat::zeros(m, 0), Mat::zeros(0, n));
    }
    let qr = mat.as_ref().qr();
    (qr.compute_thin_Q(), qr.thin_R().to_owned())
}

fn s_times_vt(s: &[f64], vt: &Mat<f64>, rank: usize) -> Mat<f64> {
    let n = vt.ncols();
    let mut r = Mat::zeros(rank, n);
    for i in 0..rank {
        for j in 0..n {
            r[(i, j)] = s[i] * vt[(i, j)];
        }
    }
    r
}

fn vec_to_mat(data: &[f64], rows: usize, cols: usize) -> Mat<f64> {
    let mut m = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m[(i, j)] = data[i * cols + j];
        }
    }
    m
}

fn mat_to_vec(m: &Mat<f64>, rows: usize, cols: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            v[i * cols + j] = m[(i, j)];
        }
    }
    v
}

fn matmul_at_b(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.ncols();
    let n = b.ncols();
    let p = a.nrows();
    assert_eq!(p, b.nrows());
    let mut c = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..p { s += a[(k, i)] * b[(k, j)]; }
            c[(i, j)] = s;
        }
    }
    c
}

/// Compute transfer tensor for an interior node.
///
/// For non-root: B_t[i,j,k] = Σ_{S_t indices} U_t[parent_idx, i] · U_{t1}[left_idx, j] · U_{t2}[right_idx, k]
/// (pure frame projection, no data needed).
///
/// For root: B_root[0,j,k] = Σ_{all indices} data[flat] · U_left[left_idx, j] · U_right[right_idx, k]
/// (projects data onto child frames).
fn compute_transfer_tensor(
    data: &[f64], shape: &[usize; 6],
    parent_dims: &[usize], left_dims: &[usize], right_dims: &[usize],
    parent_frame: &Mat<f64>, left_frame: &Mat<f64>, right_frame: &Mat<f64>,
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
        for d in (0..5).rev() { strides[d] = strides[d + 1] * shape[d + 1]; }

        for flat in 0..total {
            let val = data[flat];
            if val.abs() < 1e-30 { continue; }

            let mut rem = flat;
            let mut indices = [0usize; 6];
            for d in 0..6 { indices[d] = rem / strides[d]; rem %= strides[d]; }

            let mut left_idx = 0;
            let mut ls = 1;
            for &d in left_dims.iter().rev() { left_idx += indices[d] * ls; ls *= shape[d]; }

            let mut right_idx = 0;
            let mut rs = 1;
            for &d in right_dims.iter().rev() { right_idx += indices[d] * rs; rs *= shape[d]; }

            for j in 0..kl {
                let ul = left_frame[(left_idx, j)];
                if ul.abs() < 1e-30 { continue; }
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
                let pos = parent_dims.iter().position(|&pd| pd == d).unwrap();
                left_idx += dim_indices[pos] * ls;
                ls *= shape[d];
            }

            // Compute right_idx from right_dims
            let mut right_idx = 0;
            let mut rs = 1;
            for &d in right_dims.iter().rev() {
                let pos = parent_dims.iter().position(|&pd| pd == d).unwrap();
                right_idx += dim_indices[pos] * rs;
                rs *= shape[d];
            }

            for i in 0..kt {
                let up = parent_frame[(parent_idx, i)];
                if up.abs() < 1e-30 { continue; }
                for j in 0..kl {
                    let ul = left_frame[(left_idx, j)];
                    if ul.abs() < 1e-30 { continue; }
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
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let idx = flat_index(&shape, [i0, i1, i2, i3, i4, i5]);
                data[idx] = ((i0+1)*(i1+1)*(i2+1)*(i3+1)*(i4+1)*(i5+1)) as f64;
            }}}}}}

        let ht = HtTensor::from_full(&data, shape, &domain, 1e-12);

        let mut max_err = 0.0f64;
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let expected = ((i0+1)*(i1+1)*(i2+1)*(i3+1)*(i4+1)*(i5+1)) as f64;
                let got = ht.evaluate([i0, i1, i2, i3, i4, i5]);
                max_err = max_err.max((got - expected).abs());
            }}}}}}

        assert!(max_err < 1e-8, "rank-1 round-trip max error {max_err}");

        for d in 0..6 {
            assert!(ht.rank_at(d) <= 2, "leaf {d} rank {} too large", ht.rank_at(d));
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
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let x: Vec<f64> = [i0,i1,i2,i3,i4,i5].iter()
                    .map(|&i| (i as f64 + 0.5) / n as f64 - 0.5).collect();
                let r2: f64 = x.iter().map(|xi| xi*xi).sum();
                data[flat_index(&shape, [i0,i1,i2,i3,i4,i5])] = (-r2/(2.0*sigma*sigma)).exp();
            }}}}}}

        let tol = 1e-4;
        let ht = HtTensor::from_full(&data, shape, &domain, tol);

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let e = data[flat_index(&shape, [i0,i1,i2,i3,i4,i5])];
                let g = ht.evaluate([i0,i1,i2,i3,i4,i5]);
                err_sq += (g-e)*(g-e);
                norm_sq += e*e;
            }}}}}}

        let rel_err = (err_sq / norm_sq).sqrt();
        assert!(rel_err < tol * 10.0, "Gaussian rel error {rel_err:.2e}");
        println!("Gaussian: full={}B, HT={}B, ratio={:.1}x",
            total*8, ht.memory_bytes(), (total*8) as f64 / ht.memory_bytes() as f64);
    }

    #[test]
    fn addition() {
        let n = 4usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);

        let mut data_a = vec![0.0f64; total];
        let mut data_b = vec![0.0f64; total];
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let idx = flat_index(&shape, [i0,i1,i2,i3,i4,i5]);
                data_a[idx] = (i0+1) as f64;
                data_b[idx] = (i3+1) as f64;
            }}}}}}

        let a = HtTensor::from_full(&data_a, shape, &domain, 1e-12);
        let b = HtTensor::from_full(&data_b, shape, &domain, 1e-12);
        let s = a.add(&b);

        let mut max_err = 0.0f64;
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let expected = (i0+1) as f64 + (i3+1) as f64;
                let got = s.evaluate([i0,i1,i2,i3,i4,i5]);
                max_err = max_err.max((got - expected).abs());
            }}}}}}

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
        let lx = ext_f64(&domain.spatial);
        let lv = ext_f64_v(&domain.velocity);

        let mut data = vec![0.0f64; total];
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let x0 = -lx[0] + (i0 as f64+0.5)*dx[0];
                let x1 = -lx[1] + (i1 as f64+0.5)*dx[1];
                let x2 = -lx[2] + (i2 as f64+0.5)*dx[2];
                let v0 = -lv[0] + (i3 as f64+0.5)*dv[0];
                let v1 = -lv[1] + (i4 as f64+0.5)*dv[1];
                let v2 = -lv[2] + (i5 as f64+0.5)*dv[2];
                let gx = (-(x0*x0+x1*x1+x2*x2)/(2.0*sigma*sigma)).exp();
                let hv = (-(v0*v0+v1*v1+v2*v2)/(2.0*sigma*sigma)).exp();
                data[flat_index(&shape, [i0,i1,i2,i3,i4,i5])] = gx * hv;
            }}}}}}

        let ht = HtTensor::from_full(&data, shape, &domain, 1e-10);
        let density = ht.compute_density();

        let dv3 = dv[0]*dv[1]*dv[2];
        let mut max_err = 0.0f64;
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            let mut rho_direct = 0.0;
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                rho_direct += data[flat_index(&shape, [i0,i1,i2,i3,i4,i5])];
            }}}
            rho_direct *= dv3;
            let rho_ht = density.data[i0*n*n + i1*n + i2];
            max_err = max_err.max((rho_ht - rho_direct).abs());
        }}}

        assert!(max_err < 1e-6, "density max error {max_err:.2e}");
    }

    #[test]
    fn inner_product_and_norm() {
        let n = 4usize;
        let domain = test_domain(n as i128);
        let shape = [n; 6];
        let total = n.pow(6);

        let mut data = vec![0.0f64; total];
        for i in 0..total { data[i] = (i as f64) / (total as f64); }

        let ht = HtTensor::from_full(&data, shape, &domain, 1e-12);
        let direct_norm_sq: f64 = data.iter().map(|x| x*x).sum();
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
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let x: Vec<f64> = [i0,i1,i2,i3,i4,i5].iter()
                    .map(|&i| (i as f64 + 0.5) / n as f64 - 0.5).collect();
                let r2: f64 = x.iter().map(|xi| xi*xi).sum();
                data[flat_index(&shape, [i0,i1,i2,i3,i4,i5])] = (-r2/(2.0*sigma*sigma)).exp();
            }}}}}}

        let orig = HtTensor::from_full(&data, shape, &domain, 1e-12);
        let orig_rank = orig.total_rank();

        let mut trunc = orig.clone();
        trunc.truncate(1e-2);
        let trunc_rank = trunc.total_rank();

        assert!(trunc_rank <= orig_rank, "truncation should reduce rank: {trunc_rank} vs {orig_rank}");

        let mut err_sq = 0.0;
        let mut norm_sq = 0.0;
        for i0 in 0..n { for i1 in 0..n { for i2 in 0..n {
            for i3 in 0..n { for i4 in 0..n { for i5 in 0..n {
                let e = data[flat_index(&shape, [i0,i1,i2,i3,i4,i5])];
                let g = trunc.evaluate([i0,i1,i2,i3,i4,i5]);
                err_sq += (g-e)*(g-e);
                norm_sq += e*e;
            }}}}}}

        let rel_err = (err_sq / norm_sq).sqrt();
        assert!(rel_err < 0.1, "truncation rel error {rel_err:.2e}");
    }
}
