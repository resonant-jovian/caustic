#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! 3D Hierarchical Tucker tensor for spatial fields.
//!
//! The Poisson solver operates on 3D scalar fields (density, potential). This
//! module provides a dedicated 3D HT tensor with dimension tree:
//!
//! ```text
//!         4 (root)
//!        /      \
//!      0        3
//!    [x1]    {x2,x3}
//!             /    \
//!            1      2
//!          [x2]   [x3]
//! ```
//!
//! 5 nodes: 3 leaves (0,1,2) + 1 interior (3) + 1 root (4).
//! Memory: O(3nk² + 2k³) where n = grid size per dimension, k = max rank.
//!
//! Also provides a complex variant `HtTensor3DComplex` for Fourier-space
//! operations, where leaf frames are complex but transfer tensors remain real.

use super::aca::{BlackBoxMatrix, FnMatrix, Xorshift64, aca_partial_pivot};
use faer::Mat;
use rayon::prelude::*;

// ─── Fixed 3D dimension tree topology ────────────────────────────────────────
//   0: leaf [x1]
//   1: leaf [x2]
//   2: leaf [x3]
//   3: interior {x2,x3}   children: 1, 2
//   4: root {x1,x2,x3}    children: 0, 3

const NUM_NODES_3D: usize = 5;
const NUM_LEAVES_3D: usize = 3;
const ROOT_3D: usize = 4;

/// A node in the 3D HT dimension tree. Same enum variants as 6D HtNode.
#[derive(Clone)]
pub enum HtNode3D {
    /// Leaf node: basis matrix U ∈ R^{n x k} for one spatial dimension.
    Leaf {
        /// Dimension index (0 = x1, 1 = x2, 2 = x3).
        dim: usize,
        /// Orthonormal basis matrix, shape (n, k) where n = grid points, k = rank.
        frame: Mat<f64>,
    },
    /// Interior node: transfer tensor B ∈ R^{k_t x k_left x k_right} stored flat in row-major order.
    Interior {
        /// Index of the left child node in the nodes array.
        left: usize,
        /// Index of the right child node in the nodes array.
        right: usize,
        /// Flattened transfer tensor of size k_t * k_left * k_right.
        transfer: Vec<f64>,
        /// Rank triple [k_t, k_left, k_right].
        ranks: [usize; 3],
    },
}

impl HtNode3D {
    /// Return the rank of this node (number of columns for leaves, parent rank for interiors).
    #[inline]
    pub fn rank(&self) -> usize {
        match self {
            HtNode3D::Leaf { frame, .. } => frame.ncols(),
            HtNode3D::Interior { ranks, .. } => ranks[0],
        }
    }
}

/// 3D Hierarchical Tucker tensor for spatial scalar fields (density, potential).
///
/// Stores a 3D tensor in compressed HT format with 5 nodes: 3 leaves (one per
/// spatial dimension) and 2 interior/root nodes. Memory scales as O(n*k^2 + k^3)
/// instead of O(n^3) for the dense representation.
#[derive(Clone)]
pub struct HtTensor3D {
    /// All 5 nodes in the dimension tree (indices 0-2: leaves, 3: interior, 4: root).
    pub nodes: Vec<HtNode3D>,
    /// Grid dimensions [n_x1, n_x2, n_x3].
    pub shape: [usize; 3],
    /// Cell spacings [dx1, dx2, dx3].
    pub dx: [f64; 3],
}

impl HtTensor3D {
    /// Evaluate the tensor at a 3D index.
    pub fn evaluate(&self, idx: [usize; 3]) -> f64 {
        // Bottom-up contraction:
        // 1. Extract leaf vectors: u_d = U_d[i_d, :]  (length k_d)
        // 2. Contract interior node 3: z_3[t] = Σ_{l,r} B_3[t,l,r] * u_1[l] * u_2[r]
        // 3. Contract root node 4:     z_4[t] = Σ_{l,r} B_4[t,l,r] * u_0[l] * z_3[r]
        // Return z_4[0] (root rank = 1 for scalars)

        let u0 = self.leaf_vector(0, idx[0]);
        let u1 = self.leaf_vector(1, idx[1]);
        let u2 = self.leaf_vector(2, idx[2]);

        // Node 3: contract u1 and u2
        let z3 = self.contract_interior(3, &u1, &u2);

        // Node 4 (root): contract u0 and z3
        let z4 = self.contract_interior(4, &u0, &z3);

        z4[0]
    }

    /// Extract column vector from leaf frame at given index.
    #[inline]
    fn leaf_vector(&self, node: usize, idx: usize) -> Vec<f64> {
        match &self.nodes[node] {
            HtNode3D::Leaf { frame, .. } => {
                let k = frame.ncols();
                (0..k).map(|j| frame[(idx, j)]).collect()
            }
            _ => {
                debug_assert!(false, "Node {node} is not a leaf");
                vec![]
            }
        }
    }

    /// Contract an interior node's transfer tensor with left and right vectors.
    #[inline]
    fn contract_interior(&self, node: usize, left: &[f64], right: &[f64]) -> Vec<f64> {
        match &self.nodes[node] {
            HtNode3D::Interior {
                transfer, ranks, ..
            } => {
                let [kt, kl, kr] = *ranks;
                let mut result = vec![0.0; kt];
                for t in 0..kt {
                    let mut sum = 0.0;
                    for l in 0..kl {
                        for r in 0..kr {
                            sum += transfer[t * kl * kr + l * kr + r] * left[l] * right[r];
                        }
                    }
                    result[t] = sum;
                }
                result
            }
            _ => {
                debug_assert!(false, "Node {node} is not an interior node");
                vec![]
            }
        }
    }

    /// Construct HT3D from a dense 3D array via HSVD (Hierarchical SVD).
    pub fn from_dense(
        data: &[f64],
        shape: [usize; 3],
        dx: [f64; 3],
        tolerance: f64,
        max_rank: usize,
    ) -> Self {
        let [n0, n1, n2] = shape;
        assert_eq!(data.len(), n0 * n1 * n2);

        let eps_node = tolerance / (2.0_f64).sqrt(); // quasi-optimal per-node tolerance

        // Leaf 0 (x1): mode-0 unfolding, shape (n0, n1*n2)
        let (u0, frame0) = mode_unfolding_svd(data, n0, n1 * n2, eps_node, max_rank);

        // Leaf 1 (x2): mode-1 unfolding, shape (n1, n0*n2)
        let mut mode1 = vec![0.0; n1 * n0 * n2];
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    mode1[i1 * (n0 * n2) + i0 * n2 + i2] = data[i0 * n1 * n2 + i1 * n2 + i2];
                }
            }
        }
        let (u1, frame1) = mode_unfolding_svd(&mode1, n1, n0 * n2, eps_node, max_rank);

        // Leaf 2 (x3): mode-2 unfolding, shape (n2, n0*n1)
        let mut mode2 = vec![0.0; n2 * n0 * n1];
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    mode2[i2 * (n0 * n1) + i0 * n1 + i1] = data[i0 * n1 * n2 + i1 * n2 + i2];
                }
            }
        }
        let (u2, frame2) = mode_unfolding_svd(&mode2, n2, n0 * n1, eps_node, max_rank);

        let k0 = frame0.ncols();
        let k1 = frame1.ncols();
        let k2 = frame2.ncols();

        // Interior node 3: {x2, x3}
        // Kron frame: U_{12} = U_1 ⊗ U_2, shape (n1*n2, k1*k2)
        // Project data: R = U_{12}^T @ matricize_{12}(data)
        // SVD of R gives transfer tensor
        let k12 = k1 * k2;
        // Matricize data as (n1*n2, n0)
        let mut mat12 = vec![0.0; n1 * n2 * n0];
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    mat12[(i1 * n2 + i2) * n0 + i0] = data[i0 * n1 * n2 + i1 * n2 + i2];
                }
            }
        }

        // Projected matrix: K^T @ mat12, shape (k1*k2, n0)
        let mut projected = vec![0.0; k12 * n0];
        for l1 in 0..k1 {
            for l2 in 0..k2 {
                let row = l1 * k2 + l2;
                for i0 in 0..n0 {
                    let mut sum = 0.0;
                    for i1 in 0..n1 {
                        for i2 in 0..n2 {
                            sum += frame1[(i1, l1)]
                                * frame2[(i2, l2)]
                                * mat12[(i1 * n2 + i2) * n0 + i0];
                        }
                    }
                    projected[row * n0 + i0] = sum;
                }
            }
        }

        // SVD of projected (k12 × n0) to get transfer for node 3
        let mut proj_mat: Mat<f64> = Mat::zeros(k12, n0);
        for r in 0..k12 {
            for c in 0..n0 {
                proj_mat[(r, c)] = projected[r * n0 + c];
            }
        }
        let svd3 = proj_mat.as_ref().thin_svd();
        let (u3, sv3, _vt3) = match svd3 {
            Ok(svd) => {
                let s_col = svd.S().column_vector();
                let sv: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
                (svd.U().to_owned(), sv, svd.V().to_owned())
            }
            Err(_) => {
                return Self::zero(shape, dx);
            }
        };

        let k3 = truncation_rank(&sv3, eps_node)
            .min(max_rank)
            .min(u3.ncols())
            .max(1);

        // Transfer tensor for node 3: B_3[t, l1, l2] = U3[l1*k2+l2, t]
        let mut transfer3 = vec![0.0; k3 * k1 * k2];
        for t in 0..k3 {
            for l1 in 0..k1 {
                for l2 in 0..k2 {
                    transfer3[t * k1 * k2 + l1 * k2 + l2] = u3[(l1 * k2 + l2, t)];
                }
            }
        }

        // Root node 4: {x1, x2, x3}
        // Left child is leaf 0 (rank k0), right child is node 3 (rank k3)
        // Root transfer: B_4[t, l, r] with t=1 (scalar field)
        //
        // Project data onto frames: for each (l0, t3), compute
        //   R[l0, t3] = Σ_{i0,i1,i2} U0[i0,l0] * (Σ_{l1,l2} B3[t3,l1,l2] U1[i1,l1] U2[i2,l2]) * data[i0,i1,i2]
        //
        // This is equivalent to: R = U0^T @ data_matricized @ (B3 · (U1⊗U2))^T
        // But simpler: R[l0, t3] = Σ_{l1,l2} B3[t3,l1,l2] * (Σ_i0 U0[i0,l0] * Σ_{i1,i2} U1[i1,l1]*U2[i2,l2]*data[i0,i1,i2])

        // First compute M[l0, l1, l2] = Σ_{i0,i1,i2} U0[i0,l0] * U1[i1,l1] * U2[i2,l2] * data[i0,i1,i2]
        let mut m_tensor = vec![0.0; k0 * k1 * k2];
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    let val = data[i0 * n1 * n2 + i1 * n2 + i2];
                    if val.abs() < 1e-30 {
                        continue;
                    }
                    for l0 in 0..k0 {
                        let u0_val = frame0[(i0, l0)];
                        for l1 in 0..k1 {
                            let u01_val = u0_val * frame1[(i1, l1)];
                            for l2 in 0..k2 {
                                m_tensor[l0 * k1 * k2 + l1 * k2 + l2] +=
                                    u01_val * frame2[(i2, l2)] * val;
                            }
                        }
                    }
                }
            }
        }

        // Now R[l0, t3] = Σ_{l1,l2} B3[t3,l1,l2] * M[l0,l1,l2]
        let mut root_mat = vec![0.0; k0 * k3];
        for l0 in 0..k0 {
            for t3 in 0..k3 {
                let mut sum = 0.0;
                for l1 in 0..k1 {
                    for l2 in 0..k2 {
                        sum += transfer3[t3 * k1 * k2 + l1 * k2 + l2]
                            * m_tensor[l0 * k1 * k2 + l1 * k2 + l2];
                    }
                }
                root_mat[l0 * k3 + t3] = sum;
            }
        }

        // Root rank is always 1 for a scalar tensor.
        // B_root[0, l0, t3] = R_root[l0, t3] directly (no SVD needed at root).
        let k_root = 1;
        let mut transfer_root = vec![0.0; k0 * k3];
        for l0 in 0..k0 {
            for t3 in 0..k3 {
                transfer_root[l0 * k3 + t3] = root_mat[l0 * k3 + t3];
            }
        }

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        // Leaf 0
        nodes.push(HtNode3D::Leaf {
            dim: 0,
            frame: frame0,
        });
        // Leaf 1
        nodes.push(HtNode3D::Leaf {
            dim: 1,
            frame: frame1,
        });
        // Leaf 2
        nodes.push(HtNode3D::Leaf {
            dim: 2,
            frame: frame2,
        });
        // Interior node 3
        nodes.push(HtNode3D::Interior {
            left: 1,
            right: 2,
            transfer: transfer3,
            ranks: [k3, k1, k2],
        });
        // Root node 4
        nodes.push(HtNode3D::Interior {
            left: 0,
            right: 3,
            transfer: transfer_root,
            ranks: [k_root, k0, k3],
        });

        Self { nodes, shape, dx }
    }

    /// Construct an HT3D tensor from a `DensityField` via HSVD.
    pub fn from_density(
        density: &super::super::types::DensityField,
        dx: [f64; 3],
        tolerance: f64,
        max_rank: usize,
    ) -> Self {
        Self::from_dense(&density.data, density.shape, dx, tolerance, max_rank)
    }

    /// Build an HT3D by sampling a function f(i0, i1, i2) -> f64 via fiber-based HTACA.
    ///
    /// Avoids materializing the full dense array. Leaf frames are built from random
    /// fiber samples with column-pivoted QR, then transfer tensors are computed
    /// by projecting onto the leaf bases.
    pub fn from_function_aca<F: Fn([usize; 3]) -> f64 + Sync>(
        f: &F,
        shape: [usize; 3],
        dx: [f64; 3],
        tolerance: f64,
        max_rank: usize,
    ) -> Self {
        let [n0, n1, n2] = shape;

        let eps_node = tolerance / (2.0_f64).sqrt();

        // Phase A: leaf frames via fiber sampling + column-pivoted QR
        let n_samples = (8 * max_rank).min(n0 * n1).max(max_rank);
        let mut rng = Xorshift64::new(42);

        // Leaf 0: fibers along x1 dimension
        let mut fiber_mat0: Mat<f64> = Mat::zeros(n0, n_samples);
        for s in 0..n_samples {
            let i1 = rng.next_usize(n1);
            let i2 = rng.next_usize(n2);
            for i0 in 0..n0 {
                fiber_mat0[(i0, s)] = f([i0, i1, i2]);
            }
        }
        let frame0 = extract_frame_qr(&fiber_mat0, max_rank);

        // Leaf 1: fibers along x2 dimension
        let mut fiber_mat1: Mat<f64> = Mat::zeros(n1, n_samples);
        for s in 0..n_samples {
            let i0 = rng.next_usize(n0);
            let i2 = rng.next_usize(n2);
            for i1 in 0..n1 {
                fiber_mat1[(i1, s)] = f([i0, i1, i2]);
            }
        }
        let frame1 = extract_frame_qr(&fiber_mat1, max_rank);

        // Leaf 2: fibers along x3 dimension
        let mut fiber_mat2: Mat<f64> = Mat::zeros(n2, n_samples);
        for s in 0..n_samples {
            let i0 = rng.next_usize(n0);
            let i1 = rng.next_usize(n1);
            for i2 in 0..n2 {
                fiber_mat2[(i2, s)] = f([i0, i1, i2]);
            }
        }
        let frame2 = extract_frame_qr(&fiber_mat2, max_rank);

        let k0 = frame0.ncols();
        let k1 = frame1.ncols();
        let k2 = frame2.ncols();

        // Phase B: interior node 3 transfer tensor
        // Project f onto leaf bases for dims 1,2
        // R[l1*k2+l2, i0] = Σ_{i1,i2} U1[i1,l1] * U2[i2,l2] * f(i0, i1, i2)
        let k12 = k1 * k2;
        let projected3_flat: Vec<f64> = (0..n0)
            .into_par_iter()
            .flat_map(|i0| {
                let mut col = vec![0.0; k12];
                for i1 in 0..n1 {
                    for i2 in 0..n2 {
                        let val = f([i0, i1, i2]);
                        if val.abs() < 1e-30 {
                            continue;
                        }
                        for l1 in 0..k1 {
                            for l2 in 0..k2 {
                                col[l1 * k2 + l2] += frame1[(i1, l1)] * frame2[(i2, l2)] * val;
                            }
                        }
                    }
                }
                col
            })
            .collect();

        let mut projected3: Mat<f64> = Mat::zeros(k12, n0);
        for i0 in 0..n0 {
            for row in 0..k12 {
                projected3[(row, i0)] = projected3_flat[i0 * k12 + row];
            }
        }

        // SVD of projected3 to get transfer tensor for node 3
        let svd3 = projected3.as_ref().thin_svd();
        let (u3_mat, sv3, _vt3) = match svd3 {
            Ok(svd) => {
                let s_col = svd.S().column_vector();
                let sv: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
                (svd.U().to_owned(), sv, svd.V().to_owned())
            }
            Err(_) => {
                return Self::zero(shape, dx);
            }
        };

        let k3 = truncation_rank(&sv3, eps_node)
            .min(max_rank)
            .min(u3_mat.ncols())
            .max(1);

        let mut transfer3 = vec![0.0; k3 * k1 * k2];
        for t in 0..k3 {
            for l1 in 0..k1 {
                for l2 in 0..k2 {
                    transfer3[t * k1 * k2 + l1 * k2 + l2] = u3_mat[(l1 * k2 + l2, t)];
                }
            }
        }

        // Phase C: root transfer tensor
        // M[l0, l1, l2] = Σ_{i0,i1,i2} U0[i0,l0] * U1[i1,l1] * U2[i2,l2] * f(i0,i1,i2)
        let kk = k0 * k1 * k2;
        let m_tensor: Vec<f64> = (0..n0)
            .into_par_iter()
            .fold(
                || vec![0.0; kk],
                |mut acc, i0| {
                    for i1 in 0..n1 {
                        for i2 in 0..n2 {
                            let val = f([i0, i1, i2]);
                            if val.abs() < 1e-30 {
                                continue;
                            }
                            for l0 in 0..k0 {
                                let u0v = frame0[(i0, l0)];
                                for l1 in 0..k1 {
                                    let u01v = u0v * frame1[(i1, l1)];
                                    for l2 in 0..k2 {
                                        acc[l0 * k1 * k2 + l1 * k2 + l2] +=
                                            u01v * frame2[(i2, l2)] * val;
                                    }
                                }
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; kk],
                |mut a, b| {
                    for i in 0..kk {
                        a[i] += b[i];
                    }
                    a
                },
            );

        // R_root[l0, t3] = Σ_{l1,l2} B3[t3,l1,l2] * M[l0,l1,l2]
        let mut root_mat: Mat<f64> = Mat::zeros(k0, k3);
        for l0 in 0..k0 {
            for t3 in 0..k3 {
                let mut sum = 0.0;
                for l1 in 0..k1 {
                    for l2 in 0..k2 {
                        sum += transfer3[t3 * k1 * k2 + l1 * k2 + l2]
                            * m_tensor[l0 * k1 * k2 + l1 * k2 + l2];
                    }
                }
                root_mat[(l0, t3)] = sum;
            }
        }

        // Root rank is always 1 for a scalar tensor.
        // B_root[0, l0, t3] = R_root[l0, t3] directly.
        let k_root = 1;
        let mut transfer_root = vec![0.0; k0 * k3];
        for l0 in 0..k0 {
            for t3 in 0..k3 {
                transfer_root[l0 * k3 + t3] = root_mat[(l0, t3)];
            }
        }

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);
        nodes.push(HtNode3D::Leaf {
            dim: 0,
            frame: frame0,
        });
        nodes.push(HtNode3D::Leaf {
            dim: 1,
            frame: frame1,
        });
        nodes.push(HtNode3D::Leaf {
            dim: 2,
            frame: frame2,
        });
        nodes.push(HtNode3D::Interior {
            left: 1,
            right: 2,
            transfer: transfer3,
            ranks: [k3, k1, k2],
        });
        nodes.push(HtNode3D::Interior {
            left: 0,
            right: 3,
            transfer: transfer_root,
            ranks: [k_root, k0, k3],
        });

        Self { nodes, shape, dx }
    }

    /// Zero-pad leaf frames from N to 2N rows (for Hockney isolated-BC convolution).
    /// Transfer tensors are unchanged; only leaf frames grow.
    pub fn zero_pad(&self) -> HtTensor3D {
        let new_shape = [self.shape[0] * 2, self.shape[1] * 2, self.shape[2] * 2];
        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                HtNode3D::Leaf { dim, frame } => {
                    let n = frame.nrows();
                    let k = frame.ncols();
                    let new_n = n * 2;
                    let mut new_frame = Mat::zeros(new_n, k);
                    for r in 0..n {
                        for c in 0..k {
                            new_frame[(r, c)] = frame[(r, c)];
                        }
                    }
                    nodes.push(HtNode3D::Leaf {
                        dim: *dim,
                        frame: new_frame,
                    });
                }
                HtNode3D::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3D::Interior {
                        left: *left,
                        right: *right,
                        transfer: transfer.clone(),
                        ranks: *ranks,
                    });
                }
            }
        }

        HtTensor3D {
            nodes,
            shape: new_shape,
            dx: self.dx,
        }
    }

    /// Consuming variant of [`zero_pad`](Self::zero_pad) that moves transfer tensors
    /// instead of cloning them.
    pub fn into_zero_padded(self) -> HtTensor3D {
        let new_shape = [self.shape[0] * 2, self.shape[1] * 2, self.shape[2] * 2];
        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in self.nodes {
            match node {
                HtNode3D::Leaf { dim, frame } => {
                    let n = frame.nrows();
                    let k = frame.ncols();
                    let new_n = n * 2;
                    let mut new_frame = Mat::zeros(new_n, k);
                    for r in 0..n {
                        for c in 0..k {
                            new_frame[(r, c)] = frame[(r, c)];
                        }
                    }
                    nodes.push(HtNode3D::Leaf {
                        dim,
                        frame: new_frame,
                    });
                }
                HtNode3D::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3D::Interior {
                        left,
                        right,
                        transfer,
                        ranks,
                    });
                }
            }
        }

        HtTensor3D {
            nodes,
            shape: new_shape,
            dx: self.dx,
        }
    }

    /// Extract the first N entries from each leaf frame, undoing zero-padding.
    /// Returns a new HT3D with the given `shape`.
    pub fn extract_subgrid(&self, shape: [usize; 3]) -> HtTensor3D {
        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in &self.nodes {
            match node {
                HtNode3D::Leaf { dim, frame } => {
                    let target_n = shape[*dim];
                    let k = frame.ncols();
                    let n = target_n.min(frame.nrows());
                    let mut new_frame = Mat::zeros(n, k);
                    for r in 0..n {
                        for c in 0..k {
                            new_frame[(r, c)] = frame[(r, c)];
                        }
                    }
                    nodes.push(HtNode3D::Leaf {
                        dim: *dim,
                        frame: new_frame,
                    });
                }
                HtNode3D::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3D::Interior {
                        left: *left,
                        right: *right,
                        transfer: transfer.clone(),
                        ranks: *ranks,
                    });
                }
            }
        }

        HtTensor3D {
            nodes,
            shape,
            dx: self.dx,
        }
    }

    /// Consuming variant of [`extract_subgrid`](Self::extract_subgrid) that moves
    /// transfer tensors instead of cloning them.
    pub fn into_subgrid(self, shape: [usize; 3]) -> HtTensor3D {
        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in self.nodes {
            match node {
                HtNode3D::Leaf { dim, frame } => {
                    let target_n = shape[dim];
                    let k = frame.ncols();
                    let n = target_n.min(frame.nrows());
                    let mut new_frame = Mat::zeros(n, k);
                    for r in 0..n {
                        for c in 0..k {
                            new_frame[(r, c)] = frame[(r, c)];
                        }
                    }
                    nodes.push(HtNode3D::Leaf {
                        dim,
                        frame: new_frame,
                    });
                }
                HtNode3D::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3D::Interior {
                        left,
                        right,
                        transfer,
                        ranks,
                    });
                }
            }
        }

        HtTensor3D {
            nodes,
            shape,
            dx: self.dx,
        }
    }

    /// Expand to a dense N^3 array by full tensor reconstruction.
    pub fn to_full_3d(&self) -> Vec<f64> {
        self.to_dense_subgrid(self.shape)
    }

    /// Batch extract a sub-grid [s0, s1, s2] via Khatri-Rao product reconstruction.
    ///
    /// Uses only rows `[0..s_d]` of each leaf frame, avoiding computation in the
    /// zero-padded region. Much faster than per-point `evaluate()`:
    /// O(n₁·n₂·k₁·k₂·k₃ + n₀·n₁·n₂·k₀) vs O(n₀·n₁·n₂·k₀·k₁·k₂·k₃).
    pub fn to_dense_subgrid(&self, sub: [usize; 3]) -> Vec<f64> {
        let [s0, s1, s2] = sub;

        // Extract leaf frames and interior transfer tensors
        let (u0, u1, u2) = self.leaf_frames();
        let (b3, rk3) = self.interior_transfer(3);
        let (b4, rk4) = self.interior_transfer(4);
        let [_kt3, k1, k2] = rk3;
        let [_kt4, k0, k3] = rk4;

        // Step 1: Contract B₃ with U₂: T₁[j₃, j₁, i₂] = Σ_{j₂} B₃[j₃, j₁, j₂] · U₂[i₂, j₂]
        // T₁ shape: k3 × k1 × s2
        let mut t1 = vec![0.0f64; k3 * k1 * s2];
        for j3 in 0..k3 {
            for j1 in 0..k1 {
                for i2 in 0..s2 {
                    let mut sum = 0.0;
                    for j2 in 0..k2 {
                        sum += b3[j3 * k1 * k2 + j1 * k2 + j2] * u2[(i2, j2)];
                    }
                    t1[j3 * k1 * s2 + j1 * s2 + i2] = sum;
                }
            }
        }

        // Step 2: Contract T₁ with U₁: T₂[j₃, i₁, i₂] = Σ_{j₁} U₁[i₁, j₁] · T₁[j₃, j₁, i₂]
        // T₂ shape: k3 × s1 × s2
        let mut t2 = vec![0.0f64; k3 * s1 * s2];
        for j3 in 0..k3 {
            for i1 in 0..s1 {
                for i2 in 0..s2 {
                    let mut sum = 0.0;
                    for j1 in 0..k1 {
                        sum += u1[(i1, j1)] * t1[j3 * k1 * s2 + j1 * s2 + i2];
                    }
                    t2[j3 * s1 * s2 + i1 * s2 + i2] = sum;
                }
            }
        }

        // Step 3: Contract B₄ with T₂: T₃[j₀, i₁, i₂] = Σ_{j₃} B₄[0, j₀, j₃] · T₂[j₃, i₁, i₂]
        // T₃ shape: k0 × s1 × s2
        let mut t3 = vec![0.0f64; k0 * s1 * s2];
        for j0 in 0..k0 {
            for i1 in 0..s1 {
                for i2 in 0..s2 {
                    let mut sum = 0.0;
                    for j3 in 0..k3 {
                        sum += b4[j0 * k3 + j3] * t2[j3 * s1 * s2 + i1 * s2 + i2];
                    }
                    t3[j0 * s1 * s2 + i1 * s2 + i2] = sum;
                }
            }
        }

        // Step 4: Contract with U₀: data[i₀, i₁, i₂] = Σ_{j₀} U₀[i₀, j₀] · T₃[j₀, i₁, i₂]
        // Parallelize over i₀ slabs
        let slab_size = s1 * s2;
        let mut data = vec![0.0f64; s0 * slab_size];
        data.par_chunks_mut(slab_size)
            .enumerate()
            .for_each(|(i0, chunk)| {
                for i1 in 0..s1 {
                    for i2 in 0..s2 {
                        let mut sum = 0.0;
                        for j0 in 0..k0 {
                            sum += u0[(i0, j0)] * t3[j0 * slab_size + i1 * s2 + i2];
                        }
                        chunk[i1 * s2 + i2] = sum;
                    }
                }
            });

        data
    }

    /// Extract leaf frames as references to faer Mat.
    fn leaf_frames(&self) -> (&Mat<f64>, &Mat<f64>, &Mat<f64>) {
        let f0 = match &self.nodes[0] {
            HtNode3D::Leaf { frame, .. } => frame,
            _ => {
                debug_assert!(false, "Node 0 is not a leaf");
                // Return the first available leaf frame as fallback (unreachable path)
                match &self.nodes[0] {
                    HtNode3D::Leaf { frame, .. } => frame,
                    HtNode3D::Interior { .. } => {
                        static EMPTY: std::sync::LazyLock<Mat<f64>> =
                            std::sync::LazyLock::new(|| Mat::zeros(0, 0));
                        &EMPTY
                    }
                }
            }
        };
        let f1 = match &self.nodes[1] {
            HtNode3D::Leaf { frame, .. } => frame,
            _ => {
                debug_assert!(false, "Node 1 is not a leaf");
                static EMPTY: std::sync::LazyLock<Mat<f64>> =
                    std::sync::LazyLock::new(|| Mat::zeros(0, 0));
                &EMPTY
            }
        };
        let f2 = match &self.nodes[2] {
            HtNode3D::Leaf { frame, .. } => frame,
            _ => {
                debug_assert!(false, "Node 2 is not a leaf");
                static EMPTY: std::sync::LazyLock<Mat<f64>> =
                    std::sync::LazyLock::new(|| Mat::zeros(0, 0));
                &EMPTY
            }
        };
        (f0, f1, f2)
    }

    /// Extract interior transfer tensor and ranks.
    fn interior_transfer(&self, idx: usize) -> (&[f64], [usize; 3]) {
        match &self.nodes[idx] {
            HtNode3D::Interior {
                transfer, ranks, ..
            } => (transfer, *ranks),
            _ => {
                debug_assert!(false, "Node {idx} is not interior");
                (&[], [0, 0, 0])
            }
        }
    }

    /// Add two HT3D tensors via rank concatenation (block-diagonal transfer tensors).
    ///
    /// The result has ranks equal to the sum of the input ranks. Call `truncate()`
    /// afterward to reduce ranks if needed.
    pub fn add(&self, other: &HtTensor3D) -> HtTensor3D {
        assert_eq!(self.shape, other.shape, "Shape mismatch in HT3D addition");

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        // Concatenate leaf frames: [U_a | U_b]
        for leaf_idx in 0..NUM_LEAVES_3D {
            let (frame_a, dim_a) = match &self.nodes[leaf_idx] {
                HtNode3D::Leaf { dim, frame } => (frame, *dim),
                _ => {
                    debug_assert!(false, "Expected leaf at index {leaf_idx}");
                    continue;
                }
            };
            let frame_b = match &other.nodes[leaf_idx] {
                HtNode3D::Leaf { frame, .. } => frame,
                _ => {
                    debug_assert!(false, "Expected leaf at index {leaf_idx}");
                    continue;
                }
            };

            let n = frame_a.nrows();
            let ka = frame_a.ncols();
            let kb = frame_b.ncols();
            let mut new_frame = Mat::zeros(n, ka + kb);
            for r in 0..n {
                for c in 0..ka {
                    new_frame[(r, c)] = frame_a[(r, c)];
                }
                for c in 0..kb {
                    new_frame[(r, ka + c)] = frame_b[(r, c)];
                }
            }
            nodes.push(HtNode3D::Leaf {
                dim: dim_a,
                frame: new_frame,
            });
        }

        // Interior node 3: block-diagonal transfer
        let (ta, ra) = self.interior_transfer(3);
        let (tb, rb) = other.interior_transfer(3);

        let new_kt3 = ra[0] + rb[0];
        let new_kl3 = ra[1] + rb[1]; // k1_a + k1_b
        let new_kr3 = ra[2] + rb[2]; // k2_a + k2_b
        let mut new_t3 = vec![0.0; new_kt3 * new_kl3 * new_kr3];

        // Block A: top-left
        for t in 0..ra[0] {
            for l in 0..ra[1] {
                for r in 0..ra[2] {
                    new_t3[t * new_kl3 * new_kr3 + l * new_kr3 + r] =
                        ta[t * ra[1] * ra[2] + l * ra[2] + r];
                }
            }
        }
        // Block B: bottom-right
        for t in 0..rb[0] {
            for l in 0..rb[1] {
                for r in 0..rb[2] {
                    new_t3[(ra[0] + t) * new_kl3 * new_kr3 + (ra[1] + l) * new_kr3 + (ra[2] + r)] =
                        tb[t * rb[1] * rb[2] + l * rb[2] + r];
                }
            }
        }

        nodes.push(HtNode3D::Interior {
            left: 1,
            right: 2,
            transfer: new_t3,
            ranks: [new_kt3, new_kl3, new_kr3],
        });

        // Root node 4: block-diagonal in (l, r) indices, root rank stays 1.
        // Both A and B have root rank 1, so:
        //   B_sum[0, l, r] = B_A[0, l_A, r_A]  for l in [0..k0_A), r in [0..k3_A)
        //   B_sum[0, k0_A+l_B, k3_A+r_B] = B_B[0, l_B, r_B]
        let (ta_root, ra_root) = self.interior_transfer(4);
        let (tb_root, rb_root) = other.interior_transfer(4);

        let new_kt_root = 1; // Root rank stays 1 for scalar addition
        let new_kl_root = ra_root[1] + rb_root[1]; // k0_a + k0_b
        let new_kr_root = ra_root[2] + rb_root[2]; // k3_a + k3_b
        let mut new_t_root = vec![0.0; new_kl_root * new_kr_root];

        // Block A: B_sum[0, l, r] = B_A[0, l, r]
        for l in 0..ra_root[1] {
            for r in 0..ra_root[2] {
                new_t_root[l * new_kr_root + r] = ta_root[l * ra_root[2] + r]; // ta_root is [1, k0_A, k3_A] with t=0
            }
        }
        // Block B: B_sum[0, k0_A+l, k3_A+r] = B_B[0, l, r]
        for l in 0..rb_root[1] {
            for r in 0..rb_root[2] {
                new_t_root[(ra_root[1] + l) * new_kr_root + (ra_root[2] + r)] =
                    tb_root[l * rb_root[2] + r];
            }
        }

        nodes.push(HtNode3D::Interior {
            left: 0,
            right: 3,
            transfer: new_t_root,
            ranks: [new_kt_root, new_kl_root, new_kr_root],
        });

        HtTensor3D {
            nodes,
            shape: self.shape,
            dx: self.dx,
        }
    }

    /// Truncate ranks in-place via HSVD: bottom-up QR orthogonalization of leaves,
    /// then top-down SVD truncation at the root. Complexity is O(n*k^2 + k^3),
    /// much cheaper than a full dense roundtrip at O(N^3).
    pub fn truncate(&mut self, eps: f64, max_rank: usize) {
        // ── Phase 1: Bottom-up orthogonalization ──
        // QR each leaf frame, absorb R into parent transfer tensor.

        // Leaf 0: U₀ = Q₀ · R₀
        let (q0, r0) = leaf_qr(&self.nodes[0]);
        self.nodes[0] = HtNode3D::Leaf { dim: 0, frame: q0 };

        // Leaf 1: U₁ = Q₁ · R₁
        let (q1, r1) = leaf_qr(&self.nodes[1]);
        self.nodes[1] = HtNode3D::Leaf { dim: 1, frame: q1 };

        // Leaf 2: U₂ = Q₂ · R₂
        let (q2, r2) = leaf_qr(&self.nodes[2]);
        self.nodes[2] = HtNode3D::Leaf { dim: 2, frame: q2 };

        // Interior node 3: absorb R₁ (left) and R₂ (right) into B₃.
        // For each parent-rank slice t: B₃_abs[t] = R₁ · B₃[t] · R₂ᵀ
        // Then QR the matricization M[l·k₂+r, t] = B₃_abs[t, l, r]
        // to orthogonalize node 3's parent dimension.
        let (b3, rk3) = match &self.nodes[3] {
            HtNode3D::Interior {
                transfer, ranks, ..
            } => (transfer.clone(), *ranks),
            _ => {
                debug_assert!(false, "Node 3 not interior");
                return;
            }
        };
        let [kt3, k1, k2] = rk3;

        // Absorb: B₃_abs[t, l', r'] = Σ_l Σ_r R₁[l', l] · B₃[t, l, r] · R₂[r', r]
        // R₁ is min(n₁, k₁) × k₁, R₂ is min(n₂, k₂) × k₂
        let k1_new = r1.nrows(); // min(n1, k1) — may be less than k1 if rank > grid
        let k2_new = r2.nrows();
        let mut b3_abs = vec![0.0f64; kt3 * k1_new * k2_new];
        for t in 0..kt3 {
            for lp in 0..k1_new {
                for rp in 0..k2_new {
                    let mut sum = 0.0;
                    for l in 0..k1 {
                        let r1_lp_l = r1[(lp, l)];
                        if r1_lp_l.abs() < 1e-30 {
                            continue;
                        }
                        for r in 0..k2 {
                            sum += r1_lp_l * b3[t * k1 * k2 + l * k2 + r] * r2[(rp, r)];
                        }
                    }
                    b3_abs[t * k1_new * k2_new + lp * k2_new + rp] = sum;
                }
            }
        }

        // Matricize B₃_abs as M ∈ ℝ^{k1·k2 × kt3}: M[l·k2+r, t] = B₃_abs[t, l, r]
        let mat3_rows = k1_new * k2_new;
        let mut mat3 = Mat::<f64>::zeros(mat3_rows, kt3);
        for t in 0..kt3 {
            for l in 0..k1_new {
                for r in 0..k2_new {
                    mat3[(l * k2_new + r, t)] = b3_abs[t * k1_new * k2_new + l * k2_new + r];
                }
            }
        }

        // QR of M: M = Q₃ · R₃
        let qr3 = mat3.as_ref().qr();
        let q3 = qr3.compute_thin_Q(); // (k1·k2, min(k1·k2, kt3))
        let r3 = qr3.thin_R().to_owned(); // (min(k1·k2, kt3), kt3)
        let kt3_orth = q3.ncols(); // orthogonalized parent rank

        // Set B₃ transfer ← Q₃ reshaped: B₃_new[t', l, r] = Q₃[l·k2+r, t']
        let mut b3_orth = vec![0.0f64; kt3_orth * k1_new * k2_new];
        for tp in 0..kt3_orth {
            for l in 0..k1_new {
                for r in 0..k2_new {
                    b3_orth[tp * k1_new * k2_new + l * k2_new + r] = q3[(l * k2_new + r, tp)];
                }
            }
        }

        // ── Root node 4: absorb R₀ and R₃ ──
        // B₄_abs = R₀ · B₄[0] · R₃ᵀ   (k0 × kt3 matrix)
        let (b4, rk4) = match &self.nodes[4] {
            HtNode3D::Interior {
                transfer, ranks, ..
            } => (transfer.clone(), *ranks),
            _ => {
                debug_assert!(false, "Node 4 not interior");
                return;
            }
        };
        let [_kt4, k0, k3_old] = rk4;

        // R₃ is (kt3_orth, kt3) — transforms old parent rank to new orthogonal rank
        // B₄[0] is (k0, k3_old) where k3_old = kt3
        // B₄_abs = R₀ · B₄[0] · R₃ᵀ
        let k0_new = r0.nrows(); // min(n0, k0)
        let mut b4_abs = Mat::<f64>::zeros(k0_new, kt3_orth);
        for jp in 0..k0_new {
            for tp in 0..kt3_orth {
                let mut sum = 0.0;
                for j0 in 0..k0 {
                    let r0_jp_j0 = r0[(jp, j0)];
                    if r0_jp_j0.abs() < 1e-30 {
                        continue;
                    }
                    for j3 in 0..k3_old {
                        sum += r0_jp_j0 * b4[j0 * k3_old + j3] * r3[(tp, j3)];
                    }
                }
                b4_abs[(jp, tp)] = sum;
            }
        }

        // ── Phase 2: SVD and truncation at root ──
        let svd_result = b4_abs.as_ref().thin_svd();
        let (new_rank, new_leaf0, new_b3, new_b4) = match svd_result {
            Ok(svd) => {
                let s_col = svd.S().column_vector();
                let sv: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
                let u_full = svd.U();
                let v_full = svd.V();
                let trunc_k = truncation_rank(&sv, eps)
                    .min(max_rank)
                    .max(1)
                    .min(u_full.ncols());

                // New leaf 0: Q₀ · U[:,0:k] · diag(√σ)
                let leaf0_frame = match &self.nodes[0] {
                    HtNode3D::Leaf { frame, .. } => frame,
                    _ => unreachable!(),
                };
                let mut new_leaf0 = Mat::<f64>::zeros(leaf0_frame.nrows(), trunc_k);
                for i in 0..leaf0_frame.nrows() {
                    for j in 0..trunc_k {
                        let mut val = 0.0;
                        for jp in 0..k0_new {
                            val += leaf0_frame[(i, jp)] * u_full[(jp, j)];
                        }
                        new_leaf0[(i, j)] = val * sv[j].sqrt();
                    }
                }

                // New node 3 transfer: B₃_final[j, l, r] = Σ_{t'} √σ[j] · V[t', j] · B₃_orth[t', l, r]
                let mut new_b3 = vec![0.0f64; trunc_k * k1_new * k2_new];
                for j in 0..trunc_k {
                    let sqrt_s = sv[j].sqrt();
                    for l in 0..k1_new {
                        for r in 0..k2_new {
                            let mut sum = 0.0;
                            for tp in 0..kt3_orth {
                                sum += v_full[(tp, j)]
                                    * b3_orth[tp * k1_new * k2_new + l * k2_new + r];
                            }
                            new_b3[j * k1_new * k2_new + l * k2_new + r] = sqrt_s * sum;
                        }
                    }
                }

                // New root: identity k × k
                let mut new_b4 = vec![0.0f64; trunc_k * trunc_k];
                for j in 0..trunc_k {
                    new_b4[j * trunc_k + j] = 1.0;
                }

                (trunc_k, new_leaf0, new_b3, new_b4)
            }
            Err(_) => {
                // SVD failed — fall back to rank 1
                let leaf0_frame = match &self.nodes[0] {
                    HtNode3D::Leaf { frame, .. } => frame,
                    _ => unreachable!(),
                };
                let n0 = leaf0_frame.nrows();
                let mut new_leaf0 = Mat::<f64>::zeros(n0, 1);
                new_leaf0[(0, 0)] = 1.0;
                let new_b3 = vec![0.0f64; k1_new * k2_new];
                let new_b4 = vec![1.0f64; 1];
                (1, new_leaf0, new_b3, new_b4)
            }
        };

        // ── Phase 3: Assemble final nodes ──
        self.nodes[0] = HtNode3D::Leaf {
            dim: 0,
            frame: new_leaf0,
        };
        // Leaves 1 and 2 keep their orthogonalized frames (already set above)

        self.nodes[3] = HtNode3D::Interior {
            left: 1,
            right: 2,
            transfer: new_b3,
            ranks: [new_rank, k1_new, k2_new],
        };

        self.nodes[4] = HtNode3D::Interior {
            left: 0,
            right: 3,
            transfer: new_b4,
            ranks: [1, new_rank, new_rank],
        };
    }

    /// Create a zero-valued HT3D tensor with rank 1 in all nodes.
    pub fn zero(shape: [usize; 3], dx: [f64; 3]) -> Self {
        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for d in 0..NUM_LEAVES_3D {
            let n = shape[d];
            let mut frame = Mat::zeros(n, 1);
            frame[(0, 0)] = 1.0; // arbitrary orthonormal column
            nodes.push(HtNode3D::Leaf { dim: d, frame });
        }

        nodes.push(HtNode3D::Interior {
            left: 1,
            right: 2,
            transfer: vec![0.0; 1],
            ranks: [1, 1, 1],
        });

        nodes.push(HtNode3D::Interior {
            left: 0,
            right: 3,
            transfer: vec![0.0; 1],
            ranks: [1, 1, 1],
        });

        Self { nodes, shape, dx }
    }

    /// Return the rank of every node in the dimension tree (length 5).
    pub fn ranks(&self) -> Vec<usize> {
        self.nodes.iter().map(|n| n.rank()).collect()
    }

    /// Frobenius norm squared: `||T||^2 = sum T[i,j,k]^2`. Currently computed via dense expansion.
    pub fn norm_sq(&self) -> f64 {
        let data = self.to_full_3d();
        data.iter().map(|x| x * x).sum()
    }
}

// ─── Complex 3D HT tensor for Fourier-space operations ──────────────────────

/// A node in the complex 3D HT dimension tree.
///
/// Leaf frames are complex (split real/imaginary storage); transfer tensors remain
/// real because FFT only acts on leaves and does not change the rank structure.
#[derive(Clone)]
pub enum HtNode3DComplex {
    /// Leaf: complex frame stored as separate real and imaginary matrices.
    Leaf {
        /// Dimension index (0 = x1, 1 = x2, 2 = x3).
        dim: usize,
        /// Real part of the leaf frame, shape (n, k).
        frame_re: Mat<f64>,
        /// Imaginary part of the leaf frame, shape (n, k).
        frame_im: Mat<f64>,
    },
    /// Interior: real transfer tensor (identical layout to `HtNode3D::Interior`).
    Interior {
        /// Index of the left child node.
        left: usize,
        /// Index of the right child node.
        right: usize,
        /// Flattened transfer tensor of size k_t * k_left * k_right.
        transfer: Vec<f64>,
        /// Rank triple [k_t, k_left, k_right].
        ranks: [usize; 3],
    },
}

impl HtNode3DComplex {
    /// Return the rank of this complex node.
    #[inline]
    pub fn rank(&self) -> usize {
        match self {
            HtNode3DComplex::Leaf { frame_re, .. } => frame_re.ncols(),
            HtNode3DComplex::Interior { ranks, .. } => ranks[0],
        }
    }
}

/// Complex 3D HT tensor for Fourier-space operations (e.g., Poisson solve via FFT).
///
/// Leaf frames carry complex values (split real/imaginary) while transfer tensors
/// remain real. Produced by `HtTensor3D::fft_leaves()` and converted back via
/// `ifft_leaves()`.
#[derive(Clone)]
pub struct HtTensor3DComplex {
    /// All 5 complex nodes (indices 0-2: leaves, 3: interior, 4: root).
    pub nodes: Vec<HtNode3DComplex>,
    /// Grid dimensions [n_x1, n_x2, n_x3].
    pub shape: [usize; 3],
    /// Cell spacings [dx1, dx2, dx3].
    pub dx: [f64; 3],
}

impl HtTensor3DComplex {
    /// Evaluate the complex tensor at a 3D index. Returns (real, imag).
    pub fn evaluate(&self, idx: [usize; 3]) -> (f64, f64) {
        let (u0_re, u0_im) = self.leaf_vector_complex(0, idx[0]);
        let (u1_re, u1_im) = self.leaf_vector_complex(1, idx[1]);
        let (u2_re, u2_im) = self.leaf_vector_complex(2, idx[2]);

        // Node 3: contract u1 and u2 (complex multiplication)
        let (z3_re, z3_im) = self.contract_interior_complex(3, &u1_re, &u1_im, &u2_re, &u2_im);

        // Root: contract u0 and z3
        let (z4_re, z4_im) = self.contract_interior_complex(4, &u0_re, &u0_im, &z3_re, &z3_im);

        (z4_re[0], z4_im[0])
    }

    #[inline]
    fn leaf_vector_complex(&self, node: usize, idx: usize) -> (Vec<f64>, Vec<f64>) {
        match &self.nodes[node] {
            HtNode3DComplex::Leaf {
                frame_re, frame_im, ..
            } => {
                let k = frame_re.ncols();
                let re: Vec<f64> = (0..k).map(|j| frame_re[(idx, j)]).collect();
                let im: Vec<f64> = (0..k).map(|j| frame_im[(idx, j)]).collect();
                (re, im)
            }
            _ => {
                debug_assert!(false, "Node {node} is not a leaf");
                (vec![], vec![])
            }
        }
    }

    /// Contract interior node with complex left and right vectors.
    /// Transfer tensor B is real; complex arithmetic on vectors only.
    #[inline]
    fn contract_interior_complex(
        &self,
        node: usize,
        left_re: &[f64],
        left_im: &[f64],
        right_re: &[f64],
        right_im: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        match &self.nodes[node] {
            HtNode3DComplex::Interior {
                transfer, ranks, ..
            } => {
                let [kt, kl, kr] = *ranks;
                let mut re = vec![0.0; kt];
                let mut im = vec![0.0; kt];
                for t in 0..kt {
                    let mut sum_re = 0.0;
                    let mut sum_im = 0.0;
                    for l in 0..kl {
                        for r in 0..kr {
                            let b = transfer[t * kl * kr + l * kr + r];
                            // (a_re + i*a_im) * (b_re + i*b_im) where B is real
                            // = B * (lr_re + i*lr_im)
                            // lr = left * right (complex)
                            let lr_re = left_re[l] * right_re[r] - left_im[l] * right_im[r];
                            let lr_im = left_re[l] * right_im[r] + left_im[l] * right_re[r];
                            sum_re += b * lr_re;
                            sum_im += b * lr_im;
                        }
                    }
                    re[t] = sum_re;
                    im[t] = sum_im;
                }
                (re, im)
            }
            _ => {
                debug_assert!(false, "Node {node} is not an interior node");
                (vec![], vec![])
            }
        }
    }

    /// Return the rank of every node in the complex dimension tree (length 5).
    pub fn ranks(&self) -> Vec<usize> {
        self.nodes.iter().map(|n| n.rank()).collect()
    }
}

impl HtTensor3D {
    /// Apply 1D forward FFT to each column of each leaf frame, producing a complex HT3D.
    /// Transfer tensors are copied unchanged so rank is exactly preserved.
    pub fn fft_leaves(&self) -> HtTensor3DComplex {
        use rustfft::FftPlanner;
        use rustfft::num_complex::Complex64;

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in &self.nodes {
            match node {
                HtNode3D::Leaf { dim, frame } => {
                    let n = frame.nrows();
                    let k = frame.ncols();
                    let mut frame_re = Mat::zeros(n, k);
                    let mut frame_im = Mat::zeros(n, k);

                    let mut planner = FftPlanner::new();
                    let fft = planner.plan_fft_forward(n);

                    for col in 0..k {
                        let mut buffer: Vec<Complex64> = (0..n)
                            .map(|r| Complex64::new(frame[(r, col)], 0.0))
                            .collect();
                        fft.process(&mut buffer);
                        for r in 0..n {
                            frame_re[(r, col)] = buffer[r].re;
                            frame_im[(r, col)] = buffer[r].im;
                        }
                    }

                    nodes.push(HtNode3DComplex::Leaf {
                        dim: *dim,
                        frame_re,
                        frame_im,
                    });
                }
                HtNode3D::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3DComplex::Interior {
                        left: *left,
                        right: *right,
                        transfer: transfer.clone(),
                        ranks: *ranks,
                    });
                }
            }
        }

        HtTensor3DComplex {
            nodes,
            shape: self.shape,
            dx: self.dx,
        }
    }

    /// Apply 1D forward FFT to each leaf column using precomputed `rustfft` plans.
    /// `plans[d]` must match the number of rows in leaf d's frame.
    pub fn fft_leaves_with_plans(
        &self,
        plans: &[std::sync::Arc<dyn rustfft::Fft<f64>>; 3],
    ) -> HtTensor3DComplex {
        use rustfft::num_complex::Complex64;

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in &self.nodes {
            match node {
                HtNode3D::Leaf { dim, frame } => {
                    let n = frame.nrows();
                    let k = frame.ncols();
                    let mut frame_re = Mat::zeros(n, k);
                    let mut frame_im = Mat::zeros(n, k);
                    let fft = &plans[*dim];

                    for col in 0..k {
                        let mut buffer: Vec<Complex64> = (0..n)
                            .map(|r| Complex64::new(frame[(r, col)], 0.0))
                            .collect();
                        fft.process(&mut buffer);
                        for r in 0..n {
                            frame_re[(r, col)] = buffer[r].re;
                            frame_im[(r, col)] = buffer[r].im;
                        }
                    }

                    nodes.push(HtNode3DComplex::Leaf {
                        dim: *dim,
                        frame_re,
                        frame_im,
                    });
                }
                HtNode3D::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3DComplex::Interior {
                        left: *left,
                        right: *right,
                        transfer: transfer.clone(),
                        ranks: *ranks,
                    });
                }
            }
        }

        HtTensor3DComplex {
            nodes,
            shape: self.shape,
            dx: self.dx,
        }
    }

    /// Consuming variant of [`fft_leaves_with_plans`](Self::fft_leaves_with_plans) that moves
    /// transfer tensors instead of cloning them.
    pub fn into_fft_with_plans(
        self,
        plans: &[std::sync::Arc<dyn rustfft::Fft<f64>>; 3],
    ) -> HtTensor3DComplex {
        use rustfft::num_complex::Complex64;

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in self.nodes {
            match node {
                HtNode3D::Leaf { dim, frame } => {
                    let n = frame.nrows();
                    let k = frame.ncols();
                    let mut frame_re = Mat::zeros(n, k);
                    let mut frame_im = Mat::zeros(n, k);
                    let fft = &plans[dim];

                    for col in 0..k {
                        let mut buffer: Vec<Complex64> = (0..n)
                            .map(|r| Complex64::new(frame[(r, col)], 0.0))
                            .collect();
                        fft.process(&mut buffer);
                        for r in 0..n {
                            frame_re[(r, col)] = buffer[r].re;
                            frame_im[(r, col)] = buffer[r].im;
                        }
                    }

                    nodes.push(HtNode3DComplex::Leaf {
                        dim,
                        frame_re,
                        frame_im,
                    });
                }
                HtNode3D::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3DComplex::Interior {
                        left,
                        right,
                        transfer,
                        ranks,
                    });
                }
            }
        }

        HtTensor3DComplex {
            nodes,
            shape: self.shape,
            dx: self.dx,
        }
    }

    /// Construct a rank-1 HT3D tensor from three 1D vectors as an outer product:
    /// `T[i, j, k] = v0[i] * v1[j] * v2[k]`.
    pub fn from_rank1(v0: &[f64], v1: &[f64], v2: &[f64], dx: [f64; 3]) -> Self {
        let shape = [v0.len(), v1.len(), v2.len()];
        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        // Leaf frames: single column = the vector
        for (d, vec) in [v0, v1, v2].iter().enumerate() {
            let n = vec.len();
            let mut frame = Mat::zeros(n, 1);
            for i in 0..n {
                frame[(i, 0)] = vec[i];
            }
            nodes.push(HtNode3D::Leaf { dim: d, frame });
        }

        // Interior node 3: B[0, 0, 0] = 1.0 (rank 1)
        nodes.push(HtNode3D::Interior {
            left: 1,
            right: 2,
            transfer: vec![1.0],
            ranks: [1, 1, 1],
        });

        // Root node 4: B[0, 0, 0] = 1.0 (rank 1)
        nodes.push(HtNode3D::Interior {
            left: 0,
            right: 3,
            transfer: vec![1.0],
            ranks: [1, 1, 1],
        });

        Self { nodes, shape, dx }
    }
}

impl HtTensor3DComplex {
    /// Apply 1D inverse FFT to each leaf column, returning a real HT3D.
    /// Includes the 1/N normalization factor.
    pub fn ifft_leaves(&self) -> HtTensor3D {
        use rustfft::FftPlanner;
        use rustfft::num_complex::Complex64;

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in &self.nodes {
            match node {
                HtNode3DComplex::Leaf {
                    dim,
                    frame_re,
                    frame_im,
                } => {
                    let n = frame_re.nrows();
                    let k = frame_re.ncols();
                    let mut frame = Mat::zeros(n, k);

                    let mut planner = FftPlanner::new();
                    let ifft = planner.plan_fft_inverse(n);
                    let scale = 1.0 / n as f64;

                    for col in 0..k {
                        let mut buffer: Vec<Complex64> = (0..n)
                            .map(|r| Complex64::new(frame_re[(r, col)], frame_im[(r, col)]))
                            .collect();
                        ifft.process(&mut buffer);
                        for r in 0..n {
                            frame[(r, col)] = buffer[r].re * scale;
                        }
                    }

                    nodes.push(HtNode3D::Leaf { dim: *dim, frame });
                }
                HtNode3DComplex::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3D::Interior {
                        left: *left,
                        right: *right,
                        transfer: transfer.clone(),
                        ranks: *ranks,
                    });
                }
            }
        }

        HtTensor3D {
            nodes,
            shape: self.shape,
            dx: self.dx,
        }
    }

    /// Apply 1D inverse FFT to each leaf column using precomputed `rustfft` plans.
    /// `plans[d]` must be inverse FFT plans matching leaf d's frame row count.
    pub fn ifft_leaves_with_plans(
        &self,
        plans: &[std::sync::Arc<dyn rustfft::Fft<f64>>; 3],
    ) -> HtTensor3D {
        use rustfft::num_complex::Complex64;

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in &self.nodes {
            match node {
                HtNode3DComplex::Leaf {
                    dim,
                    frame_re,
                    frame_im,
                } => {
                    let n = frame_re.nrows();
                    let k = frame_re.ncols();
                    let mut frame = Mat::zeros(n, k);
                    let ifft = &plans[*dim];
                    let scale = 1.0 / n as f64;

                    for col in 0..k {
                        let mut buffer: Vec<Complex64> = (0..n)
                            .map(|r| Complex64::new(frame_re[(r, col)], frame_im[(r, col)]))
                            .collect();
                        ifft.process(&mut buffer);
                        for r in 0..n {
                            frame[(r, col)] = buffer[r].re * scale;
                        }
                    }

                    nodes.push(HtNode3D::Leaf { dim: *dim, frame });
                }
                HtNode3DComplex::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3D::Interior {
                        left: *left,
                        right: *right,
                        transfer: transfer.clone(),
                        ranks: *ranks,
                    });
                }
            }
        }

        HtTensor3D {
            nodes,
            shape: self.shape,
            dx: self.dx,
        }
    }

    /// Consuming variant of [`ifft_leaves_with_plans`](Self::ifft_leaves_with_plans) that moves
    /// transfer tensors instead of cloning them.
    pub fn into_ifft_with_plans(
        self,
        plans: &[std::sync::Arc<dyn rustfft::Fft<f64>>; 3],
    ) -> HtTensor3D {
        use rustfft::num_complex::Complex64;

        let mut nodes = Vec::with_capacity(NUM_NODES_3D);

        for node in self.nodes {
            match node {
                HtNode3DComplex::Leaf {
                    dim,
                    frame_re,
                    frame_im,
                } => {
                    let n = frame_re.nrows();
                    let k = frame_re.ncols();
                    let mut frame = Mat::zeros(n, k);
                    let ifft = &plans[dim];
                    let scale = 1.0 / n as f64;

                    for col in 0..k {
                        let mut buffer: Vec<Complex64> = (0..n)
                            .map(|r| Complex64::new(frame_re[(r, col)], frame_im[(r, col)]))
                            .collect();
                        ifft.process(&mut buffer);
                        for r in 0..n {
                            frame[(r, col)] = buffer[r].re * scale;
                        }
                    }

                    nodes.push(HtNode3D::Leaf { dim, frame });
                }
                HtNode3DComplex::Interior {
                    left,
                    right,
                    transfer,
                    ranks,
                } => {
                    nodes.push(HtNode3D::Interior {
                        left,
                        right,
                        transfer,
                        ranks,
                    });
                }
            }
        }

        HtTensor3D {
            nodes,
            shape: self.shape,
            dx: self.dx,
        }
    }
}

// ─── Utility functions ───────────────────────────────────────────────────────

/// QR-factorize a leaf node's frame: U = Q · R.
/// Returns (Q, R) where Q has orthonormal columns.
fn leaf_qr(node: &HtNode3D) -> (Mat<f64>, Mat<f64>) {
    match node {
        HtNode3D::Leaf { frame, .. } => {
            let m = frame.nrows();
            let n = frame.ncols();
            if m == 0 || n == 0 {
                return (Mat::zeros(m.max(1), 1), Mat::zeros(1, n.max(1)));
            }
            let qr = frame.as_ref().qr();
            (qr.compute_thin_Q(), qr.thin_R().to_owned())
        }
        _ => {
            debug_assert!(false, "leaf_qr called on non-leaf node");
            (Mat::zeros(1, 1), Mat::zeros(1, 1))
        }
    }
}

/// Mode-k unfolding SVD: given data matricized as (n_k, rest), compute thin SVD
/// and truncate. Returns (truncated_rank, leaf_frame).
fn mode_unfolding_svd(
    data: &[f64],
    rows: usize,
    cols: usize,
    eps: f64,
    max_rank: usize,
) -> (usize, Mat<f64>) {
    let mut mat: Mat<f64> = Mat::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            mat[(r, c)] = data[r * cols + c];
        }
    }

    let svd = mat.as_ref().thin_svd();
    match svd {
        Ok(svd) => {
            let s_col = svd.S().column_vector();
            let sv: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
            let rank = truncation_rank(&sv, eps)
                .min(max_rank)
                .min(svd.U().ncols())
                .max(1);

            // Extract first `rank` columns of U
            let u = svd.U();
            let mut frame = Mat::zeros(rows, rank);
            for r in 0..rows {
                for c in 0..rank {
                    frame[(r, c)] = u[(r, c)];
                }
            }

            (rank, frame)
        }
        Err(_) => {
            let mut frame = Mat::zeros(rows, 1);
            frame[(0, 0)] = 1.0;
            (1, frame)
        }
    }
}

/// Truncation rank via tail-norm criterion: find smallest k such that
/// ||sigma_{k+1:}||_2 ≤ eps.
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

/// Truncation rank for a faer Mat via SVD.
fn truncation_rank_mat(mat: &Mat<f64>, eps: f64) -> usize {
    if let Ok(svd) = mat.as_ref().thin_svd() {
        let s_col = svd.S().column_vector();
        let sv: Vec<f64> = (0..s_col.nrows()).map(|i| s_col[i]).collect();
        truncation_rank(&sv, eps)
    } else {
        1
    }
}

/// Extract an orthonormal frame from a matrix via column-pivoted QR.
fn extract_frame_qr(mat: &Mat<f64>, max_rank: usize) -> Mat<f64> {
    let m = mat.nrows();
    let n = mat.ncols();
    if m == 0 || n == 0 {
        return Mat::zeros(m.max(1), 1);
    }

    let qr = mat.as_ref().col_piv_qr();
    let q = qr.compute_thin_Q();
    let r = qr.thin_R();

    // Determine effective rank from R diagonal
    let k = m.min(n);
    let mut rank = 0;
    let threshold = 1e-12 * r[(0, 0)].abs();
    for i in 0..k {
        if i < r.nrows() && i < r.ncols() && r[(i, i)].abs() > threshold {
            rank = i + 1;
        }
    }
    rank = rank.max(1).min(max_rank);

    // Extract first `rank` columns of Q
    let mut frame = Mat::zeros(m, rank);
    for r in 0..m {
        for c in 0..rank {
            if c < q.ncols() {
                frame[(r, c)] = q[(r, c)];
            }
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ht3d_rank1_roundtrip() {
        let v0: Vec<f64> = (0..8).map(|i| i as f64 + 1.0).collect();
        let v1: Vec<f64> = (0..8).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let v2: Vec<f64> = (0..8).map(|i| (i as f64 + 1.0) * 0.3).collect();
        let dx = [0.25; 3];

        let ht = HtTensor3D::from_rank1(&v0, &v1, &v2, dx);

        let mut max_err = 0.0f64;
        for i0 in 0..8 {
            for i1 in 0..8 {
                for i2 in 0..8 {
                    let expected = v0[i0] * v1[i1] * v2[i2];
                    let got = ht.evaluate([i0, i1, i2]);
                    max_err = max_err.max((got - expected).abs());
                }
            }
        }
        assert!(max_err < 1e-12, "Rank-1 round-trip error: {max_err}");
    }

    #[test]
    fn ht3d_dense_roundtrip() {
        let shape = [4, 4, 4];
        let n = 64;
        let mut data = vec![0.0; n];

        // Simple separable function
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    data[i0 * 16 + i1 * 4 + i2] =
                        (i0 as f64 + 1.0) * (i1 as f64 + 0.5) + (i2 as f64) * 0.1;
                }
            }
        }

        let dx = [0.5; 3];
        let ht = HtTensor3D::from_dense(&data, shape, dx, 1e-10, 10);
        let reconstructed = ht.to_full_3d();

        let max_err: f64 = data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(
            max_err < 1e-8,
            "Dense round-trip error: {max_err}, ranks: {:?}",
            ht.ranks()
        );
    }

    #[test]
    fn ht3d_zero_pad_and_extract() {
        let v0: Vec<f64> = (0..4).map(|i| i as f64 + 1.0).collect();
        let v1: Vec<f64> = (0..4).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let v2: Vec<f64> = (0..4).map(|i| (i as f64 + 1.0) * 0.3).collect();
        let dx = [0.5; 3];

        let ht = HtTensor3D::from_rank1(&v0, &v1, &v2, dx);
        let padded = ht.zero_pad();

        assert_eq!(padded.shape, [8, 8, 8]);

        // First N entries should match original
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    let expected = v0[i0] * v1[i1] * v2[i2];
                    let got = padded.evaluate([i0, i1, i2]);
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "Padded value mismatch at [{i0},{i1},{i2}]: {got} vs {expected}"
                    );
                }
            }
        }

        // Padded entries should be zero
        for i2 in 4..8 {
            let val = padded.evaluate([0, 0, i2]);
            assert!(val.abs() < 1e-12, "Padded region should be zero, got {val}");
        }

        // Extract back
        let extracted = padded.extract_subgrid([4, 4, 4]);
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    let expected = v0[i0] * v1[i1] * v2[i2];
                    let got = extracted.evaluate([i0, i1, i2]);
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "Extracted value mismatch at [{i0},{i1},{i2}]"
                    );
                }
            }
        }
    }

    #[test]
    fn ht3d_addition() {
        let dx = [0.5; 3];
        let a = HtTensor3D::from_rank1(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[0.5, 0.5, 0.5, 0.5],
            dx,
        );
        let b = HtTensor3D::from_rank1(
            &[0.0, 0.0, 0.0, 1.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 1.0, 1.0, 1.0],
            dx,
        );

        let sum = a.add(&b);

        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    let expected = a.evaluate([i0, i1, i2]) + b.evaluate([i0, i1, i2]);
                    let got = sum.evaluate([i0, i1, i2]);
                    assert!(
                        (got - expected).abs() < 1e-10,
                        "Addition mismatch at [{i0},{i1},{i2}]: {got} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn ht3d_fft_roundtrip() {
        let v0: Vec<f64> = (0..8).map(|i| (-(i as f64 - 3.5).powi(2)).exp()).collect();
        let v1: Vec<f64> = (0..8).map(|i| (-(i as f64 - 3.5).powi(2)).exp()).collect();
        let v2: Vec<f64> = (0..8).map(|i| (-(i as f64 - 3.5).powi(2)).exp()).collect();
        let dx = [0.25; 3];

        let ht = HtTensor3D::from_rank1(&v0, &v1, &v2, dx);
        let original = ht.to_full_3d();

        // FFT then IFFT should recover original
        let complex = ht.fft_leaves();
        let recovered = complex.ifft_leaves();
        let recovered_data = recovered.to_full_3d();

        let max_err: f64 = original
            .iter()
            .zip(recovered_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(max_err < 1e-10, "FFT round-trip error: {max_err}");
    }

    #[test]
    fn ht3d_fft_preserves_rank() {
        let v0: Vec<f64> = (0..8).map(|i| i as f64 + 1.0).collect();
        let v1: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
        let v2: Vec<f64> = (0..8).map(|i| i as f64 * 0.3 + 1.0).collect();
        let dx = [0.25; 3];

        let ht = HtTensor3D::from_rank1(&v0, &v1, &v2, dx);
        let ranks_before = ht.ranks();

        let complex = ht.fft_leaves();
        let ranks_after = complex.ranks();

        assert_eq!(
            ranks_before, ranks_after,
            "FFT should preserve ranks: before={ranks_before:?}, after={ranks_after:?}"
        );
    }

    #[test]
    fn ht3d_from_function_aca() {
        let shape = [8, 8, 8];
        let dx = [0.25; 3];

        // Low-rank function: Gaussian blob
        let f = |idx: [usize; 3]| -> f64 {
            let x = idx[0] as f64 - 3.5;
            let y = idx[1] as f64 - 3.5;
            let z = idx[2] as f64 - 3.5;
            (-(x * x + y * y + z * z) / 4.0).exp()
        };

        let ht = HtTensor3D::from_function_aca(&f, shape, dx, 1e-6, 10);

        let mut max_err = 0.0f64;
        for i0 in 0..8 {
            for i1 in 0..8 {
                for i2 in 0..8 {
                    let expected = f([i0, i1, i2]);
                    let got = ht.evaluate([i0, i1, i2]);
                    max_err = max_err.max((got - expected).abs());
                }
            }
        }

        assert!(
            max_err < 1e-4,
            "ACA construction error: {max_err}, ranks: {:?}",
            ht.ranks()
        );
    }

    #[test]
    fn ht3d_truncate_inplace_vs_dense() {
        let dx = [0.5; 3];
        let shape = [8, 8, 8];

        // Build a rank-2 tensor by adding two rank-1 tensors (inflated rank)
        let a = HtTensor3D::from_rank1(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            dx,
        );
        let b = HtTensor3D::from_rank1(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dx,
        );
        let sum = a.add(&b);

        // Get reference dense data before truncation
        let reference = sum.to_full_3d();

        // In-place truncation
        let mut ht_inplace = sum.clone();
        ht_inplace.truncate(1e-10, 10);

        // Compare
        let reconstructed = ht_inplace.to_full_3d();
        let max_err: f64 = reference
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        let norm: f64 = reference.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert!(
            max_err / (norm + 1e-15) < 1e-8,
            "In-place truncation error: {max_err}, norm: {norm}, ranks: {:?}",
            ht_inplace.ranks()
        );
    }

    #[test]
    fn ht3d_to_dense_subgrid() {
        let dx = [0.5; 3];
        let shape = [8, 8, 8];

        // Build a non-trivial rank-2 tensor
        let a = HtTensor3D::from_rank1(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            dx,
        );
        let b = HtTensor3D::from_rank1(
            &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            &[0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4],
            dx,
        );
        let sum = a.add(&b);

        // Full extraction
        let full = sum.to_full_3d();

        // Subgrid extraction [4, 4, 4]
        let sub = sum.to_dense_subgrid([4, 4, 4]);
        assert_eq!(sub.len(), 64);

        let mut max_err = 0.0f64;
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    let expected = full[i0 * 64 + i1 * 8 + i2];
                    let got = sub[i0 * 16 + i1 * 4 + i2];
                    max_err = max_err.max((expected - got).abs());
                }
            }
        }
        assert!(max_err < 1e-10, "Subgrid extraction error: {max_err}");
    }
}
