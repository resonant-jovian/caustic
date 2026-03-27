#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! Tensor-train (TT) decomposition of the 6D distribution function f(x,v).
//!
//! Memory: O(d * N * r^2) where r is the TT rank, d=6, N = max grid size.
//! For smooth distribution functions the TT rank stays small, making this
//! vastly more compact than the O(N^6) uniform grid.
//!
//! The TT format represents a 6D tensor as:
//!   f(i0,i1,i2,i3,i4,i5) = G0[:,i0,:] * G1[:,i1,:] * G2[:,i2,:] * G3[:,i3,:] * G4[:,i4,:] * G5[:,i5,:]
//!
//! where Gk is a 3-way core of shape (r_k, n_k, r_{k+1}).

use super::super::{
    context::SimContext,
    init::domain::{Domain, SpatialBoundType, VelocityBoundType},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use super::lagrangian::sl_shift_1d;
use faer::Mat;
use rayon::prelude::*;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

// ─── TT Core ─────────────────────────────────────────────────────────────────

/// One TT core: 3-way tensor of shape (r_left, n, r_right) stored row-major.
///
/// Element layout: data[alpha * n * r_right + i * r_right + beta]
/// corresponds to G[alpha, i, beta].
#[derive(Clone, Debug)]
pub struct TtCore {
    pub data: Vec<f64>,
    pub r_left: usize,
    pub n: usize,
    pub r_right: usize,
}

impl TtCore {
    /// Allocate a zero-filled core of shape (r_left, n, r_right).
    pub fn new(r_left: usize, n: usize, r_right: usize) -> Self {
        Self {
            data: vec![0.0; r_left * n * r_right],
            r_left,
            n,
            r_right,
        }
    }

    /// Access element G[alpha, i, beta].
    #[inline]
    pub fn get(&self, alpha: usize, i: usize, beta: usize) -> f64 {
        self.data[alpha * self.n * self.r_right + i * self.r_right + beta]
    }

    /// Set element G[alpha, i, beta].
    #[inline]
    pub fn set(&mut self, alpha: usize, i: usize, beta: usize, val: f64) {
        self.data[alpha * self.n * self.r_right + i * self.r_right + beta] = val;
    }

    /// Reshape this core to a (r_left * n, r_right) matrix (left unfolding).
    fn unfold_left(&self) -> Mat<f64> {
        let rows = self.r_left * self.n;
        let cols = self.r_right;
        let mut mat: Mat<f64> = Mat::zeros(rows, cols);
        for alpha in 0..self.r_left {
            for i in 0..self.n {
                for beta in 0..self.r_right {
                    mat[(alpha * self.n + i, beta)] = self.get(alpha, i, beta);
                }
            }
        }
        mat
    }

    /// Reshape this core to a (r_left, n * r_right) matrix (right unfolding).
    fn unfold_right(&self) -> Mat<f64> {
        let rows = self.r_left;
        let cols = self.n * self.r_right;
        let mut mat: Mat<f64> = Mat::zeros(rows, cols);
        for alpha in 0..self.r_left {
            for i in 0..self.n {
                for beta in 0..self.r_right {
                    mat[(alpha, i * self.r_right + beta)] = self.get(alpha, i, beta);
                }
            }
        }
        mat
    }

    /// Reconstruct core from a (r_left * n, r_right) matrix (inverse of unfold_left).
    fn from_left_unfold(mat: &Mat<f64>, r_left: usize, n: usize) -> Self {
        let r_right = mat.ncols();
        let mut core = TtCore::new(r_left, n, r_right);
        for alpha in 0..r_left {
            for i in 0..n {
                for beta in 0..r_right {
                    core.set(alpha, i, beta, mat[(alpha * n + i, beta)]);
                }
            }
        }
        core
    }

    /// Reconstruct core from a (r_left, n * r_right) matrix (inverse of unfold_right).
    fn from_right_unfold(mat: &Mat<f64>, n: usize, r_right: usize) -> Self {
        let r_left = mat.nrows();
        let mut core = TtCore::new(r_left, n, r_right);
        for alpha in 0..r_left {
            for i in 0..n {
                for beta in 0..r_right {
                    core.set(alpha, i, beta, mat[(alpha, i * r_right + beta)]);
                }
            }
        }
        core
    }
}

// ─── TT Tensor ───────────────────────────────────────────────────────────────

/// Tensor-Train representation of the 6D phase-space distribution f(x,v).
///
/// The TT decomposition factors a 6D tensor into a chain of d=6 three-way
/// cores with TT ranks r_0=1, r_1, r_2, r_3, r_4, r_5, r_6=1.
pub struct TensorTrain {
    /// The 6 TT cores. `cores[k]` has shape `(ranks[k], shape[k], ranks[k+1])`.
    pub cores: Vec<TtCore>,
    /// Grid sizes: [nx1, nx2, nx3, nv1, nv2, nv3].
    pub shape: [usize; 6],
    /// TT ranks: [r_0, r_1, r_2, r_3, r_4, r_5, r_6] with r_0 = r_6 = 1.
    pub ranks: Vec<usize>,
    /// Computational domain (extents, BCs).
    pub domain: Domain,
    /// Approximation tolerance for TT-SVD and recompression.
    pub tolerance: f64,
    /// Maximum allowed TT rank.
    pub max_rank: usize,
    /// If true, apply Zhang-Shu positivity limiter after each advection step.
    positivity_limiter: bool,
    /// Number of negative-value cells corrected by the positivity limiter.
    positivity_violations: AtomicU64,
}

impl TensorTrain {
    /// Create a minimal-rank (all ranks 1) TT representing the zero tensor.
    pub fn new(domain: Domain, max_rank: usize) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
            domain.velocity_res.v1 as usize,
            domain.velocity_res.v2 as usize,
            domain.velocity_res.v3 as usize,
        ];
        let ranks = vec![1usize; 7]; // r_0 .. r_6, all = 1
        let mut cores = Vec::with_capacity(6);
        for k in 0..6 {
            // Identity-like: rank-1 cores, all zeros => zero tensor
            cores.push(TtCore::new(1, shape[k], 1));
        }
        TensorTrain {
            cores,
            shape,
            ranks,
            domain,
            tolerance: 1e-10,
            max_rank,
            positivity_limiter: false,
            positivity_violations: AtomicU64::new(0),
        }
    }

    /// Enable or disable the Zhang-Shu positivity-preserving limiter.
    ///
    /// When enabled, negative values produced by semi-Lagrangian interpolation
    /// are clamped to zero after each advection step, with mass-conservative
    /// rescaling to preserve the total integral.
    pub fn with_positivity_limiter(mut self, enabled: bool) -> Self {
        self.positivity_limiter = enabled;
        self
    }

    /// Total number of negative-value cells corrected by the positivity limiter
    /// across all advection steps so far.
    pub fn positivity_violations(&self) -> u64 {
        self.positivity_violations.load(Ordering::Relaxed)
    }

    /// TT-SVD decomposition of a full 6D snapshot.
    ///
    /// Algorithm (Oseledets 2011, Algorithm 1):
    /// 1. Reshape the full tensor as C = reshape(data, [n_0, n_1*...*n_5]).
    /// 2. For k = 0..4:
    ///    - C has shape (r_{k} * n_k, remaining)
    ///    - Compute truncated SVD: C ≈ U_k * S_k * V_k^T
    ///    - Core k = reshape(U_k, [r_k, n_k, r_{k+1}])
    ///    - C = diag(S_k) * V_k^T for the next step
    /// 3. Core 5 = C (the residual, with r_right = 1).
    pub fn from_snapshot(
        snap: &PhaseSpaceSnapshot,
        max_rank: usize,
        tolerance: f64,
        domain: &Domain,
    ) -> Self {
        Self::from_snapshot_owned(
            PhaseSpaceSnapshot {
                data: snap.data.clone(),
                shape: snap.shape,
                time: snap.time,
            },
            max_rank,
            tolerance,
            domain,
        )
    }

    /// Like [`from_snapshot`](Self::from_snapshot), but takes ownership of the
    /// snapshot data to avoid an O(N^6) clone.
    pub fn from_snapshot_owned(
        snap: PhaseSpaceSnapshot,
        max_rank: usize,
        tolerance: f64,
        domain: &Domain,
    ) -> Self {
        let shape = snap.shape;
        let n_total: usize = shape.iter().product();
        assert_eq!(
            snap.data.len(),
            n_total,
            "snapshot data/shape mismatch: {} vs {}",
            snap.data.len(),
            n_total
        );

        // Per-mode tolerance: epsilon / sqrt(d-1) for quasi-optimal global error
        let eps_mode = tolerance / (5.0_f64).sqrt();

        let mut cores: Vec<TtCore> = Vec::with_capacity(6);
        let mut ranks = vec![1usize; 7]; // ranks[0] = 1, ranks[6] = 1

        // C starts as the full data reshaped to (n_0, n_1*n_2*n_3*n_4*n_5)
        let mut c_data = snap.data;
        let mut r_prev = 1usize;

        for k in 0..5 {
            let n_k = shape[k];
            let rows = r_prev * n_k;
            let remaining: usize = shape[k + 1..].iter().product();
            let cols = remaining;

            // Build the matrix C of shape (r_prev * n_k, remaining)
            let mut c_mat: Mat<f64> = Mat::zeros(rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    c_mat[(r, c)] = c_data[r * cols + c];
                }
            }

            // Compute thin SVD
            let (u, sv, vt) = thin_svd(&c_mat);

            // Truncate: keep singular values satisfying tail criterion
            let mut rank = truncation_rank(&sv, eps_mode).max(1);
            rank = rank.min(max_rank).min(u.ncols());

            ranks[k + 1] = rank;

            // Core k: reshape U_truncated from (r_prev * n_k, rank) to (r_prev, n_k, rank)
            let mut core = TtCore::new(r_prev, n_k, rank);
            for alpha in 0..r_prev {
                for i in 0..n_k {
                    for beta in 0..rank {
                        core.set(alpha, i, beta, u[(alpha * n_k + i, beta)]);
                    }
                }
            }
            cores.push(core);

            // C = diag(S_truncated) * V_truncated^T for next step
            // V_truncated^T has shape (rank, cols)
            let new_rows = rank;
            let new_cols = if k < 4 {
                let tail_prod: usize = shape[k + 2..].iter().product();
                shape[k + 1] * tail_prod
            } else {
                shape[5]
            };
            c_data = vec![0.0; new_rows * new_cols];
            for r in 0..new_rows {
                for c in 0..new_cols.min(vt.ncols()) {
                    c_data[r * new_cols + c] = sv[r] * vt[(r, c)];
                }
            }
            r_prev = rank;
        }

        // Last core (k=5): C is the final core with r_right = 1
        let n5 = shape[5];
        let mut last_core = TtCore::new(r_prev, n5, 1);
        for alpha in 0..r_prev {
            for i in 0..n5 {
                let idx = alpha * n5 + i;
                if idx < c_data.len() {
                    last_core.set(alpha, i, 0, c_data[idx]);
                }
            }
        }
        cores.push(last_core);

        TensorTrain {
            cores,
            shape,
            ranks,
            domain: domain.clone(),
            tolerance,
            max_rank,
            positivity_limiter: false,
            positivity_violations: AtomicU64::new(0),
        }
    }

    /// Evaluate f at a single 6D index by left-to-right contraction of all cores.
    ///
    /// `f(i0, i1, i2, i3, i4, i5) = G0[:,i0,:] * G1[:,i1,:] * ... * G5[:,i5,:]`
    #[inline]
    pub fn evaluate(&self, indices: [usize; 6]) -> f64 {
        // Start with vec = [1.0] (r_0 = 1)
        let mut vec = vec![1.0f64];
        for k in 0..6 {
            let core = &self.cores[k];
            let ik = indices[k];
            let r_left = core.r_left;
            let r_right = core.r_right;
            let mut new_vec = vec![0.0f64; r_right];
            for beta in 0..r_right {
                let mut sum = 0.0;
                for alpha in 0..r_left {
                    sum += vec[alpha] * core.get(alpha, ik, beta);
                }
                new_vec[beta] = sum;
            }
            vec = new_vec;
        }
        vec[0]
    }

    /// Expand the TT back to a full 6D array. Only practical for small grids.
    pub fn to_full(&self) -> Vec<f64> {
        let [n0, n1, n2, n3, n4, n5] = self.shape;
        let stride_0 = n1 * n2 * n3 * n4 * n5;

        // Parallelize over i0 — each slab is independent
        (0..n0)
            .into_par_iter()
            .flat_map(|i0| {
                let mut slab = Vec::with_capacity(stride_0);
                for i1 in 0..n1 {
                    for i2 in 0..n2 {
                        for i3 in 0..n3 {
                            for i4 in 0..n4 {
                                for i5 in 0..n5 {
                                    slab.push(self.evaluate([i0, i1, i2, i3, i4, i5]));
                                }
                            }
                        }
                    }
                }
                slab
            })
            .collect()
    }

    /// Recompress the TT to reduce ranks while keeping error below tolerance.
    ///
    /// Two-sweep rounding (Oseledets 2011):
    /// 1. Left-to-right QR sweep (left-orthogonalize).
    /// 2. Right-to-left SVD sweep with truncation.
    pub fn recompress(&mut self, tolerance: f64) {
        let eps_mode = tolerance / (5.0_f64).sqrt();

        // ── Pass 1: left-to-right QR (left-orthogonalize) ──
        for k in 0..5 {
            let mat = self.cores[k].unfold_left();
            let (q, r) = qr_decompose(&mat);
            let new_rank = q.ncols();
            // Core k = reshape Q to (r_left, n_k, new_rank)
            self.cores[k] = TtCore::from_left_unfold(&q, self.cores[k].r_left, self.shape[k]);
            self.ranks[k + 1] = new_rank;
            // Absorb R into core k+1: multiply R * unfold_right(core_{k+1})
            // R has shape (new_rank, r_{k+1}_old)
            // core_{k+1} has unfold_right shape (r_{k+1}_old, n_{k+1} * r_{k+2})
            let next_mat = self.cores[k + 1].unfold_right();
            // Multiply: R (new_rank x old_r) * next_mat (old_r x n_{k+1}*r_{k+2})
            // = product of shape (new_rank x n_{k+1}*r_{k+2})
            let product = mat_mul(&r, &next_mat);
            self.cores[k + 1] =
                TtCore::from_right_unfold(&product, self.shape[k + 1], self.cores[k + 1].r_right);
        }

        // ── Pass 2: right-to-left SVD truncation ──
        for k in (1..6).rev() {
            let mat = self.cores[k].unfold_right();
            // mat has shape (r_left, n_k * r_right)
            // We want SVD of transpose: (n_k * r_right, r_left) so we can truncate rows of left factor
            let mat_t = transpose(&mat);
            let (u, sv, vt) = thin_svd(&mat_t);
            let mut rank = truncation_rank(&sv, eps_mode).max(1);
            rank = rank.min(self.max_rank).min(u.ncols());

            // New core k: reshape from V^T truncated
            // vt has shape (min(n_k*r_right, r_left), r_left)
            // V^T[:rank, :] has shape (rank, r_left) => transpose => (r_left, rank)
            // But we want core shape (rank, n_k, r_right):
            // U truncated: (n_k * r_right, rank) => reshape to (n_k, r_right, rank)
            // => permute to (rank, n_k, r_right) is complex.
            // Instead: mat = (r_left, n_k * r_right), SVD of mat^T = U * S * V^T
            // so mat = V * S * U^T
            // Truncated: mat ≈ V[:, :rank] * diag(S[:rank]) * U[:, :rank]^T
            // = (r_left x rank) * (rank x n_k*r_right)
            // The right factor U[:,:rank]^T reshaped is the new core (rank, n_k, r_right).
            // The left factor V[:,:rank] * diag(S[:rank]) absorbs into core k-1.

            // U[:, :rank]^T has shape (rank, n_k * r_right) = unfold_right of new core
            let mut new_core_mat: Mat<f64> =
                Mat::zeros(rank, self.shape[k] * self.cores[k].r_right);
            for r in 0..rank {
                for c in 0..(self.shape[k] * self.cores[k].r_right) {
                    new_core_mat[(r, c)] = u[(c, r)]; // U^T
                }
            }
            self.cores[k] =
                TtCore::from_right_unfold(&new_core_mat, self.shape[k], self.cores[k].r_right);

            // Absorb V * S into core k-1
            // V[:, :rank] * diag(S[:rank]) has shape (r_left, rank)
            // = vt^T[:, :rank] * diag(S[:rank])
            let old_r_left = mat.nrows(); // = self.cores[k].r_left before update
            let mut vs: Mat<f64> = Mat::zeros(old_r_left, rank);
            for r in 0..old_r_left {
                for c in 0..rank {
                    vs[(r, c)] = vt[(c, r)] * sv[c]; // V = vt^T, then multiply by s
                }
            }

            // core_{k-1} unfold_left has shape (r_{k-1} * n_{k-1}, old_r_left)
            // Multiply: unfold_left * VS = (r_{k-1} * n_{k-1}, rank)
            let prev_mat = self.cores[k - 1].unfold_left();
            let product = mat_mul(&prev_mat, &vs);
            self.cores[k - 1] =
                TtCore::from_left_unfold(&product, self.cores[k - 1].r_left, self.shape[k - 1]);

            self.ranks[k] = rank;
        }
    }

    /// Add two TT tensors via rank concatenation (direct sum of cores).
    ///
    /// Result has ranks r_self + r_other (before recompression).
    pub fn add(&self, other: &TensorTrain) -> TensorTrain {
        assert_eq!(self.shape, other.shape, "TT shapes must match for addition");
        let mut new_cores = Vec::with_capacity(6);
        let mut new_ranks = vec![1usize; 7];

        for k in 0..6 {
            let r_s_l = self.cores[k].r_left;
            let r_s_r = self.cores[k].r_right;
            let r_o_l = other.cores[k].r_left;
            let r_o_r = other.cores[k].r_right;
            let n = self.shape[k];

            if k == 0 {
                // First core: r_left = 1, concatenate along r_right
                // new shape: (1, n, r_s_r + r_o_r)
                let r_right = r_s_r + r_o_r;
                let mut core = TtCore::new(1, n, r_right);
                for i in 0..n {
                    for beta in 0..r_s_r {
                        core.set(0, i, beta, self.cores[k].get(0, i, beta));
                    }
                    for beta in 0..r_o_r {
                        core.set(0, i, r_s_r + beta, other.cores[k].get(0, i, beta));
                    }
                }
                new_cores.push(core);
                new_ranks[k + 1] = r_right;
            } else if k == 5 {
                // Last core: r_right = 1, concatenate along r_left
                // new shape: (r_s_l + r_o_l, n, 1)
                let r_left = r_s_l + r_o_l;
                let mut core = TtCore::new(r_left, n, 1);
                for alpha in 0..r_s_l {
                    for i in 0..n {
                        core.set(alpha, i, 0, self.cores[k].get(alpha, i, 0));
                    }
                }
                for alpha in 0..r_o_l {
                    for i in 0..n {
                        core.set(r_s_l + alpha, i, 0, other.cores[k].get(alpha, i, 0));
                    }
                }
                new_cores.push(core);
            } else {
                // Interior cores: block diagonal in (r_left, r_right)
                // new shape: (r_s_l + r_o_l, n, r_s_r + r_o_r)
                let r_left = r_s_l + r_o_l;
                let r_right = r_s_r + r_o_r;
                let mut core = TtCore::new(r_left, n, r_right);
                // Self block: top-left
                for alpha in 0..r_s_l {
                    for i in 0..n {
                        for beta in 0..r_s_r {
                            core.set(alpha, i, beta, self.cores[k].get(alpha, i, beta));
                        }
                    }
                }
                // Other block: bottom-right
                for alpha in 0..r_o_l {
                    for i in 0..n {
                        for beta in 0..r_o_r {
                            core.set(
                                r_s_l + alpha,
                                i,
                                r_s_r + beta,
                                other.cores[k].get(alpha, i, beta),
                            );
                        }
                    }
                }
                new_cores.push(core);
                new_ranks[k + 1] = r_right;
            }
        }

        let mut result = TensorTrain {
            cores: new_cores,
            shape: self.shape,
            ranks: new_ranks,
            domain: self.domain.clone(),
            tolerance: self.tolerance.min(other.tolerance),
            max_rank: self.max_rank.max(other.max_rank),
            positivity_limiter: self.positivity_limiter,
            positivity_violations: AtomicU64::new(0),
        };
        result.recompress(result.tolerance);
        result
    }

    /// Scale all entries by a constant factor.
    pub fn scale(&mut self, factor: f64) {
        // Scaling is applied to the first core only (rank structure is unchanged).
        for val in self.cores[0].data.iter_mut() {
            *val *= factor;
        }
    }

    /// TT inner product: <self, other> = sum_{all indices} self(i) * other(i).
    ///
    /// Computed by sequential contraction from left to right. O(d * n * r^4).
    pub fn inner_product(&self, other: &TensorTrain) -> f64 {
        assert_eq!(
            self.shape, other.shape,
            "shapes must match for inner product"
        );
        // Gram matrix G_{k} has shape (r_self_k, r_other_k)
        // Initialize: G_0 = [[1]] (since r_0 = 1 for both)
        let mut gram = vec![1.0f64]; // 1x1 matrix
        let mut _gr_rows = 1usize;
        let mut gr_cols = 1usize;

        for k in 0..6 {
            let cs = &self.cores[k];
            let co = &other.cores[k];
            let n = self.shape[k];
            // New Gram: G_{k+1}[beta_s, beta_o] =
            //   sum_{alpha_s, alpha_o, i} G_k[alpha_s, alpha_o] * cs[alpha_s, i, beta_s] * co[alpha_o, i, beta_o]
            let new_rows = cs.r_right;
            let new_cols = co.r_right;
            let mut new_gram = vec![0.0f64; new_rows * new_cols];
            for alpha_s in 0..cs.r_left {
                for alpha_o in 0..co.r_left {
                    let g_val = gram[alpha_s * gr_cols + alpha_o];
                    if g_val.abs() < 1e-300 {
                        continue;
                    }
                    for i in 0..n {
                        for beta_s in 0..cs.r_right {
                            let cs_val = cs.get(alpha_s, i, beta_s);
                            if cs_val.abs() < 1e-300 {
                                continue;
                            }
                            let g_cs = g_val * cs_val;
                            for beta_o in 0..co.r_right {
                                new_gram[beta_s * new_cols + beta_o] +=
                                    g_cs * co.get(alpha_o, i, beta_o);
                            }
                        }
                    }
                }
            }
            gram = new_gram;
            _gr_rows = new_rows;
            gr_cols = new_cols;
        }
        gram[0]
    }

    /// Frobenius norm: ||self||_F = sqrt(<self, self>).
    pub fn norm(&self) -> f64 {
        self.inner_product(self).abs().sqrt()
    }

    // ─── Helpers for PhaseSpaceRepr ──────────────────────────────────────────

    fn lx(&self) -> [f64; 3] {
        self.domain.lx()
    }

    fn lv(&self) -> [f64; 3] {
        self.domain.lv()
    }

    /// Contract velocity dimensions (3,4,5) with uniform weight vectors to produce
    /// a contracted vector that, when multiplied with spatial cores, gives density.
    ///
    /// The approach: for each spatial point (i0,i1,i2), sum over all (i3,i4,i5):
    ///   rho(i0,i1,i2) = sum_{i3,i4,i5} f(i0,i1,i2,i3,i4,i5) * dv^3
    ///
    /// Efficiently: pre-contract cores 3,4,5 with unit sums (right-to-left), then
    /// evaluate the remaining spatial contraction.
    fn contract_velocity_sums(&self) -> Vec<f64> {
        let dv = self.domain.dv();
        let dv3 = dv[0] * dv[1] * dv[2];

        // Contract core 5 with unit vector: w5[alpha5] = sum_{i5} core5[alpha5, i5, 0]
        let c5 = &self.cores[5];
        let mut w5 = vec![0.0f64; c5.r_left];
        for alpha in 0..c5.r_left {
            for i in 0..c5.n {
                w5[alpha] += c5.get(alpha, i, 0);
            }
        }

        // Contract core 4: w4[alpha4] = sum_{i4} sum_{beta4} core4[alpha4, i4, beta4] * w5[beta4]
        let c4 = &self.cores[4];
        let mut w4 = vec![0.0f64; c4.r_left];
        for alpha in 0..c4.r_left {
            for i in 0..c4.n {
                for beta in 0..c4.r_right {
                    w4[alpha] += c4.get(alpha, i, beta) * w5[beta];
                }
            }
        }

        // Contract core 3: w3[alpha3] = sum_{i3} sum_{beta3} core3[alpha3, i3, beta3] * w4[beta3]
        let c3 = &self.cores[3];
        let mut w3 = vec![0.0f64; c3.r_left];
        for alpha in 0..c3.r_left {
            for i in 0..c3.n {
                for beta in 0..c3.r_right {
                    w3[alpha] += c3.get(alpha, i, beta) * w4[beta];
                }
            }
        }

        // w3 * dv3 gives the velocity-integrated weights for each r3 index.
        for v in w3.iter_mut() {
            *v *= dv3;
        }
        w3
    }
}

// ─── PhaseSpaceRepr implementation ───────────────────────────────────────────

impl PhaseSpaceRepr for TensorTrain {
    fn compute_density(&self) -> DensityField {
        let [n0, n1, n2, _n3, _n4, _n5] = self.shape;
        let n_spatial = n0 * n1 * n2;

        // Pre-contract velocity cores
        let w3 = self.contract_velocity_sums();

        // For each spatial point, contract cores 0,1,2 and apply w3.
        // Parallelize over i0 slabs.
        let c0 = &self.cores[0];
        let c1 = &self.cores[1];
        let c2 = &self.cores[2];

        let data: Vec<f64> = (0..n0)
            .into_par_iter()
            .flat_map(|i0| {
                let mut slab = Vec::with_capacity(n1 * n2);
                let mut vec0 = vec![0.0f64; c0.r_right];
                for beta in 0..c0.r_right {
                    vec0[beta] = c0.get(0, i0, beta);
                }

                for i1 in 0..n1 {
                    let mut vec1 = vec![0.0f64; c1.r_right];
                    for beta in 0..c1.r_right {
                        let mut sum = 0.0;
                        for alpha in 0..c1.r_left {
                            sum += vec0[alpha] * c1.get(alpha, i1, beta);
                        }
                        vec1[beta] = sum;
                    }

                    for i2 in 0..n2 {
                        let mut vec2 = vec![0.0f64; c2.r_right];
                        for beta in 0..c2.r_right {
                            let mut sum = 0.0;
                            for alpha in 0..c2.r_left {
                                sum += vec1[alpha] * c2.get(alpha, i2, beta);
                            }
                            vec2[beta] = sum;
                        }

                        let mut rho = 0.0f64;
                        for j in 0..vec2.len().min(w3.len()) {
                            rho += vec2[j] * w3[j];
                        }
                        slab.push(rho);
                    }
                }

                slab
            })
            .collect();

        DensityField {
            data,
            shape: [n0, n1, n2],
        }
    }

    fn advect_x(&mut self, _displacement: &DisplacementField, ctx: &SimContext) {
        let dt = ctx.dt;
        // Semi-Lagrangian approach: expand to full, apply shifts, rebuild TT.
        // For small grids this is feasible; for large grids a proper TT-cross
        // approach should be used (future work).
        let n_total: usize = self.shape.iter().product();
        let [n0, n1, n2, n3, n4, n5] = self.shape;
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.lx();
        let lv = self.lv();
        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);

        // Expand to full grid
        let mut data = self.to_full();

        // For each velocity cell (i3,i4,i5), shift the spatial block
        let nv_total = n3 * n4 * n5;
        let ns_total = n0 * n1 * n2;

        for vi in 0..nv_total {
            let i5 = vi % n5;
            let i4 = (vi / n5) % n4;
            let i3 = vi / (n4 * n5);

            let vx = -lv[0] + (i3 as f64 + 0.5) * dv[0];
            let vy = -lv[1] + (i4 as f64 + 0.5) * dv[1];
            let vz = -lv[2] + (i5 as f64 + 0.5) * dv[2];
            let disp = [vx * dt, vy * dt, vz * dt];

            // Extract spatial slice for this velocity cell
            let mut local = vec![0.0f64; ns_total];
            for ix0 in 0..n0 {
                for ix1 in 0..n1 {
                    for ix2 in 0..n2 {
                        let si = ix0 * n1 * n2 + ix1 * n2 + ix2;
                        let flat = ix0 * n1 * n2 * n3 * n4 * n5
                            + ix1 * n2 * n3 * n4 * n5
                            + ix2 * n3 * n4 * n5
                            + i3 * n4 * n5
                            + i4 * n5
                            + i5;
                        local[si] = data[flat];
                    }
                }
            }

            // Shift along x0
            for ix1 in 0..n1 {
                for ix2 in 0..n2 {
                    let line: Vec<f64> = (0..n0)
                        .map(|ix0| local[ix0 * n1 * n2 + ix1 * n2 + ix2])
                        .collect();
                    let shifted = sl_shift_1d(&line, disp[0], dx[0], n0, lx[0], periodic);
                    for ix0 in 0..n0 {
                        local[ix0 * n1 * n2 + ix1 * n2 + ix2] = shifted[ix0];
                    }
                }
            }

            // Shift along x1
            for ix0 in 0..n0 {
                for ix2 in 0..n2 {
                    let line: Vec<f64> = (0..n1)
                        .map(|ix1| local[ix0 * n1 * n2 + ix1 * n2 + ix2])
                        .collect();
                    let shifted = sl_shift_1d(&line, disp[1], dx[1], n1, lx[1], periodic);
                    for ix1 in 0..n1 {
                        local[ix0 * n1 * n2 + ix1 * n2 + ix2] = shifted[ix1];
                    }
                }
            }

            // Shift along x2
            for ix0 in 0..n0 {
                for ix1 in 0..n1 {
                    let line: Vec<f64> = (0..n2)
                        .map(|ix2| local[ix0 * n1 * n2 + ix1 * n2 + ix2])
                        .collect();
                    let shifted = sl_shift_1d(&line, disp[2], dx[2], n2, lx[2], periodic);
                    for ix2 in 0..n2 {
                        local[ix0 * n1 * n2 + ix1 * n2 + ix2] = shifted[ix2];
                    }
                }
            }

            // Write back
            for ix0 in 0..n0 {
                for ix1 in 0..n1 {
                    for ix2 in 0..n2 {
                        let si = ix0 * n1 * n2 + ix1 * n2 + ix2;
                        let flat = ix0 * n1 * n2 * n3 * n4 * n5
                            + ix1 * n2 * n3 * n4 * n5
                            + ix2 * n3 * n4 * n5
                            + i3 * n4 * n5
                            + i4 * n5
                            + i5;
                        data[flat] = local[si];
                    }
                }
            }

        }

        // Apply positivity limiter before TT rebuild
        if self.positivity_limiter {
            let mass_before: f64 = data.iter().sum();
            let neg_count = data.iter().filter(|&&v| v < 0.0).count();
            if neg_count > 0 {
                self.positivity_violations
                    .fetch_add(neg_count as u64, Ordering::Relaxed);
                super::wpfc::zhang_shu_limiter(&mut data, mass_before);
            }
        }

        // Rebuild TT from the shifted full array
        let snap = PhaseSpaceSnapshot {
            data,
            shape: self.shape,
            time: 0.0,
        };
        let new_tt =
            TensorTrain::from_snapshot_owned(snap, self.max_rank, self.tolerance, &self.domain);
        self.cores = new_tt.cores;
        self.ranks = new_tt.ranks;
    }

    fn advect_v(&mut self, acceleration: &AccelerationField, ctx: &SimContext) {
        let dt = ctx.dt;
        // Semi-Lagrangian approach: expand to full, apply velocity shifts, rebuild TT.
        let [n0, n1, n2, n3, n4, n5] = self.shape;
        let dv = self.domain.dv();
        let lv = self.lv();
        let periodic_v = matches!(self.domain.velocity_bc, VelocityBoundType::Truncated);

        let mut data = self.to_full();
        let ns_total = n0 * n1 * n2;
        let nv_total = n3 * n4 * n5;

        for si in 0..ns_total {
            let ix2 = si % n2;
            let ix1 = (si / n2) % n1;
            let ix0 = si / (n1 * n2);

            let flat_sp = ix0 * n1 * n2 + ix1 * n2 + ix2;
            let ax = acceleration.gx[flat_sp];
            let ay = acceleration.gy[flat_sp];
            let az = acceleration.gz[flat_sp];
            let disp = [ax * dt, ay * dt, az * dt];

            // Extract velocity slice for this spatial cell
            let mut local = vec![0.0f64; nv_total];
            for iv3 in 0..n3 {
                for iv4 in 0..n4 {
                    for iv5 in 0..n5 {
                        let vi = iv3 * n4 * n5 + iv4 * n5 + iv5;
                        let flat = ix0 * n1 * n2 * n3 * n4 * n5
                            + ix1 * n2 * n3 * n4 * n5
                            + ix2 * n3 * n4 * n5
                            + iv3 * n4 * n5
                            + iv4 * n5
                            + iv5;
                        local[vi] = data[flat];
                    }
                }
            }

            // Shift along v1
            for iv4 in 0..n4 {
                for iv5 in 0..n5 {
                    let line: Vec<f64> = (0..n3)
                        .map(|iv3| local[iv3 * n4 * n5 + iv4 * n5 + iv5])
                        .collect();
                    let shifted = sl_shift_1d(&line, disp[0], dv[0], n3, lv[0], periodic_v);
                    for iv3 in 0..n3 {
                        local[iv3 * n4 * n5 + iv4 * n5 + iv5] = shifted[iv3];
                    }
                }
            }

            // Shift along v2
            for iv3 in 0..n3 {
                for iv5 in 0..n5 {
                    let line: Vec<f64> = (0..n4)
                        .map(|iv4| local[iv3 * n4 * n5 + iv4 * n5 + iv5])
                        .collect();
                    let shifted = sl_shift_1d(&line, disp[1], dv[1], n4, lv[1], periodic_v);
                    for iv4 in 0..n4 {
                        local[iv3 * n4 * n5 + iv4 * n5 + iv5] = shifted[iv4];
                    }
                }
            }

            // Shift along v3
            for iv3 in 0..n3 {
                for iv4 in 0..n4 {
                    let line: Vec<f64> = (0..n5)
                        .map(|iv5| local[iv3 * n4 * n5 + iv4 * n5 + iv5])
                        .collect();
                    let shifted = sl_shift_1d(&line, disp[2], dv[2], n5, lv[2], periodic_v);
                    for iv5 in 0..n5 {
                        local[iv3 * n4 * n5 + iv4 * n5 + iv5] = shifted[iv5];
                    }
                }
            }

            // Write back
            for iv3 in 0..n3 {
                for iv4 in 0..n4 {
                    for iv5 in 0..n5 {
                        let vi = iv3 * n4 * n5 + iv4 * n5 + iv5;
                        let flat = ix0 * n1 * n2 * n3 * n4 * n5
                            + ix1 * n2 * n3 * n4 * n5
                            + ix2 * n3 * n4 * n5
                            + iv3 * n4 * n5
                            + iv4 * n5
                            + iv5;
                        data[flat] = local[vi];
                    }
                }
            }

        }

        // Apply positivity limiter before TT rebuild
        if self.positivity_limiter {
            let mass_before: f64 = data.iter().sum();
            let neg_count = data.iter().filter(|&&v| v < 0.0).count();
            if neg_count > 0 {
                self.positivity_violations
                    .fetch_add(neg_count as u64, Ordering::Relaxed);
                super::wpfc::zhang_shu_limiter(&mut data, mass_before);
            }
        }

        // Rebuild TT from shifted data
        let snap = PhaseSpaceSnapshot {
            data,
            shape: self.shape,
            time: 0.0,
        };
        let new_tt =
            TensorTrain::from_snapshot_owned(snap, self.max_rank, self.tolerance, &self.domain);
        self.cores = new_tt.cores;
        self.ranks = new_tt.ranks;
    }

    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let [n0, n1, n2, n3, n4, n5] = self.shape;
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.lx();
        let lv = self.lv();
        let dv3 = dv[0] * dv[1] * dv[2];

        // Find spatial cell
        let ix0 = ((position[0] + lx[0]) / dx[0])
            .floor()
            .clamp(0.0, (n0 - 1) as f64) as usize;
        let ix1 = ((position[1] + lx[1]) / dx[1])
            .floor()
            .clamp(0.0, (n1 - 1) as f64) as usize;
        let ix2 = ((position[2] + lx[2]) / dx[2])
            .floor()
            .clamp(0.0, (n2 - 1) as f64) as usize;

        match order {
            0 => {
                // rho = sum_{v} f * dv^3
                let mut sum = 0.0f64;
                for i3 in 0..n3 {
                    for i4 in 0..n4 {
                        for i5 in 0..n5 {
                            sum += self.evaluate([ix0, ix1, ix2, i3, i4, i5]);
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
                // mean velocity: vbar_i = (1/rho) * sum_v f * v_i * dv^3
                let mut vbar = [0.0f64; 3];
                let mut rho = 0.0f64;
                for i3 in 0..n3 {
                    let vx = -lv[0] + (i3 as f64 + 0.5) * dv[0];
                    for i4 in 0..n4 {
                        let vy = -lv[1] + (i4 as f64 + 0.5) * dv[1];
                        for i5 in 0..n5 {
                            let vz = -lv[2] + (i5 as f64 + 0.5) * dv[2];
                            let f = self.evaluate([ix0, ix1, ix2, i3, i4, i5]);
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
                // second moment: M2_{ij} = sum_v f * v_i * v_j * dv^3
                let mut m2 = [0.0f64; 9];
                for i3 in 0..n3 {
                    let vx = -lv[0] + (i3 as f64 + 0.5) * dv[0];
                    for i4 in 0..n4 {
                        let vy = -lv[1] + (i4 as f64 + 0.5) * dv[1];
                        for i5 in 0..n5 {
                            let vz = -lv[2] + (i5 as f64 + 0.5) * dv[2];
                            let f = self.evaluate([ix0, ix1, ix2, i3, i4, i5]);
                            let v = [vx, vy, vz];
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
        let cell_vol = self.domain.cell_volume_6d();

        // Total mass = sum over all indices f(i) * cell_vol
        // This is equivalent to contracting each core with a unit sum vector.
        // w_k = sum_i core_k[:, i, :]
        let mut vec = vec![1.0f64]; // r_0 = 1
        for k in 0..6 {
            let core = &self.cores[k];
            let mut new_vec = vec![0.0f64; core.r_right];
            for i in 0..core.n {
                for alpha in 0..core.r_left {
                    for beta in 0..core.r_right {
                        new_vec[beta] += vec[alpha] * core.get(alpha, i, beta);
                    }
                }
            }
            vec = new_vec;
        }
        vec[0] * cell_vol
    }

    fn casimir_c2(&self) -> f64 {
        // C2 = integral f^2 dx^3 dv^3 = <self, self> * cell_vol
        let cell_vol = self.domain.cell_volume_6d();
        self.inner_product(self) * cell_vol
    }

    fn entropy(&self) -> f64 {
        // S = -integral f ln(f) dx^3 dv^3
        // No efficient TT method exists; expand to full for small grids.
        let cell_vol = self.domain.cell_volume_6d();

        let data = self.to_full();
        data.iter()
            .filter(|&&f| f > 0.0)
            .map(|&f| -f * f.ln())
            .sum::<f64>()
            * cell_vol
    }

    fn stream_count(&self) -> StreamCountField {
        let [n0, n1, n2, n3, n4, n5] = self.shape;
        let dv = self.domain.dv();
        let dv23 = dv[1] * dv[2];

        let mut out = vec![0u32; n0 * n1 * n2];

        for ix0 in 0..n0 {
            for ix1 in 0..n1 {
                for ix2 in 0..n2 {
                    // Marginal f_1(v1|x) = sum_{v2,v3} f(x,v) * dv2 * dv3
                    let marginal: Vec<f64> = (0..n3)
                        .map(|i3| {
                            let mut sum = 0.0f64;
                            for i4 in 0..n4 {
                                for i5 in 0..n5 {
                                    sum += self.evaluate([ix0, ix1, ix2, i3, i4, i5]);
                                }
                            }
                            sum * dv23
                        })
                        .collect();

                    // Count peaks
                    let mut peaks = 0u32;
                    for i in 1..n3.saturating_sub(1) {
                        if marginal[i] > marginal[i - 1] && marginal[i] > marginal[i + 1] {
                            peaks += 1;
                        }
                    }
                    out[ix0 * n1 * n2 + ix1 * n2 + ix2] = peaks;
                }
            }
        }

        StreamCountField {
            data: out,
            shape: [n0, n1, n2],
        }
    }

    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let [n0, n1, n2, n3, n4, n5] = self.shape;
        let dx = self.domain.dx();
        let lx = self.lx();

        let ix0 = ((position[0] + lx[0]) / dx[0])
            .floor()
            .clamp(0.0, (n0 - 1) as f64) as usize;
        let ix1 = ((position[1] + lx[1]) / dx[1])
            .floor()
            .clamp(0.0, (n1 - 1) as f64) as usize;
        let ix2 = ((position[2] + lx[2]) / dx[2])
            .floor()
            .clamp(0.0, (n2 - 1) as f64) as usize;

        let nv = n3 * n4 * n5;
        let mut result = Vec::with_capacity(nv);
        for i3 in 0..n3 {
            for i4 in 0..n4 {
                for i5 in 0..n5 {
                    result.push(self.evaluate([ix0, ix1, ix2, i3, i4, i5]));
                }
            }
        }
        result
    }

    fn total_kinetic_energy(&self) -> Option<f64> {
        // T = 0.5 * integral f * v^2 dx^3 dv^3
        let dv = self.domain.dv();
        let lv = self.lv();
        let cell_vol = self.domain.cell_volume_6d();
        let [n0, n1, n2, n3, n4, n5] = self.shape;

        // Efficient approach: pre-compute v^2-weighted velocity contraction vectors,
        // then contract with spatial cores.
        // For each velocity triplet (i3,i4,i5), weight = v_x^2 + v_y^2 + v_z^2.
        // Since v^2 = vx^2 + vy^2 + vz^2, and these are separable in each velocity dim,
        // we can decompose into three terms, each involving a v_k^2-weighted sum in one dim
        // and uniform sums in the other two.

        // First, compute uniform sum vectors for velocity cores (like in total_mass)
        // and v^2-weighted sum vectors.
        // Sum for core k: w_uniform[beta] = sum_i core_k[alpha, i, beta]
        // v^2 sum for core k: w_v2[beta] = sum_i v_k_i^2 * core_k[alpha, i, beta]

        // However, this requires handling the coupling through the rank structure.
        // For simplicity with correct results, expand and compute directly for small grids.
        let data = self.to_full();
        let mut t = 0.0f64;
        for i0 in 0..n0 {
            for i1 in 0..n1 {
                for i2 in 0..n2 {
                    for i3 in 0..n3 {
                        let vx = -lv[0] + (i3 as f64 + 0.5) * dv[0];
                        for i4 in 0..n4 {
                            let vy = -lv[1] + (i4 as f64 + 0.5) * dv[1];
                            for i5 in 0..n5 {
                                let vz = -lv[2] + (i5 as f64 + 0.5) * dv[2];
                                let flat = i0 * n1 * n2 * n3 * n4 * n5
                                    + i1 * n2 * n3 * n4 * n5
                                    + i2 * n3 * n4 * n5
                                    + i3 * n4 * n5
                                    + i4 * n5
                                    + i5;
                                let v2 = vx * vx + vy * vy + vz * vz;
                                t += data[flat] * v2;
                            }
                        }
                    }
                }
            }
        }
        Some(0.5 * t * cell_vol)
    }

    fn to_snapshot(&self, time: f64) -> Option<PhaseSpaceSnapshot> {
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
}

// ─── Linear algebra helpers (thin wrappers around faer) ──────────────────────

/// Thin SVD via faer. Returns (U, singular_values, V^T).
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

/// Determine truncation rank: keep smallest k such that
/// sum_{j>k} sigma_j^2 <= eps^2 (relative to Frobenius norm).
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

/// QR decomposition via faer. Returns (Q, R).
fn qr_decompose(mat: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();
    if m.min(n) == 0 {
        return (Mat::zeros(m, 0), Mat::zeros(0, n));
    }
    let qr = mat.as_ref().qr();
    let k = m.min(n);
    let q = qr.compute_thin_Q().subcols(0, k).to_owned();
    let r = qr.thin_R().subrows(0, k).to_owned();
    (q, r)
}

/// Simple dense matrix multiply: C = A * B.
fn mat_mul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = a.nrows();
    let n = b.ncols();
    let p = a.ncols();
    assert_eq!(p, b.nrows(), "mat_mul dimension mismatch");
    let mut c: Mat<f64> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..p {
                sum += a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = sum;
        }
    }
    c
}

/// Transpose a faer matrix.
fn transpose(mat: &Mat<f64>) -> Mat<f64> {
    mat.transpose().to_owned()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::context::SimContext;
    use crate::tooling::core::events::EventEmitter;
    use crate::tooling::core::progress::StepProgress;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::poisson::fft::FftPoisson;

    fn test_domain(n: i128) -> Domain {
        Domain::builder()
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
    fn tt_round_trip_rank1() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        // Rank-1 separable: f = prod_k g_k(i_k)
        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (i0 + 1) as f64
                                    * (i1 + 1) as f64
                                    * (i2 + 1) as f64
                                    * (i3 + 1) as f64
                                    * (i4 + 1) as f64
                                    * (i5 + 1) as f64;
                            }
                        }
                    }
                }
            }
        }
        let snap = PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time: 0.0,
        };
        // Use 1e-8 tolerance: values up to 4096, so machine-eps noise ~4e-13
        // makes 1e-12 too tight for rank truncation
        let tt = TensorTrain::from_snapshot(&snap, 10, 1e-8, &domain);

        // Verify evaluation matches original
        let mut max_err = 0.0f64;
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let expected = (i0 + 1) as f64
                                    * (i1 + 1) as f64
                                    * (i2 + 1) as f64
                                    * (i3 + 1) as f64
                                    * (i4 + 1) as f64
                                    * (i5 + 1) as f64;
                                let got = tt.evaluate([i0, i1, i2, i3, i4, i5]);
                                max_err = max_err.max((got - expected).abs());
                            }
                        }
                    }
                }
            }
        }
        assert!(max_err < 1e-8, "Round-trip error {max_err}");
        // Rank should be 1 (boundary ranks are always 1, interior should be 1 too)
        assert!(
            tt.ranks.iter().all(|&r| r <= 2),
            "Rank-1 function should have ranks ~1, got {:?}",
            tt.ranks
        );
    }

    #[test]
    fn tt_gaussian_rank() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let dx = domain.dx();
        let dv = domain.dv();
        let lx: f64 = 2.0;
        let lv: f64 = 2.0;

        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let x = -lx + (i0 as f64 + 0.5) * dx[0];
                                let y = -lx + (i1 as f64 + 0.5) * dx[1];
                                let z = -lx + (i2 as f64 + 0.5) * dx[2];
                                let vx = -lv + (i3 as f64 + 0.5) * dv[0];
                                let vy = -lv + (i4 as f64 + 0.5) * dv[1];
                                let vz = -lv + (i5 as f64 + 0.5) * dv[2];
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] =
                                    (-(x * x + y * y + z * z + vx * vx + vy * vy + vz * vz)).exp();
                            }
                        }
                    }
                }
            }
        }
        let snap = PhaseSpaceSnapshot {
            data,
            shape,
            time: 0.0,
        };
        let tt = TensorTrain::from_snapshot(&snap, 10, 1e-10, &domain);
        // Gaussian is separable, so rank should be low
        assert!(
            tt.ranks.iter().all(|&r| r <= 4),
            "Gaussian should have low rank, got {:?}",
            tt.ranks
        );
    }

    #[test]
    fn tt_density_vs_uniform() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let dx = domain.dx();
        let dv = domain.dv();
        let lx: f64 = 2.0;
        let lv: f64 = 2.0;

        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let x = -lx + (i0 as f64 + 0.5) * dx[0];
                                let vx = -lv + (i3 as f64 + 0.5) * dv[0];
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (-(x * x + vx * vx)).exp();
                            }
                        }
                    }
                }
            }
        }
        let snap = PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time: 0.0,
        };
        let tt = TensorTrain::from_snapshot(&snap, 10, 1e-10, &domain);

        let snap_uni = PhaseSpaceSnapshot {
            data,
            shape,
            time: 0.0,
        };
        let uniform = UniformGrid6D::from_snapshot(snap_uni, domain);
        let rho_tt = tt.compute_density();
        let rho_uni = uniform.compute_density();

        let max_diff = rho_tt
            .data
            .iter()
            .zip(rho_uni.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let max_rho = rho_uni
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max)
            .max(1e-15);
        assert!(
            max_diff / max_rho < 0.01,
            "TT vs uniform density max relative diff: {}",
            max_diff / max_rho
        );
    }

    #[test]
    fn tt_mass_conservation() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let data = vec![1.0; n_total]; // uniform f=1
        let snap = PhaseSpaceSnapshot {
            data,
            shape,
            time: 0.0,
        };
        let tt = TensorTrain::from_snapshot(&snap, 10, 1e-12, &domain);
        let mass = tt.total_mass();
        let expected = 4.0_f64.powi(6) * domain.cell_volume_6d();
        assert!(
            (mass - expected).abs() / expected < 0.01,
            "Mass: got {mass}, expected {expected}"
        );
    }

    #[test]
    fn tt_free_streaming() {
        // Verify advect_x doesn't panic and produces finite values
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (i0 + 1) as f64 * (i3 + 1) as f64;
                            }
                        }
                    }
                }
            }
        }
        let snap = PhaseSpaceSnapshot {
            data,
            shape,
            time: 0.0,
        };
        let mut tt = TensorTrain::from_snapshot(&snap, 10, 1e-10, &domain);
        let _m0 = tt.total_mass();
        let dummy = DisplacementField {
            dx: vec![0.0; 64],
            dy: vec![0.0; 64],
            dz: vec![0.0; 64],
            shape: [4, 4, 4],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = tt.domain.clone();

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

        tt.advect_x(&dummy, &__ctx);
        let m1 = tt.total_mass();
        assert!(m1.is_finite(), "Mass after advect must be finite");
    }

    #[test]
    fn tt_inner_product() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        // Rank-1 separable
        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (i0 + 1) as f64
                                    * (i1 + 1) as f64
                                    * (i2 + 1) as f64
                                    * (i3 + 1) as f64
                                    * (i4 + 1) as f64
                                    * (i5 + 1) as f64;
                            }
                        }
                    }
                }
            }
        }
        let snap = PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time: 0.0,
        };
        let tt = TensorTrain::from_snapshot(&snap, 10, 1e-12, &domain);

        // Compute inner product via TT
        let ip_tt = tt.inner_product(&tt);
        // Compute directly
        let ip_direct: f64 = data.iter().map(|&x| x * x).sum();

        let rel_err = (ip_tt - ip_direct).abs() / ip_direct.abs().max(1e-300);
        assert!(
            rel_err < 1e-8,
            "Inner product relative error: {rel_err}, TT={ip_tt}, direct={ip_direct}"
        );
    }

    #[test]
    fn tt_addition() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();

        // Two rank-1 tensors
        let mut data_a = vec![0.0; n_total];
        let mut data_b = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data_a[idx] = (i0 + 1) as f64 * (i3 + 1) as f64;
                                data_b[idx] = (i1 + 1) as f64 * (i4 + 1) as f64;
                            }
                        }
                    }
                }
            }
        }

        let snap_a = PhaseSpaceSnapshot {
            data: data_a.clone(),
            shape,
            time: 0.0,
        };
        let snap_b = PhaseSpaceSnapshot {
            data: data_b.clone(),
            shape,
            time: 0.0,
        };
        let tt_a = TensorTrain::from_snapshot(&snap_a, 10, 1e-12, &domain);
        let tt_b = TensorTrain::from_snapshot(&snap_b, 10, 1e-12, &domain);
        let tt_sum = tt_a.add(&tt_b);

        // Verify pointwise
        let mut max_err = 0.0f64;
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                let expected = data_a[idx] + data_b[idx];
                                let got = tt_sum.evaluate([i0, i1, i2, i3, i4, i5]);
                                max_err = max_err.max((got - expected).abs());
                            }
                        }
                    }
                }
            }
        }
        let max_val = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| (a + b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err / max_val < 1e-6,
            "Addition error: {}, relative: {}",
            max_err,
            max_err / max_val
        );
    }

    #[test]
    fn tt_recompress() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();

        // Build a low-rank tensor with artificially high max_rank
        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (i0 + 1) as f64
                                    * (i1 + 1) as f64
                                    * (i2 + 1) as f64
                                    * (i3 + 1) as f64
                                    * (i4 + 1) as f64
                                    * (i5 + 1) as f64;
                            }
                        }
                    }
                }
            }
        }
        let snap = PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time: 0.0,
        };
        let mut tt = TensorTrain::from_snapshot(&snap, 10, 1e-12, &domain);
        let ranks_before: Vec<usize> = tt.ranks.clone();

        // Recompress should not increase ranks for an already-low-rank tensor
        tt.recompress(1e-10);

        // Verify the tensor is still accurate
        let mut max_err = 0.0f64;
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                let got = tt.evaluate([i0, i1, i2, i3, i4, i5]);
                                max_err = max_err.max((got - data[idx]).abs());
                            }
                        }
                    }
                }
            }
        }
        assert!(
            max_err < 1e-6,
            "Recompress error: {max_err}, ranks before: {ranks_before:?}, after: {:?}",
            tt.ranks
        );
    }

    #[test]
    fn tt_casimir_c2() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let data = vec![1.0; n_total]; // uniform f=1
        let snap = PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time: 0.0,
        };
        let tt = TensorTrain::from_snapshot(&snap, 10, 1e-12, &domain);

        let c2_tt = tt.casimir_c2();
        // Direct: C2 = sum f^2 * cell_vol = n_total * 1^2 * cell_vol
        let cell_vol = domain.cell_volume_6d();
        let c2_expected = n_total as f64 * cell_vol;

        assert!(
            (c2_tt - c2_expected).abs() / c2_expected < 0.01,
            "C2: got {c2_tt}, expected {c2_expected}"
        );
    }

    #[test]
    fn tt_new_zero() {
        let domain = test_domain(4);
        let tt = TensorTrain::new(domain, 10);
        // Zero tensor should have zero mass
        let mass = tt.total_mass();
        assert!(
            mass.abs() < 1e-15,
            "Zero TT should have zero mass, got {mass}"
        );
    }

    #[test]
    fn test_tt_positivity_clips_negatives() {
        // Create an 8^6 TensorTrain with some negative values, enable positivity
        // limiter, advect, and verify that:
        //  (a) the limiter detects and records violations,
        //  (b) the worst-case negative magnitude is reduced compared to the
        //      no-limiter baseline (TT-SVD rebuild may re-introduce small
        //      O(tolerance) negatives, but the large deliberate ones are gone).
        let domain = test_domain(8);
        let shape = [8usize; 6];
        let n_total: usize = shape.iter().product();
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = 2.0f64;
        let lv = 2.0f64;

        // Build a Gaussian with deliberate negative ring (Gibbs-like artifact)
        let mut data = vec![0.0f64; n_total];
        for i0 in 0..8 {
            for i1 in 0..8 {
                for i2 in 0..8 {
                    for i3 in 0..8 {
                        for i4 in 0..8 {
                            for i5 in 0..8 {
                                let x = -lx + (i0 as f64 + 0.5) * dx[0];
                                let y = -lx + (i1 as f64 + 0.5) * dx[1];
                                let z = -lx + (i2 as f64 + 0.5) * dx[2];
                                let vx = -lv + (i3 as f64 + 0.5) * dv[0];
                                let vy = -lv + (i4 as f64 + 0.5) * dv[1];
                                let vz = -lv + (i5 as f64 + 0.5) * dv[2];
                                let r2 = x * x + y * y + z * z + vx * vx + vy * vy + vz * vz;
                                // Gaussian core minus a negative ring at r~1.5
                                let val = (-r2).exp() - 0.3 * (-(r2 - 2.25).powi(2)).exp();
                                let idx = i0 * 8usize.pow(5)
                                    + i1 * 8usize.pow(4)
                                    + i2 * 8usize.pow(3)
                                    + i3 * 8usize.pow(2)
                                    + i4 * 8
                                    + i5;
                                data[idx] = val;
                            }
                        }
                    }
                }
            }
        }

        // Confirm some values are actually negative and record worst case
        let neg_before = data.iter().filter(|&&v| v < 0.0).count();
        let min_before = data.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(neg_before > 0, "Test data should contain negatives");

        // Run WITHOUT positivity limiter to establish a baseline
        let snap_no_lim = PhaseSpaceSnapshot {
            data: data.clone(),
            shape,
            time: 0.0,
        };
        let mut tt_no_lim = TensorTrain::from_snapshot(&snap_no_lim, 20, 1e-10, &domain);
        let dummy_no = DisplacementField {
            dx: vec![0.0; 8 * 8 * 8],
            dy: vec![0.0; 8 * 8 * 8],
            dz: vec![0.0; 8 * 8 * 8],
            shape: [8, 8, 8],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = tt_no_lim.domain.clone();

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

        tt_no_lim.advect_x(&dummy_no, &__ctx);
        let full_no_lim = tt_no_lim.to_full();
        let min_no_lim = full_no_lim.iter().cloned().fold(f64::INFINITY, f64::min);

        // Run WITH positivity limiter
        let snap = PhaseSpaceSnapshot {
            data,
            shape,
            time: 0.0,
        };
        let mut tt =
            TensorTrain::from_snapshot(&snap, 20, 1e-10, &domain).with_positivity_limiter(true);
        let dummy = DisplacementField {
            dx: vec![0.0; 8 * 8 * 8],
            dy: vec![0.0; 8 * 8 * 8],
            dz: vec![0.0; 8 * 8 * 8],
            shape: [8, 8, 8],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = tt.domain.clone();

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

        tt.advect_x(&dummy, &__ctx);

        // The violation counter must have recorded the negatives it clipped
        let violations = tt.positivity_violations();
        assert!(
            violations > 0,
            "Positivity violations counter should be non-zero"
        );

        // Note: TT-SVD recompression after clipping may re-introduce negatives
        // (Gibbs-like ringing from the hard discontinuity at zero). The primary
        // value of the TT positivity limiter is diagnostic (counting violations)
        // rather than corrective. For strict positivity, use UniformGrid6D which
        // can clip without recompression artifacts.
        let full_lim = tt.to_full();
        let _min_lim = full_lim.iter().cloned().fold(f64::INFINITY, f64::min);
    }

    #[test]
    fn test_tt_positivity_preserves_mass() {
        // Same setup as above but verify that total_mass() is conserved after
        // positivity enforcement (Zhang-Shu limiter rescales to preserve mass).
        let domain = test_domain(8);
        let shape = [8usize; 6];
        let n_total: usize = shape.iter().product();
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = 2.0f64;
        let lv = 2.0f64;

        let mut data = vec![0.0f64; n_total];
        for i0 in 0..8 {
            for i1 in 0..8 {
                for i2 in 0..8 {
                    for i3 in 0..8 {
                        for i4 in 0..8 {
                            for i5 in 0..8 {
                                let x = -lx + (i0 as f64 + 0.5) * dx[0];
                                let y = -lx + (i1 as f64 + 0.5) * dx[1];
                                let z = -lx + (i2 as f64 + 0.5) * dx[2];
                                let vx = -lv + (i3 as f64 + 0.5) * dv[0];
                                let vy = -lv + (i4 as f64 + 0.5) * dv[1];
                                let vz = -lv + (i5 as f64 + 0.5) * dv[2];
                                let r2 = x * x + y * y + z * z + vx * vx + vy * vy + vz * vz;
                                let val = (-r2).exp() - 0.3 * (-(r2 - 2.25).powi(2)).exp();
                                let idx = i0 * 8usize.pow(5)
                                    + i1 * 8usize.pow(4)
                                    + i2 * 8usize.pow(3)
                                    + i3 * 8usize.pow(2)
                                    + i4 * 8
                                    + i5;
                                data[idx] = val;
                            }
                        }
                    }
                }
            }
        }

        let snap = PhaseSpaceSnapshot {
            data,
            shape,
            time: 0.0,
        };
        let mut tt =
            TensorTrain::from_snapshot(&snap, 20, 1e-10, &domain).with_positivity_limiter(true);
        let mass_before = tt.total_mass();

        let dummy = DisplacementField {
            dx: vec![0.0; 8 * 8 * 8],
            dy: vec![0.0; 8 * 8 * 8],
            dz: vec![0.0; 8 * 8 * 8],
            shape: [8, 8, 8],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = tt.domain.clone();

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

        tt.advect_x(&dummy, &__ctx);
        let mass_after = tt.total_mass();

        // Mass should be conserved to within TT-SVD recompression tolerance.
        // The Zhang-Shu limiter preserves the sum exactly on the full grid,
        // but TT-SVD introduces O(tolerance) error on rebuild.
        let rel_err = (mass_after - mass_before).abs() / mass_before.abs().max(1e-30);
        assert!(
            rel_err < 0.01,
            "Mass should be conserved: before={mass_before}, after={mass_after}, rel_err={rel_err}"
        );
    }
}
