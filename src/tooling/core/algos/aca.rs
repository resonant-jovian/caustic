#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! Adaptive Cross Approximation (ACA) for low-rank matrix construction.
//!
//! Implements partially-pivoted ACA (Bebendorf 2000) which builds a rank-k
//! approximation A ≈ U·Vᵀ by sampling only O((m+n)k) entries instead of O(mn).
//! Used as a building block for black-box HT tensor construction (HTACA).

use faer::Mat;
use rayon::prelude::*;

// ─── Black-box matrix trait ─────────────────────────────────────────────────

/// A matrix accessible only through row/column queries.
pub trait BlackBoxMatrix {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    /// Return the full row `i` as a vector of length `ncols()`.
    fn row(&self, i: usize) -> Vec<f64>;
    /// Return the full column `j` as a vector of length `nrows()`.
    fn col(&self, j: usize) -> Vec<f64>;
    /// Return a single entry. Default impl queries the full row.
    fn entry(&self, i: usize, j: usize) -> f64 {
        self.row(i)[j]
    }
}

/// A `BlackBoxMatrix` backed by a closure `F(i, j) -> f64`.
pub struct FnMatrix<F: Fn(usize, usize) -> f64> {
    m: usize,
    n: usize,
    f: F,
}

impl<F: Fn(usize, usize) -> f64> FnMatrix<F> {
    pub fn new(m: usize, n: usize, f: F) -> Self {
        Self { m, n, f }
    }
}

impl<F: Fn(usize, usize) -> f64> BlackBoxMatrix for FnMatrix<F> {
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn row(&self, i: usize) -> Vec<f64> {
        (0..self.n).map(|j| (self.f)(i, j)).collect()
    }
    fn col(&self, j: usize) -> Vec<f64> {
        (0..self.m).map(|i| (self.f)(i, j)).collect()
    }
    fn entry(&self, i: usize, j: usize) -> f64 {
        (self.f)(i, j)
    }
}

/// A `BlackBoxMatrix` wrapping a dense `faer::Mat<f64>`.
pub struct DenseBlackBox<'a>(pub &'a Mat<f64>);

impl BlackBoxMatrix for DenseBlackBox<'_> {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
    fn row(&self, i: usize) -> Vec<f64> {
        (0..self.0.ncols()).map(|j| self.0[(i, j)]).collect()
    }
    fn col(&self, j: usize) -> Vec<f64> {
        (0..self.0.nrows()).map(|i| self.0[(i, j)]).collect()
    }
    fn entry(&self, i: usize, j: usize) -> f64 {
        self.0[(i, j)]
    }
}

// ─── ACA result ─────────────────────────────────────────────────────────────

/// Result of an ACA decomposition: A ≈ Σ uₖ vₖᵀ.
pub struct AcaResult {
    pub row_pivots: Vec<usize>,
    pub col_pivots: Vec<usize>,
    pub u_vectors: Vec<Vec<f64>>,
    pub v_vectors: Vec<Vec<f64>>,
    pub rank: usize,
}

impl AcaResult {
    /// Evaluate the approximation at entry (i, j).
    pub fn evaluate(&self, i: usize, j: usize) -> f64 {
        let mut val = 0.0;
        for k in 0..self.rank {
            val += self.u_vectors[k][i] * self.v_vectors[k][j];
        }
        val
    }

    /// Convert to dense factor matrices (U, V) such that A ≈ U · Vᵀ.
    pub fn to_uv(&self) -> (Mat<f64>, Mat<f64>) {
        if self.rank == 0 {
            return (Mat::zeros(0, 0), Mat::zeros(0, 0));
        }
        let m = self.u_vectors[0].len();
        let n = self.v_vectors[0].len();
        let mut u = Mat::zeros(m, self.rank);
        let mut v = Mat::zeros(n, self.rank);
        for k in 0..self.rank {
            for i in 0..m {
                u[(i, k)] = self.u_vectors[k][i];
            }
            for j in 0..n {
                v[(j, k)] = self.v_vectors[k][j];
            }
        }
        (u, v)
    }

    /// Compute (max_error, frobenius_error) against a dense reference matrix.
    ///
    /// Parallelized over rows with rayon for large matrices.
    pub fn error_vs_dense(&self, dense: &Mat<f64>) -> (f64, f64) {
        let m = dense.nrows();
        let n = dense.ncols();
        let (max_err, frob_sq) = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut row_max = 0.0f64;
                let mut row_frob_sq = 0.0f64;
                for j in 0..n {
                    let e = (dense[(i, j)] - self.evaluate(i, j)).abs();
                    row_max = row_max.max(e);
                    row_frob_sq += e * e;
                }
                (row_max, row_frob_sq)
            })
            .reduce(
                || (0.0f64, 0.0f64),
                |(m1, f1), (m2, f2)| (m1.max(m2), f1 + f2),
            );
        (max_err, frob_sq.sqrt())
    }
}

// ─── Xorshift64 RNG ────────────────────────────────────────────────────────

/// Minimal xorshift64 PRNG (avoids `rand` dependency).
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x5A5A_5A5A_5A5A_5A5A
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % (bound as u64)) as usize
    }

    /// Generate a random multi-index where each element is in [0, shape[d]).
    pub fn random_multi_index(&mut self, shape: &[usize]) -> Vec<usize> {
        shape.iter().map(|&s| self.next_usize(s)).collect()
    }
}

// ─── Partially-pivoted ACA ─────────────────────────────────────────────────

/// Partially-pivoted ACA (Bebendorf 2000).
///
/// Builds a rank-k approximation A ≈ Σ uₖ vₖᵀ by querying only O((m+n)k) entries.
/// Stops when ‖uₖ‖·‖vₖ‖ / ‖Aₖ‖_F < tolerance, or when rank reaches max_rank.
pub fn aca_partial_pivot(mat: &dyn BlackBoxMatrix, tolerance: f64, max_rank: usize) -> AcaResult {
    let m = mat.nrows();
    let n = mat.ncols();

    if m == 0 || n == 0 {
        return AcaResult {
            row_pivots: vec![],
            col_pivots: vec![],
            u_vectors: vec![],
            v_vectors: vec![],
            rank: 0,
        };
    }

    let max_k = max_rank.min(m).min(n);

    let mut row_pivots: Vec<usize> = Vec::with_capacity(max_k);
    let mut col_pivots: Vec<usize> = Vec::with_capacity(max_k);
    let mut u_vecs: Vec<Vec<f64>> = Vec::with_capacity(max_k);
    let mut v_vecs: Vec<Vec<f64>> = Vec::with_capacity(max_k);

    let mut used_rows = vec![false; m];
    let mut used_cols = vec![false; n];

    // Frobenius norm estimate: ‖Aₖ‖²_F = Σ (‖uₖ‖·‖vₖ‖)² + 2·Σ cross terms
    // We track it incrementally.
    let mut frob_sq = 0.0f64;

    // Start with pivot row i₀ = 0
    let mut next_row = 0usize;

    for _step in 0..max_k {
        let ik = next_row;
        used_rows[ik] = true;
        row_pivots.push(ik);

        // Compute residual row: r = A(iₖ, :) - Σⱼ<ₖ uⱼ[iₖ] · vⱼ
        let mut r = mat.row(ik);
        for prev in 0..u_vecs.len() {
            let coeff = u_vecs[prev][ik];
            for j in 0..n {
                r[j] -= coeff * v_vecs[prev][j];
            }
        }

        // Column pivot: jₖ = argmax |rⱼ| among unused columns
        let mut jk = 0;
        let mut best = -1.0f64;
        for j in 0..n {
            if !used_cols[j] && r[j].abs() > best {
                best = r[j].abs();
                jk = j;
            }
        }

        // If the pivot is essentially zero, the matrix is (numerically) exhausted
        if best < 1e-15 {
            row_pivots.pop();
            break;
        }

        used_cols[jk] = true;
        col_pivots.push(jk);

        // vₖ = r / r[jₖ]
        let scale = 1.0 / r[jk];
        for val in &mut r {
            *val *= scale;
        }
        let vk = r;

        // Compute residual column: c = A(:, jₖ) - Σⱼ<ₖ vⱼ[jₖ] · uⱼ
        let mut c = mat.col(jk);
        for prev in 0..v_vecs.len() {
            let coeff = v_vecs[prev][jk];
            for i in 0..m {
                c[i] -= coeff * u_vecs[prev][i];
            }
        }
        let uk = c;

        // Norms for convergence check
        let uk_norm_sq: f64 = uk.iter().map(|x| x * x).sum();
        let vk_norm_sq: f64 = vk.iter().map(|x| x * x).sum();

        // Update Frobenius norm estimate:
        // ‖A_{k+1}‖²_F = ‖Aₖ‖²_F + ‖uₖ‖²·‖vₖ‖² + 2·Σⱼ<ₖ (uₖᵀuⱼ)(vₖᵀvⱼ)
        let mut cross = 0.0f64;
        for prev in 0..u_vecs.len() {
            let u_dot: f64 = uk.iter().zip(&u_vecs[prev]).map(|(a, b)| a * b).sum();
            let v_dot: f64 = vk.iter().zip(&v_vecs[prev]).map(|(a, b)| a * b).sum();
            cross += u_dot * v_dot;
        }
        frob_sq += uk_norm_sq * vk_norm_sq + 2.0 * cross;

        // Next pivot row: i_{k+1} = argmax |uₖ[i]| among unused rows
        let mut next_best = -1.0f64;
        let mut candidate = 0;
        for i in 0..m {
            if !used_rows[i] && uk[i].abs() > next_best {
                next_best = uk[i].abs();
                candidate = i;
            }
        }
        next_row = candidate;

        u_vecs.push(uk);
        v_vecs.push(vk);

        // Convergence check (squared comparison avoids 2 sqrt calls per iteration)
        if frob_sq > 0.0 && uk_norm_sq * vk_norm_sq < tolerance * tolerance * frob_sq {
            break;
        }
    }

    let rank = u_vecs.len();
    AcaResult {
        row_pivots,
        col_pivots,
        u_vectors: u_vecs,
        v_vectors: v_vecs,
        rank,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aca_rank1_exact() {
        // A[i,j] = (i+1)*(j+1), exact rank 1
        let m = 20;
        let n = 15;
        let mat = FnMatrix::new(m, n, |i, j| ((i + 1) * (j + 1)) as f64);
        let res = aca_partial_pivot(&mat, 1e-12, 10);

        assert!(
            res.rank <= 2,
            "rank-1 matrix should converge quickly, got rank {}",
            res.rank
        );

        let mut max_err = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let expected = ((i + 1) * (j + 1)) as f64;
                max_err = max_err.max((res.evaluate(i, j) - expected).abs());
            }
        }
        assert!(max_err < 1e-10, "rank-1 max error {max_err:.2e}");
    }

    #[test]
    fn aca_rank3() {
        // A = sum of 3 outer products
        let m = 30;
        let n = 25;
        let mat = FnMatrix::new(m, n, |i, j| {
            let fi = i as f64;
            let fj = j as f64;
            fi * fj + (fi * fi) * (fj + 1.0) + (fi + 2.0) * (fj * fj)
        });
        let res = aca_partial_pivot(&mat, 1e-10, 20);

        assert!(
            res.rank <= 5,
            "rank-3 matrix should converge quickly, got rank {}",
            res.rank
        );

        // Check accuracy against dense
        let mut dense = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                dense[(i, j)] = mat.entry(i, j);
            }
        }
        let (max_err, _frob_err) = res.error_vs_dense(&dense);
        assert!(max_err < 1e-6, "rank-3 max error {max_err:.2e}");
    }

    #[test]
    fn aca_low_rank_plus_noise() {
        // Low-rank signal with small noise (simulated via high-frequency component)
        let m = 40;
        let n = 30;
        let mat = FnMatrix::new(m, n, |i, j| {
            let signal = (i as f64) * (j as f64) + ((i + 1) as f64).sqrt() * ((j + 1) as f64);
            let noise = 1e-8 * ((i * 7 + j * 13) % 100) as f64;
            signal + noise
        });
        let res = aca_partial_pivot(&mat, 1e-6, 20);

        // Should capture the rank-2 signal with few terms
        assert!(
            res.rank <= 6,
            "low-rank+noise should converge quickly, got rank {}",
            res.rank
        );
    }

    #[test]
    fn aca_gaussian_kernel() {
        // Gaussian kernel: exp(-|xᵢ - yⱼ|² / σ²)
        let m = 50;
        let n = 40;
        let sigma = 0.5;
        let xs: Vec<f64> = (0..m).map(|i| i as f64 / m as f64).collect();
        let ys: Vec<f64> = (0..n).map(|j| j as f64 / n as f64).collect();
        let mat = FnMatrix::new(m, n, |i, j| {
            let d = xs[i] - ys[j];
            (-d * d / (sigma * sigma)).exp()
        });
        let res = aca_partial_pivot(&mat, 1e-8, 30);

        let mut dense = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                dense[(i, j)] = mat.entry(i, j);
            }
        }
        let (max_err, _) = res.error_vs_dense(&dense);
        assert!(
            max_err < 1e-6,
            "Gaussian kernel max error {max_err:.2e} too large"
        );
    }

    #[test]
    fn cur_output() {
        // Verify U·Vᵀ from to_uv() reproduces the approximation
        let m = 15;
        let n = 12;
        let mat = FnMatrix::new(m, n, |i, j| ((i as f64) + 1.0) * ((j as f64) + 0.5).sin());
        let res = aca_partial_pivot(&mat, 1e-8, 10);
        let (u, v) = res.to_uv();

        assert_eq!(u.nrows(), m);
        assert_eq!(u.ncols(), res.rank);
        assert_eq!(v.nrows(), n);
        assert_eq!(v.ncols(), res.rank);

        // U·Vᵀ should match evaluate()
        let mut max_err = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let uv: f64 = (0..res.rank).map(|k| u[(i, k)] * v[(j, k)]).sum();
                max_err = max_err.max((uv - res.evaluate(i, j)).abs());
            }
        }
        assert!(max_err < 1e-14, "to_uv consistency error {max_err:.2e}");
    }

    #[test]
    fn convergence_criterion() {
        // Check that the Frobenius norm estimate tracks correctly
        let m = 30;
        let n = 25;
        let mat = FnMatrix::new(m, n, |i, j| {
            let fi = i as f64 / m as f64;
            let fj = j as f64 / n as f64;
            (-((fi - 0.3).powi(2) + (fj - 0.5).powi(2)) / 0.1).exp()
        });

        // Tight tolerance should give more ranks
        let res_tight = aca_partial_pivot(&mat, 1e-10, 20);
        // Loose tolerance should give fewer ranks
        let res_loose = aca_partial_pivot(&mat, 1e-3, 20);

        assert!(
            res_loose.rank <= res_tight.rank,
            "loose tol rank {} should be ≤ tight tol rank {}",
            res_loose.rank,
            res_tight.rank
        );

        // Tight should be more accurate
        let mut dense = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                dense[(i, j)] = mat.entry(i, j);
            }
        }
        let (_, frob_tight) = res_tight.error_vs_dense(&dense);
        let (_, frob_loose) = res_loose.error_vs_dense(&dense);
        assert!(
            frob_tight <= frob_loose + 1e-15,
            "tight error {frob_tight:.2e} should be ≤ loose {frob_loose:.2e}"
        );
    }

    #[test]
    fn zero_matrix() {
        let m = 10;
        let n = 8;
        let mat = FnMatrix::new(m, n, |_, _| 0.0);
        let res = aca_partial_pivot(&mat, 1e-10, 10);

        assert_eq!(res.rank, 0, "zero matrix should have rank 0");
        for i in 0..m {
            for j in 0..n {
                assert_eq!(res.evaluate(i, j), 0.0);
            }
        }
    }
}
