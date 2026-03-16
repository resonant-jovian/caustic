//! Green's function for the Poisson equation in HT3D tensor format.
//!
//! Builds the Newton kernel G(x) = -1/(4π|x|) as a low-rank HT3D tensor
//! using the Braess-Hackbusch exponential sum decomposition:
//!
//! ```text
//! 1/|x| ≈ Σ_{k=1}^{R_G} c_k exp(-α_k |x|²)
//!        = Σ_k c_k exp(-α_k x₁²) · exp(-α_k x₂²) · exp(-α_k x₃²)
//! ```
//!
//! Each term is a rank-1 tensor (product of 1D Gaussians), so the full
//! Green's function has HT rank ≤ R_G. With circulant embedding on a
//! (2N)³ grid, this gives isolated (vacuum) boundary conditions.

use super::exponential_sum::ExponentialSumCoefficients;
use crate::tooling::core::algos::ht3d::HtTensor3D;

/// Build the Newton Green's function in HT3D format on a zero-padded (2N)³ grid.
///
/// G(x) = -1/(4π) * 1/|x| with minimum-image wrapping for circulant convolution.
pub fn build_green_ht(
    shape: [usize; 3],
    dx: [f64; 3],
    exp_sum: &ExponentialSumCoefficients,
) -> HtTensor3D {
    let padded_shape = [shape[0] * 2, shape[1] * 2, shape[2] * 2];
    let prefactor = -1.0 / (4.0 * std::f64::consts::PI);

    // Build each term as a rank-1 HT3D and sum them
    let mut result: Option<HtTensor3D> = None;

    for k in 0..exp_sum.r_g {
        let c_k = exp_sum.c[k];
        let alpha_k = exp_sum.alpha[k];

        // 1D Gaussian vectors with minimum-image wrapping
        let v0 = build_1d_gaussian(padded_shape[0], dx[0], alpha_k);
        let v1 = build_1d_gaussian(padded_shape[1], dx[1], alpha_k);
        let v2 = build_1d_gaussian(padded_shape[2], dx[2], alpha_k);

        // Scale: G_k = prefactor * c_k * exp(-α_k x0²) * exp(-α_k x1²) * exp(-α_k x2²)
        let scale = prefactor * c_k;
        let v0_scaled: Vec<f64> = v0.iter().map(|&x| x * scale).collect();

        let term = HtTensor3D::from_rank1(&v0_scaled, &v1, &v2, dx);

        result = Some(match result {
            None => term,
            Some(acc) => acc.add(&term),
        });
    }

    result.unwrap_or_else(|| HtTensor3D::zero(padded_shape, dx))
}

/// Build 1D Gaussian vector exp(-α * x²) with minimum-image wrapping.
fn build_1d_gaussian(n_padded: usize, dx: f64, alpha: f64) -> Vec<f64> {
    let mut v = vec![0.0; n_padded];
    let half = n_padded / 2;
    for (i, val) in v.iter_mut().enumerate() {
        let dist = if i <= half {
            i as f64 * dx
        } else {
            (n_padded - i) as f64 * dx
        };
        *val = (-alpha * dist * dist).exp();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn green_ht_construction() {
        let shape = [8, 8, 8];
        let dx = [0.5, 0.5, 0.5];
        let exp_sum = ExponentialSumCoefficients::compute(0.5, 8.0, 1e-4);

        let green = build_green_ht(shape, dx, &exp_sum);

        assert_eq!(green.shape, [16, 16, 16]);

        // At origin: G(0) diverges but exp sum gives finite value
        let g0 = green.evaluate([0, 0, 0]);
        assert!(
            g0.is_finite(),
            "Green's function at origin should be finite"
        );

        // At (1,0,0): G ≈ -1/(4π * 0.5) ≈ -0.159
        let g1 = green.evaluate([1, 0, 0]);
        let expected = -1.0 / (4.0 * std::f64::consts::PI * dx[0]);
        let rel_err = ((g1 - expected) / expected).abs();
        assert!(
            rel_err < 0.5,
            "Green at (1,0,0): {g1}, expected ~{expected}, rel_err={rel_err}"
        );
    }

    #[test]
    fn green_ht_vs_direct() {
        // Compare HT3D Green's function against direct computation
        let shape = [4, 4, 4];
        let dx = [1.0, 1.0, 1.0];
        let exp_sum = ExponentialSumCoefficients::compute(1.0, 8.0, 1e-4);

        let green_ht = build_green_ht(shape, dx, &exp_sum);
        let prefactor = -1.0 / (4.0 * std::f64::consts::PI);

        // Compare at a few non-origin points
        for &(i0, i1, i2) in &[(1, 0, 0), (1, 1, 0), (2, 1, 1)] {
            let ht_val = green_ht.evaluate([i0, i1, i2]);

            // Direct computation
            let d0 = i0 as f64 * dx[0];
            let d1 = i1 as f64 * dx[1];
            let d2 = i2 as f64 * dx[2];
            let r = (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
            let expected = prefactor / r;

            // HT3D is a sum of rank-1 tensors; slight loss from rank accumulation
            let rel_err = ((ht_val - expected) / expected).abs();
            assert!(
                rel_err < 0.5,
                "Green HT at ({i0},{i1},{i2}): {ht_val}, expected {expected}, rel_err={rel_err}"
            );
        }
    }
}
