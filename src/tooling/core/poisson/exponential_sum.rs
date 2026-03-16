//! Braess-Hackbusch exponential sum approximation of 1/r.
//!
//! The Coulomb/Newton kernel 1/r is approximated as a sum of Gaussians:
//!
//! ```text
//! 1/r ≈ (2/√π) Σ_{k=1}^{R_G} c_k exp(-α_k r²)
//! ```
//!
//! via discretization of the Gaussian integral representation
//! `1/r = (2/√π) ∫₀^∞ exp(-t² r²) dt` using the substitution `t = exp(u)`
//! and the trapezoidal rule.
//!
//! Reference: Braess & Hackbusch, "Approximation of 1/x by exponential sums
//! in [1,∞)", IMA J. Numer. Anal. 25(4), 685–697, 2005.

/// Exponential sum coefficients for 1/r ≈ Σ c_k exp(-α_k r²).
pub struct ExponentialSumCoefficients {
    /// Weights c_k (include the 2/√π prefactor and step h).
    pub c: Vec<f64>,
    /// Exponents α_k = exp(2 u_k).
    pub alpha: Vec<f64>,
    /// Number of terms R_G.
    pub r_g: usize,
}

impl ExponentialSumCoefficients {
    /// Compute exponential sum coefficients for approximating 1/r
    /// on [delta, r_max] to accuracy epsilon.
    ///
    /// Uses the Braess-Hackbusch formula: substitute t = exp(u) in the
    /// Gaussian integral 1/r = (2/√π) ∫ exp(-t²r²) dt, then apply
    /// the trapezoidal rule with optimized step h.
    pub fn compute(delta: f64, r_max: f64, epsilon: f64) -> Self {
        assert!(delta > 0.0, "delta must be positive");
        assert!(r_max > delta, "r_max must exceed delta");
        assert!(epsilon > 0.0 && epsilon < 1.0, "epsilon must be in (0,1)");

        // The BH formula gives R_G = O(log(1/ε) · log(R/δ)) terms.
        // Optimal step h ≈ π / (2 · √(log(1/ε) · log(R/δ))) balances the
        // trapezoidal discretization error against the support width.
        let ln_r = (r_max / delta).ln().max(1.0);
        let ln_eps = (1.0 / epsilon).ln().max(1.0);

        let h = std::f64::consts::PI / (2.0 * (ln_eps * ln_r).sqrt());

        // Cutoffs in u-space (unnormalized, i.e., for r in [delta, r_max]):
        // Upper: exp(-exp(2u_max) * delta²) < machine_eps
        //   → exp(2u_max) > 36 / delta²  → u_max = 0.5 * ln(36 / delta²)
        let u_max = 0.5 * (36.0 / (delta * delta)).ln();

        // Lower: for u → -∞, exp(2u) → 0 so the Gaussian ≈ 1, and the
        // integrand ≈ exp(u). The tail integral from -∞ to u_min is ≈ exp(u_min).
        // The full integral is √π/(2r). So relative error at r_max is
        // 2·r_max·exp(u_min)/√π.  For this < ε:
        //   u_min < ln(ε·√π / (2·r_max))
        let u_min = (epsilon * std::f64::consts::PI.sqrt() / (2.0 * r_max)).ln();

        let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;

        let n_neg = ((0.0 - u_min) / h).ceil() as isize;
        let n_pos = (u_max / h).ceil() as isize;

        let mut c = Vec::with_capacity((n_neg + n_pos + 1) as usize);
        let mut alpha = Vec::with_capacity((n_neg + n_pos + 1) as usize);

        for k in -n_neg..=n_pos {
            let u = k as f64 * h;
            let exp_u = u.exp();
            let alpha_k = exp_u * exp_u; // exp(2u)
            let c_k = two_over_sqrt_pi * h * exp_u;

            // Skip negligible weights
            if c_k < 1e-20 {
                continue;
            }
            // Skip terms where the Gaussian is dead at r = delta
            if (-alpha_k * delta * delta).exp() < 1e-18 {
                continue;
            }

            c.push(c_k);
            alpha.push(alpha_k);
        }

        let r_g = c.len();
        Self { c, alpha, r_g }
    }

    /// Evaluate the exponential sum approximation at distance r.
    pub fn evaluate(&self, r: f64) -> f64 {
        let r2 = r * r;
        let mut sum = 0.0;
        for k in 0..self.r_g {
            sum += self.c[k] * (-self.alpha[k] * r2).exp();
        }
        sum
    }

    /// Absolute error |approx(r) - 1/r|.
    pub fn error_at(&self, r: f64) -> f64 {
        (self.evaluate(r) - 1.0 / r).abs()
    }

    /// Relative error |approx(r)*r - 1|.
    pub fn relative_error_at(&self, r: f64) -> f64 {
        (self.evaluate(r) * r - 1.0).abs()
    }

    /// Maximum relative error on [delta, r_max] sampled at n_test log-spaced points.
    pub fn max_relative_error(&self, delta: f64, r_max: f64, n_test: usize) -> f64 {
        let mut max_err = 0.0f64;
        for i in 0..n_test {
            let t = i as f64 / (n_test - 1) as f64;
            let r = delta * (r_max / delta).powf(t);
            max_err = max_err.max(self.relative_error_at(r));
        }
        max_err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_sum_accuracy_1e4() {
        let delta = 0.1;
        let r_max = 10.0;
        let epsilon = 1e-4;

        let coeffs = ExponentialSumCoefficients::compute(delta, r_max, epsilon);

        // R_G should scale as O(log(1/ε) * log(R/δ))
        // For ε=1e-4, R/δ=100: expect ~30-150 terms
        assert!(coeffs.r_g > 5, "Too few terms: {}", coeffs.r_g);

        let max_err = coeffs.max_relative_error(delta, r_max, 500);
        assert!(
            max_err < epsilon,
            "Max relative error {max_err} exceeds tolerance {epsilon}, R_G={}",
            coeffs.r_g
        );
    }

    #[test]
    fn exp_sum_accuracy_1e8() {
        let delta = 0.05;
        let r_max = 20.0;
        let epsilon = 1e-8;

        let coeffs = ExponentialSumCoefficients::compute(delta, r_max, epsilon);

        let max_err = coeffs.max_relative_error(delta, r_max, 500);
        assert!(
            max_err < epsilon,
            "Max relative error {max_err} exceeds tolerance {epsilon}, R_G={}",
            coeffs.r_g
        );
    }

    #[test]
    fn exp_sum_evaluate_at_one() {
        let coeffs = ExponentialSumCoefficients::compute(0.1, 10.0, 1e-6);
        let val = coeffs.evaluate(1.0);
        assert!(
            (val - 1.0).abs() < 1e-5,
            "evaluate(1.0) = {val}, expected ~1.0"
        );
    }

    #[test]
    fn exp_sum_scaling() {
        // Smaller step h (tighter tolerance) should give more terms
        let c1 = ExponentialSumCoefficients::compute(0.1, 10.0, 1e-4);
        let c2 = ExponentialSumCoefficients::compute(0.1, 10.0, 1e-8);

        assert!(
            c2.r_g > c1.r_g,
            "Higher accuracy should require more terms: {} vs {}",
            c2.r_g,
            c1.r_g
        );
    }
}
