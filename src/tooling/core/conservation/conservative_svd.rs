//! Conservative SVD: moment-preserving truncation for low-rank tensors.
//!
//! Standard SVD truncation minimizes ‖f − f_r‖_F but destroys conservation
//! because discarded modes may carry nonzero contributions to the velocity
//! moments ∫ψ(v)f dv for ψ ∈ {1, v, ½|v|²}.
//!
//! The conservative SVD decomposes f = Πf + f_⊥, where:
//! - Πf carries all conserved moments (mass, momentum, energy)
//! - f_⊥ = f - Πf has zero moments by construction
//!
//! Then standard SVD truncation is applied only to f_⊥. Since f_⊥ has
//! zero moments, any truncation error in f_⊥ cannot affect conservation.
//!
//! Reference: Guo & Qiu, arXiv:2207.00518, Section 3.2

use super::kfvs::MacroState;

/// Velocity moment basis functions for 3D.
///
/// The conserved quantities are:
/// - ψ₀(v) = 1              (mass)
/// - ψ₁(v) = v₁             (x-momentum)
/// - ψ₂(v) = v₂             (y-momentum)
/// - ψ₃(v) = v₃             (z-momentum)
/// - ψ₄(v) = ½|v|²          (kinetic energy)
///
/// These form a 5-dimensional subspace of the velocity function space.
pub const N_CONSERVED: usize = 5;

/// Compute the conserved-moment projection of a distribution function.
///
/// Given f(x,v) on a spatial-velocity grid, computes the projection Πf
/// that preserves the specified macroscopic moments (ρ, ρu, e) at each
/// spatial cell.
///
/// # Arguments
/// * `f` - Distribution function values on 6D grid, shape [nx*ny*nz * nv1*nv2*nv3]
/// * `spatial_shape` - [nx, ny, nz]
/// * `velocity_shape` - [nv1, nv2, nv3]
/// * `target_moments` - Target macroscopic state per spatial cell
/// * `dv` - Velocity cell spacings [dv1, dv2, dv3]
/// * `v_min` - Minimum velocity coordinates [v1_min, v2_min, v3_min]
///
/// # Returns
/// The projected distribution Πf, same shape as f, guaranteed to have
/// the exact target moments when integrated over velocity.
pub fn moment_preserving_projection(
    f: &[f64],
    spatial_shape: [usize; 3],
    velocity_shape: [usize; 3],
    target_moments: &[MacroState],
    dv: [f64; 3],
    v_min: [f64; 3],
) -> Vec<f64> {
    let [nx, ny, nz] = spatial_shape;
    let [nv1, nv2, nv3] = velocity_shape;
    let n_spatial = nx * ny * nz;
    let n_vel = nv1 * nv2 * nv3;
    let dv3 = dv[0] * dv[1] * dv[2];

    assert_eq!(f.len(), n_spatial * n_vel);
    assert_eq!(target_moments.len(), n_spatial);

    let mut result = f.to_vec();

    // Pre-compute velocity grid coordinates and basis functions
    let mut psi = vec![[0.0f64; N_CONSERVED]; n_vel]; // basis functions at each v point
    let mut v_coords = vec![[0.0f64; 3]; n_vel];

    for iv1 in 0..nv1 {
        for iv2 in 0..nv2 {
            for iv3 in 0..nv3 {
                let iv = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                let v1 = v_min[0] + (iv1 as f64 + 0.5) * dv[0];
                let v2 = v_min[1] + (iv2 as f64 + 0.5) * dv[1];
                let v3 = v_min[2] + (iv3 as f64 + 0.5) * dv[2];

                v_coords[iv] = [v1, v2, v3];
                psi[iv] = [1.0, v1, v2, v3, 0.5 * (v1 * v1 + v2 * v2 + v3 * v3)];
            }
        }
    }

    // Gram matrix G_{ij} = ∫ ψ_i(v) ψ_j(v) dv³ (same for all spatial cells)
    let mut gram = [[0.0f64; N_CONSERVED]; N_CONSERVED];
    for psi_v in psi.iter() {
        for i in 0..N_CONSERVED {
            for j in 0..N_CONSERVED {
                gram[i][j] += psi_v[i] * psi_v[j] * dv3;
            }
        }
    }

    // Invert Gram matrix (5×5, use direct Gaussian elimination)
    let gram_inv = invert_5x5(&gram);

    // For each spatial cell: adjust f so that moments match target
    for (ix, target) in target_moments.iter().enumerate().take(n_spatial) {
        let f_offset = ix * n_vel;

        // Current moments of f
        let mut current = [0.0f64; N_CONSERVED];
        for iv in 0..n_vel {
            let fval = result[f_offset + iv];
            for m in 0..N_CONSERVED {
                current[m] += fval * psi[iv][m] * dv3;
            }
        }

        // Target moments
        let target_vec = [
            target.density,
            target.momentum[0],
            target.momentum[1],
            target.momentum[2],
            target.energy,
        ];

        // Moment deficit: δ = target - current
        let mut delta = [0.0f64; N_CONSERVED];
        for m in 0..N_CONSERVED {
            delta[m] = target_vec[m] - current[m];
        }

        // Correction coefficients: c = G⁻¹ δ
        let mut coeffs = [0.0f64; N_CONSERVED];
        for i in 0..N_CONSERVED {
            for j in 0..N_CONSERVED {
                coeffs[i] += gram_inv[i][j] * delta[j];
            }
        }

        // Add correction: f_corrected(v) = f(v) + Σ_m c_m ψ_m(v)
        for iv in 0..n_vel {
            let mut correction = 0.0;
            for m in 0..N_CONSERVED {
                correction += coeffs[m] * psi[iv][m];
            }
            result[f_offset + iv] += correction;
        }
    }

    result
}

/// Apply conservative truncation to a distribution function.
///
/// This is the main entry point for the LoMaC conservative SVD:
/// 1. Compute current moments from f
/// 2. Apply standard truncation/compression (done externally)
/// 3. Project the truncated result to restore exact moments
///
/// This function handles step 3: given the truncated f and the
/// original target moments, it returns f_corrected with exact moments.
pub fn conservative_truncation(
    f_truncated: &[f64],
    spatial_shape: [usize; 3],
    velocity_shape: [usize; 3],
    target_moments: &[MacroState],
    dv: [f64; 3],
    v_min: [f64; 3],
) -> Vec<f64> {
    moment_preserving_projection(
        f_truncated,
        spatial_shape,
        velocity_shape,
        target_moments,
        dv,
        v_min,
    )
}

/// Extract macroscopic moments from a distribution function.
pub fn extract_moments(
    f: &[f64],
    spatial_shape: [usize; 3],
    velocity_shape: [usize; 3],
    dv: [f64; 3],
    v_min: [f64; 3],
) -> Vec<MacroState> {
    let [nx, ny, nz] = spatial_shape;
    let [nv1, nv2, nv3] = velocity_shape;
    let n_spatial = nx * ny * nz;
    let n_vel = nv1 * nv2 * nv3;
    let dv3 = dv[0] * dv[1] * dv[2];

    let mut moments = Vec::with_capacity(n_spatial);

    for ix in 0..n_spatial {
        let f_offset = ix * n_vel;
        let mut rho = 0.0;
        let mut mom = [0.0f64; 3];
        let mut energy = 0.0;

        for iv1 in 0..nv1 {
            for iv2 in 0..nv2 {
                for iv3 in 0..nv3 {
                    let iv = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                    let fval = f[f_offset + iv];
                    let v1 = v_min[0] + (iv1 as f64 + 0.5) * dv[0];
                    let v2 = v_min[1] + (iv2 as f64 + 0.5) * dv[1];
                    let v3 = v_min[2] + (iv3 as f64 + 0.5) * dv[2];

                    rho += fval * dv3;
                    mom[0] += fval * v1 * dv3;
                    mom[1] += fval * v2 * dv3;
                    mom[2] += fval * v3 * dv3;
                    energy += 0.5 * fval * (v1 * v1 + v2 * v2 + v3 * v3) * dv3;
                }
            }
        }

        moments.push(MacroState {
            density: rho,
            momentum: mom,
            energy,
        });
    }

    moments
}

/// Invert a 5×5 matrix using Gaussian elimination with partial pivoting.
fn invert_5x5(a: &[[f64; N_CONSERVED]; N_CONSERVED]) -> [[f64; N_CONSERVED]; N_CONSERVED] {
    let n = N_CONSERVED;
    let mut aug = [[0.0f64; 2 * N_CONSERVED]; N_CONSERVED];

    // Build augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot row with largest absolute value in this column
        let (max_row, _) = aug[col..n]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a[col].abs().partial_cmp(&b[col].abs()).unwrap())
            .map(|(i, row)| (i + col, row[col].abs()))
            .unwrap_or((col, 0.0));
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-30 {
            // Singular — return identity (graceful degradation)
            let mut result = [[0.0f64; N_CONSERVED]; N_CONSERVED];
            for (i, row) in result.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            return result;
        }

        // Scale pivot row
        for val in aug[col].iter_mut().take(2 * n) {
            *val /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            // Copy pivot row to avoid borrow conflict
            let pivot_row: [f64; 2 * N_CONSERVED] = aug[col];
            for j in 0..2 * n {
                aug[row][j] -= factor * pivot_row[j];
            }
        }
    }

    // Extract inverse
    let mut inv = [[0.0f64; N_CONSERVED]; N_CONSERVED];
    for (inv_row, aug_row) in inv.iter_mut().zip(aug.iter()) {
        inv_row[..n].copy_from_slice(&aug_row[n..2 * n]);
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conservative_svd_preserves_moments() {
        // Create a simple 2³ spatial × 4³ velocity grid
        let spatial_shape = [2, 2, 2];
        let velocity_shape = [4, 4, 4];
        let n_spatial = 8;
        let n_vel = 64;
        let dv = [1.0; 3];
        let v_min = [-2.0; 3];

        // Create a distribution: Maxwellian-like
        let mut f = vec![0.0; n_spatial * n_vel];
        for ix in 0..n_spatial {
            for iv1 in 0..4 {
                for iv2 in 0..4 {
                    for iv3 in 0..4 {
                        let iv = iv1 * 16 + iv2 * 4 + iv3;
                        let v1 = v_min[0] + (iv1 as f64 + 0.5) * dv[0];
                        let v2 = v_min[1] + (iv2 as f64 + 0.5) * dv[1];
                        let v3 = v_min[2] + (iv3 as f64 + 0.5) * dv[2];
                        let v2_total = v1 * v1 + v2 * v2 + v3 * v3;
                        f[ix * n_vel + iv] = (-v2_total / 2.0).exp();
                    }
                }
            }
        }

        // Extract original moments
        let original_moments = extract_moments(&f, spatial_shape, velocity_shape, dv, v_min);

        // Perturb f asymmetrically (simulating truncation that removes mass)
        let mut f_perturbed = f.clone();
        for i in 0..f_perturbed.len() {
            // Reduce some entries to zero (simulating rank truncation)
            if i % 3 == 0 {
                f_perturbed[i] *= 0.5;
            }
        }

        // Verify perturbation changed moments
        let perturbed_moments =
            extract_moments(&f_perturbed, spatial_shape, velocity_shape, dv, v_min);
        let mass_diff = (perturbed_moments[0].density - original_moments[0].density).abs();
        assert!(
            mass_diff > 1e-10,
            "Perturbation should change moments: diff={mass_diff}"
        );

        // Apply conservative projection to restore original moments
        let f_corrected = conservative_truncation(
            &f_perturbed,
            spatial_shape,
            velocity_shape,
            &original_moments,
            dv,
            v_min,
        );

        // Verify moments are restored
        let corrected_moments =
            extract_moments(&f_corrected, spatial_shape, velocity_shape, dv, v_min);

        for ix in 0..n_spatial {
            let orig = &original_moments[ix];
            let corr = &corrected_moments[ix];

            assert!(
                (corr.density - orig.density).abs() < 1e-12,
                "Cell {ix}: density {:.6e} vs {:.6e}",
                corr.density,
                orig.density
            );
            for d in 0..3 {
                assert!(
                    (corr.momentum[d] - orig.momentum[d]).abs() < 1e-12,
                    "Cell {ix}: momentum[{d}] {:.6e} vs {:.6e}",
                    corr.momentum[d],
                    orig.momentum[d]
                );
            }
            assert!(
                (corr.energy - orig.energy).abs() < 1e-11,
                "Cell {ix}: energy {:.6e} vs {:.6e}",
                corr.energy,
                orig.energy
            );
        }
    }

    #[test]
    fn extract_moments_maxwellian() {
        // For a symmetric Maxwellian centered at v=0, momentum should be ~0
        let spatial_shape = [1, 1, 1];
        let velocity_shape = [8, 8, 8];
        let dv = [0.5; 3];
        let v_min = [-2.0; 3];
        let n_vel = 512;

        let mut f = vec![0.0; n_vel];
        for iv1 in 0..8 {
            for iv2 in 0..8 {
                for iv3 in 0..8 {
                    let iv = iv1 * 64 + iv2 * 8 + iv3;
                    let v1 = v_min[0] + (iv1 as f64 + 0.5) * dv[0];
                    let v2 = v_min[1] + (iv2 as f64 + 0.5) * dv[1];
                    let v3 = v_min[2] + (iv3 as f64 + 0.5) * dv[2];
                    f[iv] = (-0.5 * (v1 * v1 + v2 * v2 + v3 * v3)).exp();
                }
            }
        }

        let moments = extract_moments(&f, spatial_shape, velocity_shape, dv, v_min);
        assert_eq!(moments.len(), 1);

        // Density should be close to (2π)^{3/2} ≈ 15.75 but on truncated grid
        assert!(moments[0].density > 0.0);

        // Momentum should be ~0 by symmetry
        for d in 0..3 {
            assert!(
                moments[0].momentum[d].abs() < 1e-14,
                "Momentum[{d}] = {}, expected ~0",
                moments[0].momentum[d]
            );
        }
    }

    #[test]
    fn gram_inverse_identity() {
        let id = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let inv = invert_5x5(&id);
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i][j] - expected).abs() < 1e-14,
                    "inv[{i}][{j}] = {}, expected {expected}",
                    inv[i][j]
                );
            }
        }
    }
}
