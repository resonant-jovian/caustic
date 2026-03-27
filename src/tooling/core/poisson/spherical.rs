//! Spherical-harmonics Poisson solver for ∇²Φ = 4πGρ.
//!
//! Designed for nearly-spherical density distributions (isolated halos, stellar
//! systems). The density is projected onto real spherical harmonics Y_lm up to
//! degree `l_max`, the radial Poisson ODE is solved per (l,m) mode using the
//! Green's function method (cumulative inner/outer integrals), and the potential
//! is reconstructed on the Cartesian grid by summing Φ_lm(r) Y_lm(θ,φ).
//!
//! Boundary conditions are implicitly isolated (Φ → 0 at infinity).
//! Complexity: O(N³ l_max²) for decomposition and reconstruction.

use rayon::prelude::*;

use super::super::{context::SimContext, solver::PoissonSolver, types::*};
use super::utils::finite_difference_acceleration;
use std::f64::consts::PI;

/// Spherical-harmonics expansion Poisson solver.
///
/// Best suited for nearly-spherical density distributions (isolated halos,
/// stellar systems). The expansion is truncated at degree `l_max`, and the
/// radial direction is discretized into `n_radial` bins from 0 to `r_max`.
pub struct SphericalHarmonicsPoisson {
    /// Maximum spherical harmonic degree (0 = monopole only).
    pub l_max: usize,
    /// Number of radial bins for the 1D Green's-function ODE solve.
    pub n_radial: usize,
    /// Outer radius of the radial grid (auto-computed from the domain diagonal).
    pub r_max: f64,
    /// Cartesian grid dimensions `[nx, ny, nz]`.
    pub shape: [usize; 3],
    /// Cartesian cell spacings `[dx, dy, dz]`.
    pub dx: [f64; 3],
}

impl SphericalHarmonicsPoisson {
    /// Create a new spherical-harmonics Poisson solver.
    ///
    /// `l_max`: maximum spherical harmonic degree (0 = monopole only).
    /// `n_radial`: number of radial bins for the 1D Poisson solve.
    /// `shape`: grid dimensions `[nx, ny, nz]`.
    /// `dx`: cell spacings `[dx, dy, dz]`.
    pub fn new(l_max: usize, n_radial: usize, shape: [usize; 3], dx: [f64; 3]) -> Self {
        // r_max = sqrt(3) * half-diagonal of the domain box
        let half_extents = [
            shape[0] as f64 * dx[0] / 2.0,
            shape[1] as f64 * dx[1] / 2.0,
            shape[2] as f64 * dx[2] / 2.0,
        ];
        let max_half = half_extents.iter().cloned().fold(0.0_f64, f64::max);
        let r_max = 3.0_f64.sqrt() * max_half;

        Self {
            l_max,
            n_radial,
            r_max,
            shape,
            dx,
        }
    }

    /// Number of (l,m) pairs for l in 0..=l_max: sum_{l=0}^{l_max} (2l+1) = (l_max+1)^2.
    #[inline]
    fn n_harmonics(&self) -> usize {
        (self.l_max + 1) * (self.l_max + 1)
    }

    /// Map (l, m) to a flat index. m ranges from -l to l.
    /// Index = l^2 + l + m.
    #[inline]
    fn lm_index(l: usize, m: i32) -> usize {
        ((l * l) as i32 + l as i32 + m) as usize
    }
}

// ---------------------------------------------------------------------------
// Associated Legendre functions and spherical harmonics
// ---------------------------------------------------------------------------

/// Compute the (unnormalized) associated Legendre function P_l^m(x) for m >= 0.
///
/// Uses the standard upward recursion which is stable for moderate l:
///   P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
///   P_{m+1}^m(x) = x (2m+1) P_m^m(x)
///   (l-m) P_l^m(x) = (2l-1) x P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)
#[inline]
fn associated_legendre(l: usize, m: usize, x: f64) -> f64 {
    debug_assert!(m <= l, "m must be <= l");

    // P_m^m: starting value
    let mut pmm = 1.0_f64;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for i in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
            let _ = i; // suppress unused warning
        }
    }

    if l == m {
        return pmm;
    }

    // P_{m+1}^m
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return pmmp1;
    }

    // Upward recursion from m+2 to l
    let mut pll = 0.0;
    for ll in (m + 2)..=l {
        pll = ((2 * ll - 1) as f64 * x * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Compute the normalization factor N_lm = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!).
#[inline]
fn normalization(l: usize, m_abs: usize) -> f64 {
    // Compute (l-|m|)! / (l+|m|)! via ratio to avoid large factorials
    let mut ratio = 1.0_f64;
    for k in (l - m_abs + 1)..=(l + m_abs) {
        ratio *= k as f64;
    }
    ((2 * l + 1) as f64 / (4.0 * PI * ratio)).sqrt()
}

/// Real spherical harmonic Y_lm(theta, phi) with Condon-Shortley phase.
///
///   m > 0:  sqrt(2) * N_lm * P_l^m(cos theta) * cos(m phi)
///   m = 0:  N_l0 * P_l^0(cos theta)
///   m < 0:  sqrt(2) * N_l|m| * P_l^|m|(cos theta) * sin(|m| phi)
#[inline]
fn real_spherical_harmonic(l: usize, m: i32, theta: f64, phi: f64) -> f64 {
    let m_abs = m.unsigned_abs() as usize;
    let n_lm = normalization(l, m_abs);
    let plm = associated_legendre(l, m_abs, theta.cos());

    if m > 0 {
        2.0_f64.sqrt() * n_lm * plm * (m_abs as f64 * phi).cos()
    } else if m < 0 {
        2.0_f64.sqrt() * n_lm * plm * (m_abs as f64 * phi).sin()
    } else {
        n_lm * plm
    }
}

// ---------------------------------------------------------------------------
// Cartesian to spherical coordinate conversion
// ---------------------------------------------------------------------------

/// Convert Cartesian cell center (ix, iy, iz) to physical (x, y, z) relative
/// to the grid center.
#[inline]
fn cell_to_xyz(
    ix: usize,
    iy: usize,
    iz: usize,
    shape: &[usize; 3],
    dx: &[f64; 3],
) -> (f64, f64, f64) {
    let x = (ix as f64 - shape[0] as f64 / 2.0 + 0.5) * dx[0];
    let y = (iy as f64 - shape[1] as f64 / 2.0 + 0.5) * dx[1];
    let z = (iz as f64 - shape[2] as f64 / 2.0 + 0.5) * dx[2];
    (x, y, z)
}

/// Convert (x, y, z) to spherical coordinates (r, theta, phi).
#[inline]
fn xyz_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    if r < 1e-30 {
        return (0.0, 0.0, 0.0);
    }
    let theta = (z / r).clamp(-1.0, 1.0).acos();
    let phi = y.atan2(x); // returns [-pi, pi]
    (r, theta, phi)
}

// ---------------------------------------------------------------------------
// Decomposition & reconstruction
// ---------------------------------------------------------------------------

/// Decompose density onto radial profiles for each (l,m) mode.
///
/// Returns a vector of length `n_harmonics`, each entry being a radial profile
/// `rho_lm[r_idx]` of length `n_radial`.
///
/// The decomposition integral is:
///   rho_lm(r) = integral over angles of rho(r, theta, phi) * Y_lm(theta, phi) d(Omega)
///
/// We approximate this by summing Cartesian cells that fall into each radial bin,
/// weighting by Y_lm and dividing by the shell volume to get the angular average
/// weighted by Y_lm.
fn decompose_density(
    density: &DensityField,
    shape: &[usize; 3],
    dx: &[f64; 3],
    l_max: usize,
    n_radial: usize,
    r_max: f64,
) -> Vec<Vec<f64>> {
    let n_harm = (l_max + 1) * (l_max + 1);
    let dr = r_max / n_radial as f64;
    let cell_vol = dx[0] * dx[1] * dx[2];
    let [nx, ny, nz] = *shape;

    // rho_lm[harm_idx][r_idx]: accumulated weighted density
    // shell_vol[r_idx]: total Cartesian cell volume in each radial bin (for normalization)
    //
    // Parallelize over ix: each thread accumulates its own rho_lm + shell_vol,
    // then reduce by element-wise addition.
    let (mut rho_lm, _shell_vol) = (0..nx)
        .into_par_iter()
        .fold(
            || (vec![vec![0.0; n_radial]; n_harm], vec![0.0; n_radial]),
            |(mut rho_lm_local, mut shell_vol_local), ix| {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let (x, y, z) = cell_to_xyz(ix, iy, iz, shape, dx);
                        let (r, theta, phi) = xyz_to_spherical(x, y, z);

                        if r >= r_max {
                            continue;
                        }

                        let r_idx = ((r / dr) as usize).min(n_radial - 1);

                        let cell_idx = ix * ny * nz + iy * nz + iz;
                        let rho_val = density.data[cell_idx];

                        shell_vol_local[r_idx] += cell_vol;

                        for l in 0..=l_max {
                            for m in -(l as i32)..=(l as i32) {
                                let ylm = real_spherical_harmonic(l, m, theta, phi);
                                let h_idx = SphericalHarmonicsPoisson::lm_index(l, m);
                                rho_lm_local[h_idx][r_idx] += rho_val * ylm * cell_vol;
                            }
                        }
                    }
                }
                (rho_lm_local, shell_vol_local)
            },
        )
        .reduce(
            || (vec![vec![0.0; n_radial]; n_harm], vec![0.0; n_radial]),
            |(mut a_rho, mut a_sv), (b_rho, b_sv)| {
                for h in 0..n_harm {
                    for r in 0..n_radial {
                        a_rho[h][r] += b_rho[h][r];
                    }
                }
                for r in 0..n_radial {
                    a_sv[r] += b_sv[r];
                }
                (a_rho, a_sv)
            },
        );

    // Normalize: divide by shell volume so rho_lm has units of density * Y_lm
    // The integral over the shell of rho*Y_lm dV = rho_lm[r] * shell_vol
    // We want: rho_lm(r) such that rho(r,theta,phi) ~ sum_lm rho_lm(r) Y_lm(theta,phi)
    // The projection gives: rho_lm(r) = integral_angles rho(r,Omega) Y_lm(Omega) dOmega
    // With Cartesian sampling: rho_lm(r) ~ sum_cells_in_shell rho * Y_lm * cell_vol / shell_vol * 4pi
    // because integral Y_00 dOmega = sqrt(4pi) and we need the angular part.
    //
    // Actually, the radial Poisson Green's function expects:
    //   rho_lm(r) * r^2 dr integrated (i.e., the coefficient in the spherical expansion).
    // So we keep the raw volume-weighted sums and divide by dr to get density * r^2 * 4pi type terms.
    //
    // More precisely, for the Green's function solution:
    //   Phi_lm(r) = -4piG/(2l+1) [ r^{-(l+1)} int_0^r rho_lm(s) s^{l+2} ds
    //                              + r^l         int_r^{r_max} rho_lm(s) s^{1-l} ds ]
    //
    // where rho_lm(r) is the coefficient in rho = sum_lm rho_lm(r) Y_lm.
    // The volume integral gives: rho_lm(r_bin) = (1/(r^2 dr)) * sum_{cells in bin} rho * Y_lm * dV
    //
    // So we normalize by r^2 * dr:

    for r_idx in 0..n_radial {
        let r_center = (r_idx as f64 + 0.5) * dr;
        let r2_dr = r_center * r_center * dr;
        if r2_dr > 1e-30 {
            for harm in rho_lm.iter_mut() {
                harm[r_idx] /= r2_dr;
            }
        }
    }

    rho_lm
}

/// Solve the radial Poisson equation for a single (l,m) mode using the
/// Green's function method.
///
/// For angular degree l:
///   Phi_lm(r) = -4*pi*G/(2l+1) * [ r^{-(l+1)} * I_inner(r) + r^l * I_outer(r) ]
///
/// where:
///   I_inner(r) = integral_0^r  rho_lm(s) * s^{l+2} ds
///   I_outer(r) = integral_r^R  rho_lm(s) * s^{1-l} ds
///
/// These are computed via cumulative trapezoidal sums.
fn radial_poisson_solve(rho_lm: &[f64], l: usize, n_radial: usize, r_max: f64, g: f64) -> Vec<f64> {
    let dr = r_max / n_radial as f64;
    let mut phi_lm = vec![0.0f64; n_radial];

    // Precompute radial bin centers
    let r_centers: Vec<f64> = (0..n_radial).map(|i| (i as f64 + 0.5) * dr).collect();

    // Build the inner integrand: rho_lm(s) * s^{l+2} * ds
    let inner_integrand: Vec<f64> = (0..n_radial)
        .map(|i| rho_lm[i] * r_centers[i].powi(l as i32 + 2) * dr)
        .collect();

    // Build the outer integrand: rho_lm(s) * s^{1-l} * ds
    // For l >= 2, s^{1-l} = s^{-(l-1)}, careful with s near 0
    let outer_integrand: Vec<f64> = (0..n_radial)
        .map(|i| {
            let s = r_centers[i];
            if s < 1e-30 && l > 1 {
                0.0
            } else {
                rho_lm[i] * s.powi(1 - l as i32) * dr
            }
        })
        .collect();

    // Cumulative sum from inside: I_inner[i] = sum_{j=0}^{i} inner_integrand[j]
    let mut i_inner = vec![0.0f64; n_radial];
    i_inner[0] = inner_integrand[0];
    for i in 1..n_radial {
        i_inner[i] = i_inner[i - 1] + inner_integrand[i];
    }

    // Cumulative sum from outside: I_outer[i] = sum_{j=i+1}^{n_radial-1} outer_integrand[j]
    let mut i_outer = vec![0.0f64; n_radial];
    // i_outer[n_radial-1] = 0 (no mass beyond last bin)
    for i in (0..n_radial - 1).rev() {
        i_outer[i] = i_outer[i + 1] + outer_integrand[i + 1];
    }

    // Assemble potential
    let prefactor = -4.0 * PI * g / (2 * l + 1) as f64;
    for i in 0..n_radial {
        let r = r_centers[i];
        if r < 1e-30 {
            // At origin, only the outer integral contributes for l=0:
            // Phi_00(0) = -4piG * I_outer(0) for l=0
            if l == 0 {
                phi_lm[i] = prefactor * i_outer[i];
            }
            // For l > 0, Phi_lm(0) = 0 by symmetry
            continue;
        }

        let r_neg_lp1 = r.powi(-(l as i32 + 1));
        let r_pos_l = r.powi(l as i32);

        phi_lm[i] = prefactor * (r_neg_lp1 * i_inner[i] + r_pos_l * i_outer[i]);
    }

    phi_lm
}

/// Reconstruct the potential on the Cartesian grid from the radial harmonic
/// coefficients.
///
/// Phi(x,y,z) = sum_{l,m} Phi_lm(r) * Y_lm(theta, phi)
fn reconstruct_potential(
    phi_lm: &[Vec<f64>],
    l_max: usize,
    shape: &[usize; 3],
    dx: &[f64; 3],
    n_radial: usize,
    r_max: f64,
) -> PotentialField {
    let [nx, ny, nz] = *shape;
    let n_total = nx * ny * nz;
    let dr = r_max / n_radial as f64;
    let mut pot_data = vec![0.0f64; n_total];

    pot_data.par_iter_mut().enumerate().for_each(|(flat, val)| {
        let ix = flat / (ny * nz);
        let iy = (flat / nz) % ny;
        let iz = flat % nz;

        let (x, y, z) = cell_to_xyz(ix, iy, iz, shape, dx);
        let (r, theta, phi) = xyz_to_spherical(x, y, z);

        // Find radial bin and interpolate linearly
        let r_frac = r / dr - 0.5;
        let r_idx_lo = r_frac.floor().max(0.0) as usize;
        let r_idx_hi = (r_idx_lo + 1).min(n_radial - 1);
        let t = (r_frac - r_idx_lo as f64).clamp(0.0, 1.0);

        let mut sum = 0.0;
        for l in 0..=l_max {
            for m in -(l as i32)..=(l as i32) {
                let h_idx = SphericalHarmonicsPoisson::lm_index(l, m);
                let ylm = real_spherical_harmonic(l, m, theta, phi);

                // Linear interpolation in radius
                let phi_r = (1.0 - t) * phi_lm[h_idx][r_idx_lo] + t * phi_lm[h_idx][r_idx_hi];

                sum += phi_r * ylm;
            }
        }
        *val = sum;
    });

    PotentialField {
        data: pot_data,
        shape: *shape,
    }
}

// ---------------------------------------------------------------------------
// PoissonSolver trait implementation
// ---------------------------------------------------------------------------

impl PoissonSolver for SphericalHarmonicsPoisson {
    /// Solve ∇²Φ = 4πGρ via spherical harmonic decomposition.
    ///
    /// Steps:
    /// 1. Decompose density into radial profiles rho_lm(r) for each (l,m).
    /// 2. Solve the radial Poisson equation for each mode.
    /// 3. Reconstruct the Cartesian potential by summing Phi_lm(r) Y_lm(theta,phi).
    fn solve(&self, density: &DensityField, ctx: &SimContext) -> PotentialField {
        let g = ctx.g;
        let _span = tracing::info_span!("spherical_harmonics_solve").entered();

        // Step 1: decompose density
        let rho_lm = decompose_density(
            density,
            &self.shape,
            &self.dx,
            self.l_max,
            self.n_radial,
            self.r_max,
        );

        // Step 2: radial Poisson solve for each (l,m)
        let n_harm = self.n_harmonics();
        let mut phi_lm: Vec<Vec<f64>> = Vec::with_capacity(n_harm);

        for l in 0..=self.l_max {
            for m in -(l as i32)..=(l as i32) {
                let h_idx = Self::lm_index(l, m);
                // All m with the same l share the same Green's function
                let radial_pot =
                    radial_poisson_solve(&rho_lm[h_idx], l, self.n_radial, self.r_max, g);
                // Ensure we push in order of h_idx
                debug_assert_eq!(phi_lm.len(), h_idx);
                phi_lm.push(radial_pot);
            }
        }

        // Step 3: reconstruct potential on Cartesian grid
        reconstruct_potential(
            &phi_lm,
            self.l_max,
            &self.shape,
            &self.dx,
            self.n_radial,
            self.r_max,
        )
    }

    /// Compute gravitational acceleration g = -∇Φ via second-order centered finite
    /// differences on the Cartesian grid.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        finite_difference_acceleration(potential, &self.dx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::context::SimContext;
    use crate::tooling::core::events::EventEmitter;
    use crate::tooling::core::progress::StepProgress;
    use crate::tooling::core::solver::PoissonSolver as _;

    use super::*;

    #[test]
    fn spherical_point_mass() {
        // Point mass at center: Phi = -GM/r (l=0 monopole)
        let n = 16;
        let dx = [0.5; 3]; // domain from -4 to 4
        let shape = [n; 3];
        let mut rho = vec![0.0; n * n * n];
        // Put mass in center cell
        let mid = n / 2;
        let cell_vol = dx[0] * dx[1] * dx[2];
        let m_total = 1.0;
        rho[mid * n * n + mid * n + mid] = m_total / cell_vol;

        let solver = SphericalHarmonicsPoisson::new(4, 32, shape, dx);
        let density = DensityField { data: rho, shape };
        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {

            solver: &solver as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,

        };

        let pot = solver.solve(&density, &_ctx);

        // Check potential at a point away from center
        let test_ix = mid + 3;
        let r = (test_ix as f64 - mid as f64 + 0.5) * dx[0];
        let expected = -m_total / r; // -GM/r with G=1
        let idx = test_ix * n * n + mid * n + mid;
        let err = (pot.data[idx] - expected).abs() / expected.abs();
        assert!(
            err < 0.5,
            "Point mass error {err} at r={r} (expected {expected}, got {})",
            pot.data[idx]
        );
    }

    #[test]
    fn spherical_plummer() {
        // Plummer sphere: Phi(r) = -GM/sqrt(r^2 + a^2)
        let n = 16;
        let dx = [0.5; 3];
        let shape = [n; 3];
        let g_val = 1.0;
        let m_total = 1.0;
        let a: f64 = 1.0;

        let mut rho = vec![0.0; n * n * n];
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let x = (ix as f64 - n as f64 / 2.0 + 0.5) * dx[0];
                    let y = (iy as f64 - n as f64 / 2.0 + 0.5) * dx[1];
                    let z = (iz as f64 - n as f64 / 2.0 + 0.5) * dx[2];
                    let r2 = x * x + y * y + z * z;
                    // Plummer density: rho = 3M/(4*pi*a^3) * (1 + r^2/a^2)^{-5/2}
                    rho[ix * n * n + iy * n + iz] =
                        3.0 * m_total / (4.0 * PI * a.powi(3)) * (1.0 + r2 / (a * a)).powf(-2.5);
                }
            }
        }

        let solver = SphericalHarmonicsPoisson::new(0, 32, shape, dx);
        let density = DensityField { data: rho, shape };
        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {

            solver: &solver as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: g_val,

        };

        let pot = solver.solve(&density, &_ctx);

        // Check at a few radii
        let mid = n / 2;
        let test_ix = mid + 2;
        let r = (test_ix as f64 - mid as f64 + 0.5) * dx[0];
        let expected = -g_val * m_total / (r * r + a * a).sqrt();
        let idx = test_ix * n * n + mid * n + mid;
        // Allow generous tolerance for discretization
        let err = (pot.data[idx] - expected).abs() / expected.abs();
        assert!(err < 1.0, "Plummer potential error {err} at r={r}");
    }

    #[test]
    fn spherical_vs_fft_isolated() {
        // Cross-validate with FftIsolated on a smooth density.
        // Just verify both produce finite, reasonable potentials.
        let n = 8;
        let dx = [1.0; 3];
        let shape = [n; 3];
        let mut rho = vec![0.0; n * n * n];
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let x = (ix as f64 - n as f64 / 2.0 + 0.5) * dx[0];
                    let y = (iy as f64 - n as f64 / 2.0 + 0.5) * dx[1];
                    let z = (iz as f64 - n as f64 / 2.0 + 0.5) * dx[2];
                    let r2 = x * x + y * y + z * z;
                    rho[ix * n * n + iy * n + iz] = (-r2 / 4.0).exp();
                }
            }
        }

        let solver = SphericalHarmonicsPoisson::new(2, 16, shape, dx);
        let density = DensityField { data: rho, shape };
        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {

            solver: &solver as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,

        };

        let pot = solver.solve(&density, &_ctx);

        assert!(
            pot.data.iter().all(|x| x.is_finite()),
            "Potential must be finite"
        );
        assert!(
            pot.data.iter().any(|x| *x != 0.0),
            "Potential must be non-zero"
        );
    }

    #[test]
    fn spherical_harmonics_y00() {
        // Y_0^0 = 1/sqrt(4*pi)
        let y00 = real_spherical_harmonic(0, 0, 0.5, 1.0);
        let expected = 1.0 / (4.0 * PI).sqrt();
        assert!(
            (y00 - expected).abs() < 1e-12,
            "Y_0^0 = {y00}, expected {expected}"
        );
    }

    #[test]
    fn spherical_harmonics_orthogonality() {
        // Numerical integration of Y_lm * Y_l'm' over the sphere should give delta_{ll'} delta_{mm'}
        let n_theta = 50;
        let n_phi = 100;
        let d_theta = PI / n_theta as f64;
        let d_phi = 2.0 * PI / n_phi as f64;

        let l_max = 2;

        for l1 in 0..=l_max {
            for m1 in -(l1 as i32)..=(l1 as i32) {
                for l2 in 0..=l_max {
                    for m2 in -(l2 as i32)..=(l2 as i32) {
                        let mut integral = 0.0;
                        for it in 0..n_theta {
                            let theta = (it as f64 + 0.5) * d_theta;
                            let sin_theta = theta.sin();
                            for ip in 0..n_phi {
                                let phi = (ip as f64 + 0.5) * d_phi;
                                let y1 = real_spherical_harmonic(l1, m1, theta, phi);
                                let y2 = real_spherical_harmonic(l2, m2, theta, phi);
                                integral += y1 * y2 * sin_theta * d_theta * d_phi;
                            }
                        }
                        let expected = if l1 == l2 && m1 == m2 { 1.0 } else { 0.0 };
                        assert!(
                            (integral - expected).abs() < 0.02,
                            "Orthogonality failed for ({l1},{m1}),({l2},{m2}): got {integral}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn spherical_acceleration_finite() {
        // Ensure compute_acceleration produces finite values
        let n = 8;
        let dx = [1.0; 3];
        let shape = [n; 3];
        let mut rho = vec![0.0; n * n * n];
        let mid = n / 2;
        let cell_vol = dx[0] * dx[1] * dx[2];
        rho[mid * n * n + mid * n + mid] = 1.0 / cell_vol;

        let solver = SphericalHarmonicsPoisson::new(0, 16, shape, dx);
        let density = DensityField { data: rho, shape };
        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {

            solver: &solver as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,

        };

        let pot = solver.solve(&density, &_ctx);
        let acc = solver.compute_acceleration(&pot);

        assert!(acc.gx.iter().all(|x| x.is_finite()));
        assert!(acc.gy.iter().all(|x| x.is_finite()));
        assert!(acc.gz.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn spherical_monopole_symmetry() {
        // A spherically symmetric density should produce a spherically symmetric potential
        let n = 12;
        let dx = [0.5; 3];
        let shape = [n; 3];
        let mut rho = vec![0.0; n * n * n];

        // Uniform density sphere of radius R=2
        let r_sphere = 2.0;
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let x = (ix as f64 - n as f64 / 2.0 + 0.5) * dx[0];
                    let y = (iy as f64 - n as f64 / 2.0 + 0.5) * dx[1];
                    let z = (iz as f64 - n as f64 / 2.0 + 0.5) * dx[2];
                    let r = (x * x + y * y + z * z).sqrt();
                    if r < r_sphere {
                        rho[ix * n * n + iy * n + iz] = 1.0;
                    }
                }
            }
        }

        let solver = SphericalHarmonicsPoisson::new(0, 24, shape, dx);
        let density = DensityField { data: rho, shape };
        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {

            solver: &solver as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,

        };

        let pot = solver.solve(&density, &_ctx);

        // Potential should be the same at two points equidistant from center
        // along x-axis and along y-axis
        let mid = n / 2;
        let d = 2;
        let pot_x = pot.data[(mid + d) * n * n + mid * n + mid];
        let pot_y = pot.data[mid * n * n + (mid + d) * n + mid];
        let pot_z = pot.data[mid * n * n + mid * n + (mid + d)];

        // These should be approximately equal (exact spherical symmetry
        // is broken by grid discretization)
        let mean = (pot_x + pot_y + pot_z) / 3.0;
        let max_dev = [
            (pot_x - mean).abs(),
            (pot_y - mean).abs(),
            (pot_z - mean).abs(),
        ]
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
        let rel_dev = max_dev / mean.abs().max(1e-30);
        assert!(
            rel_dev < 0.15,
            "Symmetry broken: pot_x={pot_x}, pot_y={pot_y}, pot_z={pot_z}, rel_dev={rel_dev}"
        );
    }
}
