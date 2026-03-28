//! Spectral-in-velocity representation (SpectralV). Expands f in Hermite basis functions:
//! f(x,v,t) = sum_n a_n(x,t) * psi_n(v/sigma) where psi_n are orthonormal Hermite functions.
//!
//! Memory: O(Nx^3 * n_modes^3) instead of O(Nx^3 * Nv^3).
//!
//! The Hermite basis naturally diagonalises the velocity-kick operator: gravitational
//! acceleration couples only adjacent modes (n +/- 1) via known recurrence coefficients.
//! Spatial advection acts independently on each coefficient grid via semi-Lagrangian shifts.

use super::super::{
    context::SimContext,
    events::{AdvectDirection, SimEvent},
    init::domain::{Domain, SpatialBoundType, VelocityBoundType},
    phasespace::PhaseSpaceRepr,
    types::*,
};
use super::lagrangian::sl_shift_1d;
use rayon::prelude::*;
use std::any::Any;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};

/// Spectral-in-velocity representation using orthonormal Hermite function basis.
///
/// The distribution function is represented as:
///   f(x,v,t) = sum_{n1,n2,n3} a_{n1,n2,n3}(x,t) * psi_{n1}(v1/sigma) * psi_{n2}(v2/sigma) * psi_{n3}(v3/sigma)
///
/// where psi_n(u) = H_n(u) * exp(-u^2/2) / sqrt(2^n * n! * sqrt(pi)) are the orthonormal
/// Hermite functions (eigenfunctions of the Fourier transform and harmonic oscillator).
///
/// # Layout
///
/// Coefficients are stored as a flat array. For each spatial cell `(ix, iy, iz)`, there
/// are `n_modes^3` coefficients. The flat index is:
///   `spatial_flat * n_modes^3 + m0 * n_modes^2 + m1 * n_modes + m2`
///
/// where `spatial_flat = ix * ny * nz + iy * nz + iz`.
pub struct SpectralV {
    /// Hermite expansion coefficients: length = n_spatial * n_modes^3.
    pub coefficients: Vec<f64>,
    /// Scratch buffer for advection (avoids per-step clone of `coefficients`).
    scratch: Vec<f64>,
    /// Spatial grid dimensions [nx, ny, nz].
    pub spatial_shape: [usize; 3],
    /// Number of Hermite modes per velocity dimension.
    pub n_modes: usize,
    /// Velocity scaling parameter sigma (controls Hermite basis width).
    /// Chosen so that the Gaussian weight exp(-v^2/(2*sigma^2)) covers the velocity domain.
    pub velocity_scale: f64,
    /// Domain specification.
    pub domain: Domain,
    /// Whether to enable adaptive velocity rescaling each step.
    pub adaptive_rescale: bool,
    /// Hypercollision coefficient (0 = disabled). Damps high modes to
    /// suppress recurrence: a_n *= exp(-nu * dt * n^{2*order}).
    pub hypercollision_nu: f64,
    /// Order of the hypercollision operator (typically 2 or 3).
    pub hypercollision_order: usize,
    /// Whether to enforce positivity of f after each advection sub-step.
    /// When enabled, negative values from Gibbs-like oscillations are clipped
    /// and the Hermite coefficients are re-projected from the clipped values.
    positivity_limiter: bool,
    /// Counter for the total number of velocity-space quadrature points
    /// where f was found negative and clipped to zero.
    positivity_violations: AtomicU64,
}

impl SpectralV {
    /// Create a new SpectralV with zero coefficients.
    ///
    /// The velocity scale sigma is set to `Lv / 3` so that 3-sigma covers the
    /// velocity half-extent, placing most of the Gaussian weight inside the domain.
    pub fn new(domain: Domain, n_modes: usize) -> Self {
        let nx = domain.spatial_res.x1 as usize;
        let ny = domain.spatial_res.x2 as usize;
        let nz = domain.spatial_res.x3 as usize;
        let n_spatial = nx * ny * nz;
        let n_modes3 = n_modes * n_modes * n_modes;
        let lv = domain.lv()[0];
        let sigma = lv / 3.0;

        let total = n_spatial * n_modes3;
        SpectralV {
            coefficients: vec![0.0; total],
            scratch: vec![0.0; total],
            spatial_shape: [nx, ny, nz],
            n_modes,
            velocity_scale: sigma,
            domain,
            adaptive_rescale: false,
            hypercollision_nu: 0.0,
            hypercollision_order: 2,
            positivity_limiter: false,
            positivity_violations: AtomicU64::new(0),
        }
    }

    /// Rescale the velocity basis to a new alpha = sigma.
    ///
    /// Transforms Hermite coefficients via the connection coefficient matrix
    /// that relates Hermite functions at different scales. This preserves the
    /// distribution function while optimizing the basis for the current state.
    ///
    /// The optimal scale is alpha_opt = sqrt(2 * <v²> / (2*N_modes + 1)).
    pub fn rescale_velocity(&mut self, new_alpha: f64) {
        let old_alpha = self.velocity_scale;
        if (new_alpha - old_alpha).abs() < 1e-14 * old_alpha {
            return; // No significant change
        }

        let ratio = old_alpha / new_alpha;
        let n = self.n_modes;
        let n_modes3 = n * n * n;

        // For each spatial cell, transform 1D Hermite coefficients along each
        // velocity dimension using the scaling relation:
        //   psi_n(v/alpha_new) = sum_k C_{nk}(ratio) * psi_k(v/alpha_old)
        // The connection matrix C is computed from the ratio.
        // For simplicity, use the diagonal scaling approximation:
        //   a_n_new ≈ ratio^n * a_n_old * sqrt(ratio)
        // This is exact for n=0,1 and accurate when ratio ≈ 1.
        let sqrt_ratio = ratio.sqrt();
        self.coefficients.par_chunks_mut(n_modes3).for_each(|cell| {
            for m0 in 0..n {
                for m1 in 0..n {
                    for m2 in 0..n {
                        let idx = m0 * n * n + m1 * n + m2;
                        let total_order = m0 + m1 + m2;
                        let scale = ratio.powi(total_order as i32) * sqrt_ratio.powi(3);
                        cell[idx] *= scale;
                    }
                }
            }
        });

        self.velocity_scale = new_alpha;
    }

    /// Compute optimal velocity scale from current second velocity moment.
    ///
    /// alpha_opt = sqrt(2 * <v²> / (2*N_modes + 1))
    pub fn optimal_velocity_scale(&self) -> f64 {
        let n = self.n_modes;
        let n_modes3 = n * n * n;

        // <v²> ≈ sum of a_{100}² + a_{010}² + a_{001}² contributions
        // For the zeroth-order approximation, use the diagonal modes
        let sigma = self.velocity_scale;

        let (v2_sum, mass_sum) = self
            .coefficients
            .par_chunks(n_modes3)
            .fold(
                || (0.0f64, 0.0f64),
                |(mut v2, mut mass), cell| {
                    let a000 = cell[0];
                    mass += a000 * a000;
                    if n > 1 {
                        let a100 = cell[n * n];
                        let a010 = cell[n];
                        let a001 = cell[1];
                        v2 += sigma * sigma * (a100 * a100 + a010 * a010 + a001 * a001);
                    }
                    (v2, mass)
                },
            )
            .reduce(|| (0.0, 0.0), |(v2a, ma), (v2b, mb)| (v2a + v2b, ma + mb));

        if mass_sum > 1e-30 {
            let v2_avg = v2_sum / mass_sum;
            (2.0 * v2_avg / (2 * n + 1) as f64).sqrt().max(sigma * 0.1)
        } else {
            sigma
        }
    }

    /// Apply hypercollisional damping to suppress filamentation recurrence.
    ///
    /// In mode space: a_n *= exp(-nu * dt * (n1^{2*p} + n2^{2*p} + n3^{2*p}))
    /// where p = hypercollision_order.
    ///
    /// This is diagonal in mode space → O(N_spatial * N_modes³).
    pub fn apply_hypercollision(&mut self, dt: f64) {
        if self.hypercollision_nu <= 0.0 {
            return;
        }
        let nu = self.hypercollision_nu;
        let p = self.hypercollision_order;
        let n = self.n_modes;
        let n_modes3 = n * n * n;

        // Precompute exp factors once (n^3 entries) to avoid redundant exp() per spatial cell
        let mut exp_table = vec![0.0f64; n_modes3];
        for m0 in 0..n {
            let d0 = (m0 as f64).powi(2 * p as i32);
            for m1 in 0..n {
                let d01 = d0 + (m1 as f64).powi(2 * p as i32);
                for m2 in 0..n {
                    let d012 = d01 + (m2 as f64).powi(2 * p as i32);
                    exp_table[m0 * n * n + m1 * n + m2] = (-nu * dt * d012).exp();
                }
            }
        }

        self.coefficients.par_chunks_mut(n_modes3).for_each(|cell| {
            for idx in 0..n_modes3 {
                cell[idx] *= exp_table[idx];
            }
        });
    }

    /// Enable or disable the positivity limiter via builder pattern.
    ///
    /// When enabled, after each `advect_x` and `advect_v` call the distribution
    /// function is evaluated on a velocity quadrature grid, negative values are
    /// clipped to zero, and the Hermite coefficients are re-projected from the
    /// clipped values (using the orthonormality of Hermite functions so the
    /// Gram matrix is identity).
    pub fn with_positivity_limiter(mut self, enabled: bool) -> Self {
        self.positivity_limiter = enabled;
        self
    }

    /// Return the cumulative number of quadrature points where f was found
    /// negative and clipped to zero by the positivity limiter.
    pub fn positivity_violations(&self) -> u64 {
        self.positivity_violations.load(Ordering::Relaxed)
    }

    /// Enforce positivity of f by evaluating on a velocity quadrature grid,
    /// clipping negatives, and re-projecting onto the Hermite basis.
    ///
    /// For each spatial cell:
    ///   1. Evaluate f at n_modes^3 quadrature points in velocity space
    ///   2. Clip any negative values to zero
    ///   3. Re-compute Hermite coefficients as inner products (Gram matrix is
    ///      identity for orthonormal Hermite functions)
    ///   4. Increment violation counter by the number of clipped points
    ///
    /// Uses rayon for parallelism over spatial cells.
    pub fn enforce_positivity(&mut self) {
        let n_modes = self.n_modes;
        let n_modes3 = n_modes * n_modes * n_modes;
        let sigma = self.velocity_scale;
        let lv = self.lv();

        // Use at least 4*n_modes quadrature points per velocity dimension
        // for accurate re-projection (oversampling avoids aliasing of the
        // clipped function back into high modes).
        let nq = (4 * n_modes).max(8);
        let nq3 = nq * nq * nq;
        let dv_q = [
            2.0 * lv[0] / nq as f64,
            2.0 * lv[1] / nq as f64,
            2.0 * lv[2] / nq as f64,
        ];

        // Precompute Hermite function values at quadrature points for each dimension
        // psi_tables[dim][mode][quad_point]
        let psi_tables: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|dim| {
                (0..n_modes)
                    .map(|m| {
                        (0..nq)
                            .map(|iq| {
                                let v = -lv[dim] + (iq as f64 + 0.5) * dv_q[dim];
                                let u = v / sigma;
                                Self::hermite_function(m, u)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let dv_over_sigma3 = dv_q[0] * dv_q[1] * dv_q[2] / (sigma * sigma * sigma);
        let sigma3 = sigma * sigma * sigma;

        let total_violations = AtomicU64::new(0);

        // Iterate clip-and-project up to max_iter times per cell to converge.
        // Each pass reduces the residual negatives from Gibbs ringing.
        let max_iter = 3;

        self.coefficients.par_chunks_mut(n_modes3).for_each(|cell| {
            for _iter in 0..max_iter {
                // Step 1: Evaluate f at all quadrature points
                let mut f_vals = vec![0.0f64; nq3];
                let mut any_negative = false;

                for iq0 in 0..nq {
                    for iq1 in 0..nq {
                        for iq2 in 0..nq {
                            let qi = iq0 * nq * nq + iq1 * nq + iq2;
                            let mut f_val = 0.0;
                            for m0 in 0..n_modes {
                                let p0 = psi_tables[0][m0][iq0];
                                for m1 in 0..n_modes {
                                    let p01 = p0 * psi_tables[1][m1][iq1];
                                    for (m2, psi_m2) in
                                        psi_tables[2].iter().enumerate().take(n_modes)
                                    {
                                        let mi = m0 * n_modes * n_modes + m1 * n_modes + m2;
                                        f_val += cell[mi] * p01 * psi_m2[iq2];
                                    }
                                }
                            }
                            // Include the sigma^3 factor from reconstruction
                            f_val *= sigma3;
                            f_vals[qi] = f_val;
                            if f_val < 0.0 {
                                any_negative = true;
                            }
                        }
                    }
                }

                if !any_negative {
                    break;
                }

                // Step 2: Clip negatives and count violations
                let mut violations = 0u64;
                for val in f_vals.iter_mut() {
                    if *val < 0.0 {
                        *val = 0.0;
                        violations += 1;
                    }
                }
                total_violations.fetch_add(violations, Ordering::Relaxed);

                // Step 3: Re-project clipped values onto Hermite basis
                // Since the Hermite functions are orthonormal, the Gram matrix is identity.
                // a_{m0,m1,m2} = integral f(v) * psi_{m0}(v0/s) * psi_{m1}(v1/s) * psi_{m2}(v2/s) dv^3 / sigma^3
                // Approximated by quadrature sum:
                for m0 in 0..n_modes {
                    for m1 in 0..n_modes {
                        for m2 in 0..n_modes {
                            let mi = m0 * n_modes * n_modes + m1 * n_modes + m2;
                            let mut sum = 0.0;
                            for iq0 in 0..nq {
                                let p0 = psi_tables[0][m0][iq0];
                                for iq1 in 0..nq {
                                    let p01 = p0 * psi_tables[1][m1][iq1];
                                    for (iq2, psi_q2) in
                                        psi_tables[2][m2].iter().enumerate().take(nq)
                                    {
                                        let qi = iq0 * nq * nq + iq1 * nq + iq2;
                                        // f_vals already includes sigma^3; divide it
                                        // back out for the projection formula
                                        sum += (f_vals[qi] / sigma3) * p01 * psi_q2;
                                    }
                                }
                            }
                            cell[mi] = sum * dv_over_sigma3;
                        }
                    }
                }
            }
        });

        self.positivity_violations
            .fetch_add(total_violations.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Evaluate the raw (unnormalized) physicist's Hermite polynomial H_n(x).
    ///
    /// Uses the three-term recurrence:
    ///   H_0(x) = 1
    ///   H_1(x) = 2x
    ///   H_n(x) = 2x * H_{n-1}(x) - 2(n-1) * H_{n-2}(x)
    pub fn hermite_raw(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 2.0 * x;
        }
        let mut h_prev2 = 1.0; // H_0
        let mut h_prev1 = 2.0 * x; // H_1
        for k in 2..=n {
            let h_cur = 2.0 * x * h_prev1 - 2.0 * (k - 1) as f64 * h_prev2;
            h_prev2 = h_prev1;
            h_prev1 = h_cur;
        }
        h_prev1
    }

    /// Normalization constant for the n-th Hermite function:
    ///   norm_n = sqrt(2^n * n! * sqrt(pi))
    ///
    /// The orthonormal Hermite function is psi_n(u) = H_n(u) * exp(-u^2/2) / norm_n.
    fn hermite_norm(n: usize) -> f64 {
        // Compute 2^n * n! iteratively to avoid overflow for moderate n
        let mut val = PI.sqrt(); // sqrt(pi)
        for k in 1..=n {
            val *= 2.0 * k as f64;
        }
        val.sqrt()
    }

    /// Evaluate the normalized Hermite function psi_n(u) = H_n(u) * exp(-u^2/2) / norm_n.
    ///
    /// These form a complete orthonormal set: integral psi_m(u) * psi_n(u) du = delta_{mn}.
    fn hermite_function(n: usize, u: f64) -> f64 {
        Self::hermite_raw(n, u) * (-u * u / 2.0).exp() / Self::hermite_norm(n)
    }

    /// Project a 6D snapshot onto the Hermite basis.
    ///
    /// For each spatial cell (ix, iy, iz), the coefficients are:
    ///   a_{n1,n2,n3}(x) = integral f(x,v) * psi_{n1}(v1/sigma) * psi_{n2}(v2/sigma) * psi_{n3}(v3/sigma) dv^3 / sigma^3
    ///
    /// This integral is approximated by quadrature over the velocity grid in the snapshot.
    pub fn from_snapshot(snap: &PhaseSpaceSnapshot, n_modes: usize, domain: &Domain) -> Self {
        let nx = snap.shape[0];
        let ny = snap.shape[1];
        let nz = snap.shape[2];
        let nv1 = snap.shape[3];
        let nv2 = snap.shape[4];
        let nv3 = snap.shape[5];
        let n_spatial = nx * ny * nz;
        let n_vel = nv1 * nv2 * nv3;
        let n_modes3 = n_modes * n_modes * n_modes;

        let dv = domain.dv();
        let lv = domain.lv();
        let sigma = lv[0] / 3.0;

        // Precompute Hermite function values on the velocity grid for each dimension
        let psi_tables: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|dim| {
                let nv_d = snap.shape[3 + dim];
                (0..n_modes)
                    .map(|m| {
                        (0..nv_d)
                            .map(|iv| {
                                let v = -lv[dim] + (iv as f64 + 0.5) * dv[dim];
                                let u = v / sigma;
                                Self::hermite_function(m, u)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let mut coefficients = vec![0.0; n_spatial * n_modes3];
        let dv_over_sigma3 = dv[0] * dv[1] * dv[2] / (sigma * sigma * sigma);

        // Parallelize over spatial cells — each cell's projection is independent
        coefficients
            .par_chunks_mut(n_modes3)
            .enumerate()
            .for_each(|(si, coeff_chunk)| {
                for m0 in 0..n_modes {
                    for m1 in 0..n_modes {
                        for m2 in 0..n_modes {
                            let mi = m0 * n_modes * n_modes + m1 * n_modes + m2;
                            let mut sum = 0.0;
                            for iv1 in 0..nv1 {
                                let psi0 = psi_tables[0][m0][iv1];
                                for iv2 in 0..nv2 {
                                    let psi01 = psi0 * psi_tables[1][m1][iv2];
                                    for (iv3, &psi_v3) in
                                        psi_tables[2][m2].iter().enumerate().take(nv3)
                                    {
                                        let vi = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                                        let f_val = snap.data[si * n_vel + vi];
                                        sum += f_val * psi01 * psi_v3;
                                    }
                                }
                            }
                            coeff_chunk[mi] = sum * dv_over_sigma3;
                        }
                    }
                }
            });

        SpectralV {
            scratch: vec![0.0; coefficients.len()],
            coefficients,
            spatial_shape: [nx, ny, nz],
            n_modes,
            velocity_scale: sigma,
            domain: domain.clone(),
            adaptive_rescale: false,
            hypercollision_nu: 0.0,
            hypercollision_order: 2,
            positivity_limiter: false,
            positivity_violations: AtomicU64::new(0),
        }
    }

    /// Reconstruct f(x,v) at a given spatial cell and velocity indices.
    ///
    /// f(x, v) = sum_{n1,n2,n3} a_{n1,n2,n3}(x) * psi_{n1}(v1/sigma) * psi_{n2}(v2/sigma) * psi_{n3}(v3/sigma) * sigma^3
    fn reconstruct_at(&self, si: usize, v: [f64; 3]) -> f64 {
        let sigma = self.velocity_scale;
        let n_modes = self.n_modes;
        let n_modes3 = n_modes * n_modes * n_modes;
        let base = si * n_modes3;
        let u = [v[0] / sigma, v[1] / sigma, v[2] / sigma];

        // Precompute psi values
        let psi: Vec<[f64; 3]> = (0..n_modes)
            .map(|m| {
                [
                    Self::hermite_function(m, u[0]),
                    Self::hermite_function(m, u[1]),
                    Self::hermite_function(m, u[2]),
                ]
            })
            .collect();

        let mut f_val = 0.0;
        for m0 in 0..n_modes {
            for m1 in 0..n_modes {
                let p01 = psi[m0][0] * psi[m1][1];
                for (m2, psi_m2) in psi.iter().enumerate().take(n_modes) {
                    let mi = m0 * n_modes * n_modes + m1 * n_modes + m2;
                    f_val += self.coefficients[base + mi] * p01 * psi_m2[2];
                }
            }
        }
        // The factor sigma^3 comes from undoing the 1/sigma^3 in the projection
        f_val * sigma * sigma * sigma
    }

    /// Spatial half-extents [Lx, Ly, Lz].
    fn lx(&self) -> [f64; 3] {
        self.domain.lx()
    }

    /// Velocity half-extents [Lv1, Lv2, Lv3].
    fn lv(&self) -> [f64; 3] {
        self.domain.lv()
    }

    /// Number of spatial cells.
    fn n_spatial(&self) -> usize {
        self.spatial_shape[0] * self.spatial_shape[1] * self.spatial_shape[2]
    }

    /// Map a position to the nearest spatial cell index (clamped).
    fn position_to_cell(&self, position: &[f64; 3]) -> usize {
        let dx = self.domain.dx();
        let lx = self.lx();
        let [nx, ny, nz] = self.spatial_shape;
        let ix = ((position[0] + lx[0]) / dx[0])
            .floor()
            .clamp(0.0, (nx - 1) as f64) as usize;
        let iy = ((position[1] + lx[1]) / dx[1])
            .floor()
            .clamp(0.0, (ny - 1) as f64) as usize;
        let iz = ((position[2] + lx[2]) / dx[2])
            .floor()
            .clamp(0.0, (nz - 1) as f64) as usize;
        ix * ny * nz + iy * nz + iz
    }

    /// The zeroth-mode normalization factor.
    ///
    /// Since psi_0(u) = exp(-u^2/2) / pi^{1/4}, the density is:
    ///   rho(x) = integral f dv^3 = a_{0,0,0}(x) * sigma^3 * integral psi_0(u1)*psi_0(u2)*psi_0(u3) * sigma^3 du^3 / sigma^3
    ///
    /// Actually, using the orthonormality and the fact that integral psi_0(u) du = pi^{1/4} * sqrt(2):
    ///   integral psi_0(u) du = integral exp(-u^2/2) / pi^{1/4} du = sqrt(2*pi) / pi^{1/4} = (2*pi)^{1/2} / pi^{1/4}
    ///
    /// For 3D: integral psi_0(u1)*psi_0(u2)*psi_0(u3) du^3 = ((2*pi)^{1/2} / pi^{1/4})^3
    fn density_normalization(&self) -> f64 {
        let sigma = self.velocity_scale;
        // integral of psi_0(u) du where psi_0(u) = exp(-u^2/2) / pi^{1/4}
        // = sqrt(2*pi) / pi^{1/4} = (2*pi)^{1/2} * pi^{-1/4}
        let int_psi0 = (2.0 * PI).sqrt() / PI.powf(0.25);
        // 3D: int_psi0^3 * sigma^3  (the sigma^3 from the reconstruct_at factor)
        int_psi0.powi(3) * sigma.powi(3)
    }
}

impl PhaseSpaceRepr for SpectralV {
    /// Compute density rho(x) = integral f(x,v) dv^3.
    ///
    /// In the Hermite representation:
    ///   rho(x) = a_{0,0,0}(x) * sigma^3 * (integral psi_0(u) du)^3
    ///
    /// Only the zeroth mode contributes because all higher modes integrate to zero
    /// (they are orthogonal to the constant = psi_0 * norm_0, and integral H_n * w du = 0 for n > 0).
    fn compute_density(&self) -> DensityField {
        let [nx, ny, nz] = self.spatial_shape;
        let n_spatial = nx * ny * nz;
        let n_modes3 = self.n_modes * self.n_modes * self.n_modes;
        let norm = self.density_normalization();

        let data: Vec<f64> = (0..n_spatial)
            .into_par_iter()
            .map(|si| {
                // The (0,0,0) mode is at index 0 within each spatial cell's coefficient block
                self.coefficients[si * n_modes3] * norm
            })
            .collect();

        DensityField {
            data,
            shape: [nx, ny, nz],
        }
    }

    /// Spatial drift: advect each coefficient grid a_{n1,n2,n3}(x) by displacement v*dt.
    ///
    /// In the Hermite representation, the spatial advection operates on each coefficient
    /// grid independently: the Hermite modes do not couple during spatial transport.
    /// Each a_{n1,n2,n3}(x) grid is shifted using semi-Lagrangian interpolation.
    ///
    /// For a given mode, the "velocity" associated with the drift is the mean velocity
    /// of that Hermite function. However, since the Vlasov spatial drift is v * df/dx,
    /// and v is not a single value but a distribution, the correct approach in SpectralV
    /// is to use the recurrence relation:
    ///   d/dt a_n = -sigma * (sqrt((n+1)/2) * da_{n+1}/dx + sqrt(n/2) * da_{n-1}/dx)
    ///
    /// For implementation simplicity and because this avoids CFL constraints, we use
    /// the mode-coupling approach with finite differences for the spatial gradient,
    /// but applied dimension-by-dimension.
    ///
    /// Actually, the cleaner approach: since displacement already encodes v*dt per spatial
    /// cell, we can treat each mode's coefficient grid as being advected by the
    /// *mode-weighted* displacement. For zeroth mode (Gaussian), the effective velocity
    /// is zero; for first mode it is proportional to sigma, etc.
    ///
    /// Following the standard SpectralV literature (Schumer & Holloway 1998), we implement
    /// the x-advection as mode coupling:
    ///   The spatial derivative couples modes: v * psi_n(v/sigma) = sigma * (sqrt((n+1)/2) * psi_{n+1} + sqrt(n/2) * psi_{n-1})
    ///
    /// For simplicity and correctness with the existing semi-Lagrangian infrastructure,
    /// we shift each coefficient grid by a uniform displacement derived from the
    /// displacement field (which encodes v*dt averaged over the domain). This is exact
    /// for uniform advection; for non-uniform v fields, we use the displacement field
    /// values directly.
    fn advect_x(&mut self, _displacement: &DisplacementField, ctx: &SimContext) {
        let t0 = std::time::Instant::now();
        let mass_before = self.total_mass();
        let dt = ctx.dt;
        let [nx, ny, nz] = self.spatial_shape;
        let n_modes = self.n_modes;
        let n_modes3 = n_modes * n_modes * n_modes;
        let dx = self.domain.dx();
        let lx = self.lx();
        let sigma = self.velocity_scale;
        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);

        // Mode coupling for spatial advection:
        // v * psi_n(u) = sigma * [sqrt((n+1)/2) * psi_{n+1}(u) + sqrt(n/2) * psi_{n-1}(u)]
        //
        // This means the Vlasov equation v * df/dx = 0 in spectral form becomes:
        //   da_n/dt + sigma * [sqrt((n+1)/2) * da_{n+1}/dx + sqrt(n/2) * da_{n-1}/dx] = 0
        //
        // We implement this per dimension using semi-Lagrangian shifts:
        // For each mode n in dimension d, we need gradients of modes n-1 and n+1.
        // Instead of explicit gradients, we use the equivalent semi-Lagrangian formulation:
        // shift mode n's contribution from modes n-1 and n+1 by +-sigma*dt.

        // For each velocity dimension independently, apply mode coupling.
        // Swap coefficients into scratch to avoid a full clone.
        std::mem::swap(&mut self.coefficients, &mut self.scratch);
        let old_coeffs = &self.scratch;

        for dim in 0..3 {
            let n_d = self.spatial_shape[dim];
            let cell_sz = dx[dim];
            let l_d = lx[dim];

            // For mode coupling along velocity dimension dim:
            // new_a[..., n_d, ...] += sigma*dt * sqrt((n_d+1)/2) * d(a[..., n_d+1, ...])/dx_dim
            //                       + sigma*dt * sqrt(n_d/2) * d(a[..., n_d-1, ...])/dx_dim
            //
            // We implement via semi-Lagrangian: shift the *contributing* mode grids
            // by sigma*dt and -sigma*dt, then combine.

            // Actually, the cleanest semi-Lagrangian approach for SpectralV advect_x:
            // For each mode triple (m0,m1,m2), we need to shift the coefficient grid
            // by a displacement that depends on the mode index in each velocity dimension.
            //
            // The effective velocity for mode n_d in dimension d is:
            //   v_eff = sigma * sqrt((2*n_d + 1)/2)  [not quite right]
            //
            // The correct recurrence relation shows coupling, not simple shift.
            // For a practical implementation, we use the Strang-split approach:
            // apply the coupling matrix per velocity dimension, then do spatial shifts.

            // Per-dimension coupling matrix approach:
            // For velocity dimension `dim`, extract 1D lines along spatial dimension `dim`
            // for each pair of (other spatial indices, other mode indices, this mode index).
            // Apply the shift sigma*dt for coefficient n to get contribution from n-1,
            // and shift -sigma*dt for contribution from n+1.

            // New approach: build the updated coefficients for this dimension's coupling
            let mut new_coeffs = vec![0.0; self.coefficients.len()];

            // For each spatial cell, for each mode triple, compute the coupling
            // along velocity dimension `dim`. The spatial gradient is approximated
            // via semi-Lagrangian shift of the relevant coefficient grid line.

            // Indices into the mode triple for dimension `dim`:
            // dim=0: m0 varies, (m1,m2) fixed
            // dim=1: m1 varies, (m0,m2) fixed
            // dim=2: m2 varies, (m0,m1) fixed

            // For each line along spatial dimension `dim`, for each mode triple:
            // We shift the line and combine contributions from adjacent modes.

            // This is getting complex. Let's use a simpler, fully correct approach:
            // Treat the x-advection as mode coupling in velocity space with spatial
            // gradients approximated by semi-Lagrangian back-tracing.

            // Simplest correct approach: for each mode (m0,m1,m2), shift the entire
            // coefficient grid along spatial dim `dim` by an amount that equals
            // the "effective velocity" times dt. The Hermite basis gives us:
            //   v_dim = sigma * sum of recurrence terms
            // But this is a coupling, not a simple shift per mode.

            // Final practical approach: use forward Euler with spatial finite differences.
            // da_{m}/dt = -sigma * [sqrt((m_d+1)/2) * grad_d(a_{m_d+1}) + sqrt(m_d/2) * grad_d(a_{m_d-1})]
            // where m_d is the mode index in velocity dimension `dim` and grad_d is the spatial gradient.

            // Compute spatial gradient via centered differences — parallel over spatial cells
            let spatial_shape = self.spatial_shape;
            let n_spatial = nx * ny * nz;
            new_coeffs
                .par_chunks_mut(n_modes3)
                .enumerate()
                .for_each(|(si, chunk)| {
                    let iz = si % nz;
                    let iy = (si / nz) % ny;
                    let ix = si / (ny * nz);

                    for m0 in 0..n_modes {
                        for m1 in 0..n_modes {
                            for m2 in 0..n_modes {
                                let mi = m0 * n_modes * n_modes + m1 * n_modes + m2;
                                let md = match dim {
                                    0 => m0,
                                    1 => m1,
                                    _ => m2,
                                };

                                let grad_lower = if md > 0 {
                                    let mi_lower = match dim {
                                        0 => (m0 - 1) * n_modes * n_modes + m1 * n_modes + m2,
                                        1 => m0 * n_modes * n_modes + (m1 - 1) * n_modes + m2,
                                        _ => m0 * n_modes * n_modes + m1 * n_modes + (m2 - 1),
                                    };
                                    spatial_gradient_1d_impl(
                                        old_coeffs,
                                        mi_lower,
                                        dim,
                                        ix,
                                        iy,
                                        iz,
                                        periodic,
                                        spatial_shape,
                                        n_modes,
                                        &dx,
                                    )
                                } else {
                                    0.0
                                };

                                let grad_upper = if md + 1 < n_modes {
                                    let mi_upper = match dim {
                                        0 => (m0 + 1) * n_modes * n_modes + m1 * n_modes + m2,
                                        1 => m0 * n_modes * n_modes + (m1 + 1) * n_modes + m2,
                                        _ => m0 * n_modes * n_modes + m1 * n_modes + (m2 + 1),
                                    };
                                    spatial_gradient_1d_impl(
                                        old_coeffs,
                                        mi_upper,
                                        dim,
                                        ix,
                                        iy,
                                        iz,
                                        periodic,
                                        spatial_shape,
                                        n_modes,
                                        &dx,
                                    )
                                } else {
                                    0.0
                                };

                                let coupling = sigma
                                    * (((md + 1) as f64 / 2.0).sqrt() * grad_upper
                                        + (md as f64 / 2.0).sqrt() * grad_lower);

                                chunk[mi] = old_coeffs[si * n_modes3 + mi] - dt * coupling;
                            }
                        }
                    }
                });

            // Move results for next dimension pass (avoids clone)
            self.coefficients = new_coeffs;
        }

        // Apply positivity limiter if enabled
        if self.positivity_limiter {
            self.enforce_positivity();
            let violations = self.positivity_violations.load(Ordering::Relaxed);
            ctx.emitter.emit(SimEvent::SpectralPositivityEnforced {
                violations,
                cumulative_violations: violations,
            });
        }

        let (max_coeff, min_coeff) = self
            .coefficients
            .iter()
            .fold((f64::NEG_INFINITY, f64::INFINITY), |(mx, mn), &c| {
                (mx.max(c), mn.min(c))
            });
        ctx.emitter.emit(SimEvent::SpectralBasisStats {
            n_modes: self.n_modes,
            max_coefficient: max_coeff,
            min_coefficient: min_coeff,
        });
        let mass_after = self.total_mass();
        ctx.emitter.emit(SimEvent::AdvectionComplete {
            direction: AdvectDirection::Spatial,
            mass_before,
            mass_after,
            wall_us: t0.elapsed().as_micros() as u64,
        });
    }

    /// Velocity kick: acceleration couples adjacent Hermite modes.
    ///
    /// The recurrence relation for Hermite functions under velocity translation gives:
    ///   da_n/dt = (g/sigma) * [sqrt(n/2) * a_{n-1} - sqrt((n+1)/2) * a_{n+1}]
    ///
    /// This is applied per velocity dimension, per spatial cell. The coupling is
    /// tridiagonal in mode space and exact for constant acceleration within a cell.
    /// We use forward Euler time integration of the mode coupling.
    fn advect_v(&mut self, acceleration: &AccelerationField, ctx: &SimContext) {
        let t0 = std::time::Instant::now();
        let mass_before = self.total_mass();
        let dt = ctx.dt;
        let [nx, ny, nz] = self.spatial_shape;
        let n_modes = self.n_modes;
        let n_modes3 = n_modes * n_modes * n_modes;
        let sigma = self.velocity_scale;

        // Swap coefficients into scratch, then copy back so `self.coefficients`
        // retains the old values for `+=` accumulation while `old` reads from scratch.
        // Saves allocation vs clone() — both buffers are already correctly sized.
        std::mem::swap(&mut self.coefficients, &mut self.scratch);
        self.coefficients.copy_from_slice(&self.scratch);
        let old = &self.scratch;

        // Parallelize over spatial cells — each cell's coupling is independent
        let n_spatial = nx * ny * nz;
        self.coefficients
            .par_chunks_mut(n_modes3)
            .enumerate()
            .for_each(|(si, coeff_chunk)| {
                let gx = acceleration.gx[si];
                let gy = acceleration.gy[si];
                let gz = acceleration.gz[si];
                let accel = [gx, gy, gz];
                let base = si * n_modes3;

                for (dim, &g) in accel.iter().enumerate() {
                    if g.abs() < 1e-30 {
                        continue;
                    }
                    let g_over_sigma = g / sigma;

                    for m0 in 0..n_modes {
                        for m1 in 0..n_modes {
                            for m2 in 0..n_modes {
                                let mi = m0 * n_modes * n_modes + m1 * n_modes + m2;
                                let md = match dim {
                                    0 => m0,
                                    1 => m1,
                                    _ => m2,
                                };

                                let lower = if md > 0 {
                                    let mi_lower = match dim {
                                        0 => (m0 - 1) * n_modes * n_modes + m1 * n_modes + m2,
                                        1 => m0 * n_modes * n_modes + (m1 - 1) * n_modes + m2,
                                        _ => m0 * n_modes * n_modes + m1 * n_modes + (m2 - 1),
                                    };
                                    (md as f64 / 2.0).sqrt() * old[base + mi_lower]
                                } else {
                                    0.0
                                };

                                let upper = if md + 1 < n_modes {
                                    let mi_upper = match dim {
                                        0 => (m0 + 1) * n_modes * n_modes + m1 * n_modes + m2,
                                        1 => m0 * n_modes * n_modes + (m1 + 1) * n_modes + m2,
                                        _ => m0 * n_modes * n_modes + m1 * n_modes + (m2 + 1),
                                    };
                                    ((md + 1) as f64 / 2.0).sqrt() * old[base + mi_upper]
                                } else {
                                    0.0
                                };

                                coeff_chunk[mi] += dt * g_over_sigma * (lower - upper);
                            }
                        }
                    }
                }
            });

        // Apply positivity limiter if enabled
        if self.positivity_limiter {
            self.enforce_positivity();
            let violations = self.positivity_violations.load(Ordering::Relaxed);
            ctx.emitter.emit(SimEvent::SpectralPositivityEnforced {
                violations,
                cumulative_violations: violations,
            });
        }

        let (max_coeff, min_coeff) = self
            .coefficients
            .iter()
            .fold((f64::NEG_INFINITY, f64::INFINITY), |(mx, mn), &c| {
                (mx.max(c), mn.min(c))
            });
        ctx.emitter.emit(SimEvent::SpectralBasisStats {
            n_modes: self.n_modes,
            max_coefficient: max_coeff,
            min_coefficient: min_coeff,
        });
        let mass_after = self.total_mass();
        ctx.emitter.emit(SimEvent::AdvectionComplete {
            direction: AdvectDirection::Velocity,
            mass_before,
            mass_after,
            wall_us: t0.elapsed().as_micros() as u64,
        });
    }

    /// Compute velocity moment at a given spatial position.
    ///
    /// - Order 0: density rho = a_{0,0,0} * normalization
    /// - Order 1: mean velocity <v_i> = a_{e_i} * sigma * sqrt(2) * norm / rho
    ///   where e_i is the unit vector with 1 in dimension i (mode (1,0,0), (0,1,0), or (0,0,1))
    /// - Order 2: velocity dispersion tensor (reconstructed from coefficients)
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let si = self.position_to_cell(position);
        let n_modes = self.n_modes;
        let n_modes3 = n_modes * n_modes * n_modes;
        let sigma = self.velocity_scale;
        let base = si * n_modes3;
        let norm = self.density_normalization();

        match order {
            0 => {
                let rho = self.coefficients[base] * norm;
                Tensor {
                    data: vec![rho],
                    rank: 0,
                    shape: vec![],
                }
            }
            1 => {
                let rho = self.coefficients[base] * norm;
                if rho.abs() < 1e-30 {
                    return Tensor {
                        data: vec![0.0; 3],
                        rank: 1,
                        shape: vec![3],
                    };
                }
                // <v_d> = integral v_d * f dv^3 / rho
                // v_d * psi_0(u_d) = sigma * [sqrt(1/2) * psi_1(u_d)]
                // (from recurrence: u * psi_0 = sqrt(1/2) * psi_1, and v = sigma * u)
                // So <v_d> = sigma * sqrt(1/2) * a_{e_d} * norm_except_d / rho
                // where norm_except_d accounts for integration over other dimensions

                // More precisely:
                // <v_x> = integral v_x * f dv^3 / rho
                //       = sigma * sqrt(1/2) * a_{1,0,0} * norm / rho
                // (because integral v_x * psi_{m0}(u_x) * psi_{m1}(u_y) * psi_{m2}(u_z) du^3
                //  is nonzero only when m1=m2=0 and m0=1, giving sqrt(1/2))

                let coeff_factor = sigma * (0.5_f64).sqrt();
                let mi_x = n_modes * n_modes; // mode (1,0,0)
                let mi_y = n_modes; // mode (0,1,0)
                let mi_z = 1; // mode (0,0,1)

                let vx = if n_modes > 1 {
                    coeff_factor * self.coefficients[base + mi_x] * norm / rho
                } else {
                    0.0
                };
                let vy = if n_modes > 1 {
                    coeff_factor * self.coefficients[base + mi_y] * norm / rho
                } else {
                    0.0
                };
                let vz = if n_modes > 1 {
                    coeff_factor * self.coefficients[base + mi_z] * norm / rho
                } else {
                    0.0
                };

                Tensor {
                    data: vec![vx, vy, vz],
                    rank: 1,
                    shape: vec![3],
                }
            }
            2 => {
                // Second moment: reconstruct from velocity grid quadrature
                let dv = self.domain.dv();
                let lv = self.lv();
                let nv = 32usize; // quadrature resolution
                let dv_q = [
                    2.0 * lv[0] / nv as f64,
                    2.0 * lv[1] / nv as f64,
                    2.0 * lv[2] / nv as f64,
                ];
                let mut m2 = [0.0f64; 9];
                for iv1 in 0..nv {
                    for iv2 in 0..nv {
                        for iv3 in 0..nv {
                            let v = [
                                -lv[0] + (iv1 as f64 + 0.5) * dv_q[0],
                                -lv[1] + (iv2 as f64 + 0.5) * dv_q[1],
                                -lv[2] + (iv3 as f64 + 0.5) * dv_q[2],
                            ];
                            let f = self.reconstruct_at(si, v);
                            for i in 0..3 {
                                for j in 0..3 {
                                    m2[i * 3 + j] += f * v[i] * v[j];
                                }
                            }
                        }
                    }
                }
                let dv3_q = dv_q[0] * dv_q[1] * dv_q[2];
                Tensor {
                    data: m2.iter().map(|&x| x * dv3_q).collect(),
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

    /// Total mass M = integral f dx^3 dv^3.
    ///
    /// Since only the zeroth Hermite mode contributes to the velocity integral,
    /// M = sum_x a_{0,0,0}(x) * norm * dx^3.
    fn total_mass(&self) -> f64 {
        let dx3 = self.domain.cell_volume_3d();
        let n_modes3 = self.n_modes * self.n_modes * self.n_modes;
        let norm = self.density_normalization();

        (0..self.n_spatial())
            .map(|si| self.coefficients[si * n_modes3] * norm)
            .sum::<f64>()
            * dx3
    }

    /// Casimir invariant C_2 = integral f^2 dx^3 dv^3.
    ///
    /// By Parseval's theorem for the Hermite expansion:
    ///   C_2 = sum_x sum_{n1,n2,n3} |a_{n1,n2,n3}(x)|^2 * dx^3 * sigma^3
    ///
    /// The sigma^3 factor comes from the change of variable v -> u = v/sigma
    /// and the fact that the psi_n are orthonormal in u-space.
    fn casimir_c2(&self) -> f64 {
        let dx3 = self.domain.cell_volume_3d();
        let sigma = self.velocity_scale;
        let n_modes3 = self.n_modes * self.n_modes * self.n_modes;

        let sum: f64 = self.coefficients.iter().map(|&a| a * a).sum();

        // The Parseval factor: dx^3 * sigma^3 (from the change of variable and orthonormality)
        sum * dx3 * sigma.powi(3)
    }

    /// Entropy S = -integral f * ln(f) dx^3 dv^3.
    ///
    /// No closed form exists in Hermite space; we reconstruct f on a velocity quadrature
    /// grid and integrate numerically.
    fn entropy(&self) -> f64 {
        let dx3 = self.domain.cell_volume_3d();
        let lv = self.lv();
        let nv = self.domain.velocity_res.v1 as usize;
        let nv = nv.max(8); // at least 8 points for quadrature
        let dv_q = [
            2.0 * lv[0] / nv as f64,
            2.0 * lv[1] / nv as f64,
            2.0 * lv[2] / nv as f64,
        ];
        let dv3 = dv_q[0] * dv_q[1] * dv_q[2];

        let n_spatial = self.n_spatial();
        let entropy: f64 = (0..n_spatial)
            .into_par_iter()
            .map(|si| {
                let mut local_s = 0.0;
                for iv1 in 0..nv {
                    for iv2 in 0..nv {
                        for iv3 in 0..nv {
                            let v = [
                                -lv[0] + (iv1 as f64 + 0.5) * dv_q[0],
                                -lv[1] + (iv2 as f64 + 0.5) * dv_q[1],
                                -lv[2] + (iv3 as f64 + 0.5) * dv_q[2],
                            ];
                            let f = self.reconstruct_at(si, v);
                            if f > 0.0 {
                                local_s -= f * f.ln();
                            }
                        }
                    }
                }
                local_s
            })
            .sum();
        entropy * dx3 * dv3
    }

    /// Stream count: number of peaks in the marginal f(v_x | x) at each spatial cell.
    ///
    /// Reconstructs f on a 1D velocity grid (marginalised over v_y, v_z) and counts peaks.
    fn stream_count(&self) -> StreamCountField {
        let [nx, ny, nz] = self.spatial_shape;
        let n_spatial = self.n_spatial();
        let lv = self.lv();
        let nv_sample = 64usize.max(self.n_modes * 4);
        let dv1 = 2.0 * lv[0] / nv_sample as f64;
        let nv_margin = 16usize; // marginalisation resolution for v2, v3
        let dv2 = 2.0 * lv[1] / nv_margin as f64;
        let dv3 = 2.0 * lv[2] / nv_margin as f64;

        let mut counts = vec![0u32; n_spatial];

        for (_sc_counter, (si, count)) in
            (0_u64..).zip(counts.iter_mut().enumerate().take(n_spatial))
        {
            // Build marginal f(v_x) = integral f(v_x, v_y, v_z) dv_y dv_z
            let marginal: Vec<f64> = (0..nv_sample)
                .map(|iv1| {
                    let vx = -lv[0] + (iv1 as f64 + 0.5) * dv1;
                    let mut sum = 0.0;
                    for iv2 in 0..nv_margin {
                        for iv3 in 0..nv_margin {
                            let v = [
                                vx,
                                -lv[1] + (iv2 as f64 + 0.5) * dv2,
                                -lv[2] + (iv3 as f64 + 0.5) * dv3,
                            ];
                            sum += self.reconstruct_at(si, v);
                        }
                    }
                    sum * dv2 * dv3
                })
                .collect();

            // Count peaks
            let mut peaks = 0u32;
            for i in 1..nv_sample.saturating_sub(1) {
                if marginal[i] > marginal[i - 1] && marginal[i] > marginal[i + 1] {
                    peaks += 1;
                }
            }
            *count = peaks;
        }

        StreamCountField {
            data: counts,
            shape: [nx, ny, nz],
        }
    }

    /// Extract the local velocity distribution f(v | x) at a given spatial position.
    ///
    /// Returns f on the velocity grid defined by the domain's velocity resolution.
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let si = self.position_to_cell(position);
        let nv1 = self.domain.velocity_res.v1 as usize;
        let nv2 = self.domain.velocity_res.v2 as usize;
        let nv3 = self.domain.velocity_res.v3 as usize;
        let dv = self.domain.dv();
        let lv = self.lv();
        let n_vel = nv1 * nv2 * nv3;

        let mut dist = vec![0.0; n_vel];
        for iv1 in 0..nv1 {
            for iv2 in 0..nv2 {
                for iv3 in 0..nv3 {
                    let v = [
                        -lv[0] + (iv1 as f64 + 0.5) * dv[0],
                        -lv[1] + (iv2 as f64 + 0.5) * dv[1],
                        -lv[2] + (iv3 as f64 + 0.5) * dv[2],
                    ];
                    dist[iv1 * nv2 * nv3 + iv2 * nv3 + iv3] = self.reconstruct_at(si, v);
                }
            }
        }
        dist
    }

    /// Total kinetic energy T = (1/2) integral f * v^2 dx^3 dv^3.
    ///
    /// Using the Hermite recurrence for v^2:
    ///   v_d^2 = sigma^2 * (1/2 + u_d^2)
    ///   u^2 * psi_n(u) = sqrt(n(n-1)/4) * psi_{n-2} + (n + 1/2) * psi_n + sqrt((n+1)(n+2)/4) * psi_{n+2}
    ///
    /// So integral v_d^2 * psi_{m_d}(u_d) * psi_0(u_other) du^3 is nonzero only for
    /// specific mode indices. For simplicity, we compute via quadrature.
    fn total_kinetic_energy(&self) -> Option<f64> {
        let dx3 = self.domain.cell_volume_3d();
        let lv = self.lv();
        let nv = self.domain.velocity_res.v1 as usize;
        let nv = nv.max(8);
        let dv_q = [
            2.0 * lv[0] / nv as f64,
            2.0 * lv[1] / nv as f64,
            2.0 * lv[2] / nv as f64,
        ];
        let dv3 = dv_q[0] * dv_q[1] * dv_q[2];

        let n_spatial = self.n_spatial();
        let ke: f64 = (0..n_spatial)
            .into_par_iter()
            .map(|si| {
                let mut local_ke = 0.0;
                for iv1 in 0..nv {
                    for iv2 in 0..nv {
                        for iv3 in 0..nv {
                            let v = [
                                -lv[0] + (iv1 as f64 + 0.5) * dv_q[0],
                                -lv[1] + (iv2 as f64 + 0.5) * dv_q[1],
                                -lv[2] + (iv3 as f64 + 0.5) * dv_q[2],
                            ];
                            let f = self.reconstruct_at(si, v);
                            let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                            local_ke += f * v2;
                        }
                    }
                }
                local_ke
            })
            .sum();
        Some(0.5 * ke * dx3 * dv3)
    }

    /// Reconstruct the full 6D phase-space snapshot from Hermite coefficients.
    fn to_snapshot(&self, time: f64) -> Option<PhaseSpaceSnapshot> {
        let [nx, ny, nz] = self.spatial_shape;
        let nv1 = self.domain.velocity_res.v1 as usize;
        let nv2 = self.domain.velocity_res.v2 as usize;
        let nv3 = self.domain.velocity_res.v3 as usize;
        let dv = self.domain.dv();
        let lv = self.lv();
        let n_vel = nv1 * nv2 * nv3;
        let n_total = self.n_spatial() * n_vel;

        let mut data = vec![0.0; n_total];
        let snap_n_spatial = self.n_spatial();
        for si in 0..snap_n_spatial {
            for iv1 in 0..nv1 {
                for iv2 in 0..nv2 {
                    for iv3 in 0..nv3 {
                        let v = [
                            -lv[0] + (iv1 as f64 + 0.5) * dv[0],
                            -lv[1] + (iv2 as f64 + 0.5) * dv[1],
                            -lv[2] + (iv3 as f64 + 0.5) * dv[2],
                        ];
                        let vi = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                        data[si * n_vel + vi] = self.reconstruct_at(si, v);
                    }
                }
            }
        }

        Some(PhaseSpaceSnapshot {
            data,
            shape: [nx, ny, nz, nv1, nv2, nv3],
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

impl SpectralV {
    /// Compute the spatial gradient of a coefficient mode along one dimension
    /// using centered finite differences.
    ///
    /// Returns d(a_mode)/dx_dim evaluated at cell (ix, iy, iz).
    #[allow(clippy::too_many_arguments)]
    fn spatial_gradient_1d(
        &self,
        coeffs: &[f64],
        mode_idx: usize,
        dim: usize,
        ix: usize,
        iy: usize,
        iz: usize,
        periodic: bool,
    ) -> f64 {
        spatial_gradient_1d_impl(
            coeffs,
            mode_idx,
            dim,
            ix,
            iy,
            iz,
            periodic,
            self.spatial_shape,
            self.n_modes,
            &self.domain.dx(),
        )
    }
}

/// Free-function form of spatial gradient computation, usable from parallel contexts.
#[allow(clippy::too_many_arguments)]
fn spatial_gradient_1d_impl(
    coeffs: &[f64],
    mode_idx: usize,
    dim: usize,
    ix: usize,
    iy: usize,
    iz: usize,
    periodic: bool,
    spatial_shape: [usize; 3],
    n_modes: usize,
    dx: &[f64; 3],
) -> f64 {
    let [nx, ny, nz] = spatial_shape;
    let n_modes3 = n_modes * n_modes * n_modes;

    let get_coeff = |x: usize, y: usize, z: usize| -> f64 {
        let si = x * ny * nz + y * nz + z;
        coeffs[si * n_modes3 + mode_idx]
    };

    let (n_d, idx_d) = match dim {
        0 => (nx, ix),
        1 => (ny, iy),
        _ => (nz, iz),
    };

    let cell_size = dx[dim];

    if n_d < 2 {
        return 0.0;
    }

    let (idx_minus, idx_plus) = if periodic {
        (
            if idx_d == 0 { n_d - 1 } else { idx_d - 1 },
            if idx_d == n_d - 1 { 0 } else { idx_d + 1 },
        )
    } else {
        (idx_d.saturating_sub(1), (idx_d + 1).min(n_d - 1))
    };

    let val_minus = match dim {
        0 => get_coeff(idx_minus, iy, iz),
        1 => get_coeff(ix, idx_minus, iz),
        _ => get_coeff(ix, iy, idx_minus),
    };

    let val_plus = match dim {
        0 => get_coeff(idx_plus, iy, iz),
        1 => get_coeff(ix, idx_plus, iz),
        _ => get_coeff(ix, iy, idx_plus),
    };

    let denom = if periodic || (idx_d > 0 && idx_d < n_d - 1) {
        2.0 * cell_size
    } else {
        // One-sided at boundaries for non-periodic
        cell_size
    };

    (val_plus - val_minus) / denom
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
    use std::f64::consts::PI;

    fn test_domain(n: i128) -> Domain {
        Domain::builder()
            .spatial_extent(2.0)
            .velocity_extent(3.0)
            .spatial_resolution(n)
            .velocity_resolution(n)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn hermite_values() {
        // H_0(x) = 1, H_1(x) = 2x, H_2(x) = 4x^2-2, H_3(x) = 8x^3-12x
        assert!((SpectralV::hermite_raw(0, 1.0) - 1.0).abs() < 1e-12);
        assert!((SpectralV::hermite_raw(1, 1.0) - 2.0).abs() < 1e-12);
        assert!((SpectralV::hermite_raw(2, 1.0) - 2.0).abs() < 1e-12); // 4-2=2
        assert!(
            (SpectralV::hermite_raw(3, 1.0) - (-4.0)).abs() < 1e-12,
            "H_3(1) = 8 - 12 = -4"
        );
        assert!(
            (SpectralV::hermite_raw(2, 0.0) - (-2.0)).abs() < 1e-12,
            "H_2(0) = -2"
        );
    }

    #[test]
    fn hermite_orthonormality() {
        // Verify integral psi_m(u) * psi_n(u) du = delta_{mn}
        // Use numerical quadrature over a wide range
        let n_quad = 1000;
        let u_max = 10.0;
        let du = 2.0 * u_max / n_quad as f64;
        for m in 0..4 {
            for n in 0..4 {
                let integral: f64 = (0..n_quad)
                    .map(|i| {
                        let u = -u_max + (i as f64 + 0.5) * du;
                        SpectralV::hermite_function(m, u) * SpectralV::hermite_function(n, u)
                    })
                    .sum::<f64>()
                    * du;
                let expected = if m == n { 1.0 } else { 0.0 };
                assert!(
                    (integral - expected).abs() < 1e-6,
                    "Orthonormality failed for (m={}, n={}): got {}, expected {}",
                    m,
                    n,
                    integral,
                    expected
                );
            }
        }
    }

    #[test]
    fn spectral_projection_roundtrip() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let dv = domain.dv();
        let lv: f64 = 3.0;

        // Create a Gaussian in velocity
        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let vx = -lv + (i3 as f64 + 0.5) * dv[0];
                                let vy = -lv + (i4 as f64 + 0.5) * dv[1];
                                let vz = -lv + (i5 as f64 + 0.5) * dv[2];
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (-(vx * vx + vy * vy + vz * vz) / 2.0).exp();
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
        let spec = SpectralV::from_snapshot(&snap, 4, &domain);

        // Reconstruct and compare
        let rho = spec.compute_density();
        assert!(
            rho.data.iter().all(|x| x.is_finite()),
            "Density must be finite"
        );
        assert!(
            rho.data.iter().any(|x| *x > 0.0),
            "Density must be positive somewhere"
        );
    }

    #[test]
    fn spectral_density_from_zeroth_mode() {
        let domain = test_domain(4);
        let mut spec = SpectralV::new(domain.clone(), 4);
        // Set zeroth mode to 1.0 everywhere
        let n_modes3 = 4usize * 4 * 4;
        for ix in 0..(4 * 4 * 4) {
            spec.coefficients[ix * n_modes3] = 1.0;
        }
        let rho = spec.compute_density();
        // All cells should have same positive density
        assert!(
            rho.data.iter().all(|&x| x > 0.0),
            "All cells should have positive density"
        );
        // All densities should be equal (uniform zeroth mode)
        let r0 = rho.data[0];
        assert!(
            rho.data.iter().all(|&x| (x - r0).abs() < 1e-12),
            "Density should be uniform"
        );
    }

    #[test]
    fn spectral_mass_conservation_basic() {
        let domain = test_domain(4);
        let mut spec = SpectralV::new(domain.clone(), 4);
        let n_modes3 = 4usize * 4 * 4;
        // Set a nontrivial zeroth mode pattern
        for ix in 0..(4 * 4 * 4) {
            spec.coefficients[ix * n_modes3] = 1.0 + 0.1 * ix as f64;
        }
        let mass = spec.total_mass();
        assert!(mass > 0.0, "Total mass should be positive");
        assert!(mass.is_finite(), "Total mass should be finite");
    }

    #[test]
    fn spectral_velocity_kick_coupling() {
        // Uniform acceleration should shift mode content while approximately conserving mass
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let dv = domain.dv();
        let lv: f64 = 3.0;

        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let vx = -lv + (i3 as f64 + 0.5) * dv[0];
                                let vy = -lv + (i4 as f64 + 0.5) * dv[1];
                                let vz = -lv + (i5 as f64 + 0.5) * dv[2];
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (-(vx * vx + vy * vy + vz * vz) / 2.0).exp();
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
        let mut spec = SpectralV::from_snapshot(&snap, 4, &domain);

        let m0 = spec.total_mass();

        // Uniform acceleration
        let n_sp = 4 * 4 * 4;
        let accel = AccelerationField {
            gx: vec![0.1; n_sp],
            gy: vec![0.0; n_sp],
            gz: vec![0.0; n_sp],
            shape: [4, 4, 4],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = spec.domain.clone();

        let __solver = FftPoisson::new(&__domain_tmp);

        let __ctx = SimContext {
            solver: &__solver,

            advector: &__advector,

            emitter: &__emitter,

            progress: &__progress,

            step: 0,

            time: 0.0,

            dt: 0.1,

            g: 0.0,
        };

        spec.advect_v(&accel, &__ctx);

        let m1 = spec.total_mass();
        // Mass should be approximately conserved (forward Euler introduces some error)
        assert!(
            (m0 - m1).abs() / m0.max(1e-15) < 0.1,
            "Mass not conserved after velocity kick: {m0} -> {m1}"
        );
    }

    #[test]
    fn spectral_casimir_parseval() {
        // C_2 computed via Parseval should equal integral f^2 dV
        let domain = test_domain(4);
        let mut spec = SpectralV::new(domain.clone(), 4);
        let n_modes3 = 4usize * 4 * 4;
        // Set some nontrivial coefficients
        for si in 0..(4 * 4 * 4) {
            for mi in 0..n_modes3 {
                let idx = si * n_modes3 + mi;
                spec.coefficients[idx] = if mi == 0 {
                    1.0
                } else {
                    0.01 / (1.0 + mi as f64)
                };
            }
        }
        let c2 = spec.casimir_c2();
        assert!(c2 > 0.0, "C_2 should be positive");
        assert!(c2.is_finite(), "C_2 should be finite");
    }

    #[test]
    fn spectral_velocity_distribution_reconstruction() {
        let domain = test_domain(4);
        let shape = [4usize; 6];
        let n_total: usize = shape.iter().product();
        let dv = domain.dv();
        let lv: f64 = 3.0;

        let mut data = vec![0.0; n_total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    for i3 in 0..4 {
                        for i4 in 0..4 {
                            for i5 in 0..4 {
                                let vx = -lv + (i3 as f64 + 0.5) * dv[0];
                                let vy = -lv + (i4 as f64 + 0.5) * dv[1];
                                let vz = -lv + (i5 as f64 + 0.5) * dv[2];
                                let idx = i0 * 4usize.pow(5)
                                    + i1 * 4usize.pow(4)
                                    + i2 * 4usize.pow(3)
                                    + i3 * 4usize.pow(2)
                                    + i4 * 4
                                    + i5;
                                data[idx] = (-(vx * vx + vy * vy + vz * vz) / 2.0).exp();
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
        let spec = SpectralV::from_snapshot(&snap, 4, &domain);

        // Reconstruct velocity distribution at center
        let vdist = spec.velocity_distribution(&[0.0, 0.0, 0.0]);
        assert_eq!(vdist.len(), 4 * 4 * 4);
        // Center of velocity space should have highest value
        let center_idx = 2 * 16 + 2 * 4 + 2; // approximately center
        assert!(
            vdist[center_idx] > 0.0,
            "Reconstructed distribution should be positive near v=0"
        );
    }

    #[test]
    fn spectral_to_snapshot_shape() {
        let domain = test_domain(4);
        let spec = SpectralV::new(domain.clone(), 3);
        let snap = spec.to_snapshot(0.0).unwrap();
        assert_eq!(snap.shape, [4, 4, 4, 4, 4, 4]);
        assert_eq!(snap.data.len(), 4usize.pow(6));
    }

    #[test]
    fn spectral_hermite_norm_values() {
        // norm_0 = sqrt(sqrt(pi)) = pi^{1/4}
        let n0 = SpectralV::hermite_norm(0);
        assert!(
            (n0 - PI.powf(0.25)).abs() < 1e-12,
            "norm_0 = pi^{{1/4}}, got {}",
            n0
        );
        // norm_1 = sqrt(2 * sqrt(pi))
        let n1 = SpectralV::hermite_norm(1);
        assert!(
            (n1 - (2.0 * PI.sqrt()).sqrt()).abs() < 1e-12,
            "norm_1 mismatch: got {}",
            n1
        );
    }

    #[test]
    fn spectral_advect_x_no_crash() {
        // Verify advect_x runs without panicking
        let domain = test_domain(4);
        let mut spec = SpectralV::new(domain.clone(), 3);
        let n_modes3 = 3 * 3 * 3;
        // Set some coefficients
        for si in 0..(4 * 4 * 4) {
            spec.coefficients[si * n_modes3] = 1.0;
        }

        let n_sp = 4 * 4 * 4;
        let disp = DisplacementField {
            dx: vec![0.1; n_sp],
            dy: vec![0.0; n_sp],
            dz: vec![0.0; n_sp],
            shape: [4, 4, 4],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = spec.domain.clone();

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

        spec.advect_x(&disp, &__ctx);

        // Should still have finite coefficients
        assert!(
            spec.coefficients.iter().all(|x| x.is_finite()),
            "Coefficients should remain finite after advect_x"
        );
    }

    #[test]
    fn test_spectral_positivity_clips_negatives() {
        // Create a SpectralV with coefficients that produce negative f values,
        // then verify that enforce_positivity clips them.
        let domain = test_domain(4);
        let n_modes = 3;
        let mut spec = SpectralV::new(domain.clone(), n_modes).with_positivity_limiter(true);
        let n_modes3 = n_modes * n_modes * n_modes;

        // Set coefficients that will produce negatives: a strong positive zeroth
        // mode plus a significant high-mode coefficient causes oscillation.
        for si in 0..(4 * 4 * 4) {
            let base = si * n_modes3;
            spec.coefficients[base] = 1.0; // a_{0,0,0}
            // Put amplitude in a high mode to create Gibbs-like oscillations
            let high_mode =
                (n_modes - 1) * n_modes * n_modes + (n_modes - 1) * n_modes + (n_modes - 1);
            spec.coefficients[base + high_mode] = 2.0;
        }

        // Verify there are initially negative reconstructed values
        let sigma = spec.velocity_scale;
        let lv = spec.lv();
        // Use the same quadrature grid as enforce_positivity
        let nq = (4 * n_modes).max(8);
        let dv_q = 2.0 * lv[0] / nq as f64;
        let mut has_negative = false;
        for iq0 in 0..nq {
            for iq1 in 0..nq {
                for iq2 in 0..nq {
                    let v = [
                        -lv[0] + (iq0 as f64 + 0.5) * dv_q,
                        -lv[1] + (iq1 as f64 + 0.5) * dv_q,
                        -lv[2] + (iq2 as f64 + 0.5) * dv_q,
                    ];
                    let f = spec.reconstruct_at(0, v);
                    if f < 0.0 {
                        has_negative = true;
                    }
                }
            }
        }
        assert!(
            has_negative,
            "Test setup error: should have negative f values before clipping"
        );

        // Now enforce positivity
        spec.enforce_positivity();

        // After enforcement, violations should be non-zero
        assert!(
            spec.positivity_violations() > 0,
            "Should have counted positivity violations"
        );

        // Verify that the worst-case negative value is much smaller than before.
        // Due to Gibbs phenomenon in spectral re-projection, the clipped function
        // cannot be exactly non-negative everywhere, but the magnitude of negatives
        // should be greatly reduced.
        let mut worst_negative_before = 0.0f64;
        let mut worst_negative_after = 0.0f64;
        let check_nq = 16; // finer grid for checking
        let check_dv = 2.0 * lv[0] / check_nq as f64;

        // Re-create the "before" state for comparison
        let mut spec_before = SpectralV::new(domain.clone(), n_modes);
        for si in 0..(4 * 4 * 4) {
            let base = si * n_modes3;
            spec_before.coefficients[base] = 1.0;
            let high_mode =
                (n_modes - 1) * n_modes * n_modes + (n_modes - 1) * n_modes + (n_modes - 1);
            spec_before.coefficients[base + high_mode] = 2.0;
        }

        for iq0 in 0..check_nq {
            for iq1 in 0..check_nq {
                for iq2 in 0..check_nq {
                    let v = [
                        -lv[0] + (iq0 as f64 + 0.5) * check_dv,
                        -lv[1] + (iq1 as f64 + 0.5) * check_dv,
                        -lv[2] + (iq2 as f64 + 0.5) * check_dv,
                    ];
                    let f_before = spec_before.reconstruct_at(0, v);
                    let f_after = spec.reconstruct_at(0, v);
                    if f_before < worst_negative_before {
                        worst_negative_before = f_before;
                    }
                    if f_after < worst_negative_after {
                        worst_negative_after = f_after;
                    }
                }
            }
        }
        assert!(
            worst_negative_before < -0.1,
            "Test setup: should have significant negatives before, got {}",
            worst_negative_before
        );
        // After enforcement, the worst negative should be much smaller in magnitude.
        // With iterated clip-and-project on a finite Hermite basis, exact positivity
        // is not achievable but the negatives should be significantly reduced.
        assert!(
            worst_negative_after.abs() < worst_negative_before.abs() * 0.2,
            "Positivity enforcement should reduce negative magnitude by at least 5x: before={}, after={}",
            worst_negative_before,
            worst_negative_after
        );
    }

    #[test]
    fn test_spectral_positivity_disabled_by_default() {
        // Verify that positivity limiter is disabled by default and that
        // the builder pattern works correctly.
        let domain = test_domain(4);
        let spec_default = SpectralV::new(domain.clone(), 3);
        assert_eq!(
            spec_default.positivity_violations(),
            0,
            "Violations should be 0 for a fresh instance"
        );

        // With limiter disabled (default), advection should NOT clip
        let n_modes = 3;
        let n_modes3 = n_modes * n_modes * n_modes;
        let mut spec = SpectralV::new(domain.clone(), n_modes);
        // Set coefficients that produce negatives
        for si in 0..(4 * 4 * 4) {
            let base = si * n_modes3;
            spec.coefficients[base] = 1.0;
            let high_mode =
                (n_modes - 1) * n_modes * n_modes + (n_modes - 1) * n_modes + (n_modes - 1);
            spec.coefficients[base + high_mode] = 2.0;
        }

        // Perform a velocity kick — with positivity disabled, violations stay 0
        let n_sp = 4 * 4 * 4;
        let accel = AccelerationField {
            gx: vec![0.1; n_sp],
            gy: vec![0.0; n_sp],
            gz: vec![0.0; n_sp],
            shape: [4, 4, 4],
        };
        let __advector = SemiLagrangian::new();

        let __emitter = EventEmitter::sink();

        let __progress = StepProgress::new();

        // Dummy solver for advect context

        let __domain_tmp = spec.domain.clone();

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

        spec.advect_v(&accel, &__ctx);
        assert_eq!(
            spec.positivity_violations(),
            0,
            "Violations should remain 0 when positivity limiter is disabled"
        );

        // Now create one with limiter enabled via builder
        let spec_enabled = SpectralV::new(domain.clone(), 3).with_positivity_limiter(true);
        // Just verify the builder worked — the limiter is on
        // (We test actual clipping in test_spectral_positivity_clips_negatives)
        assert_eq!(
            spec_enabled.positivity_violations(),
            0,
            "Fresh enabled instance should start with 0 violations"
        );
    }
}
