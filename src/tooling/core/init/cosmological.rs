//! Zel'dovich approximation initial conditions for cosmological structure formation.
//! f is a 3D sheet in 6D phase space (cold dark matter).

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use rayon::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Cosmological parameters for Friedmann background.
pub struct CosmologyParams {
    pub h0: Decimal,
    pub omega_m: Decimal,
    pub omega_lambda: Decimal,
    pub a_init: Decimal,
    // Cached f64 values for hot-path computation
    h0_f64: f64,
    omega_m_f64: f64,
    omega_lambda_f64: f64,
    a_init_f64: f64,
}

impl CosmologyParams {
    /// Create from f64 parameters (backward-compatible).
    pub fn new(h0: f64, omega_m: f64, omega_lambda: f64, a_init: f64) -> Self {
        Self {
            h0: Decimal::from_f64_retain(h0).unwrap_or(Decimal::ZERO),
            omega_m: Decimal::from_f64_retain(omega_m).unwrap_or(Decimal::ZERO),
            omega_lambda: Decimal::from_f64_retain(omega_lambda).unwrap_or(Decimal::ZERO),
            a_init: Decimal::from_f64_retain(a_init).unwrap_or(Decimal::ZERO),
            h0_f64: h0,
            omega_m_f64: omega_m,
            omega_lambda_f64: omega_lambda,
            a_init_f64: a_init,
        }
    }

    /// Create from Decimal parameters (exact config).
    pub fn new_decimal(
        h0: Decimal,
        omega_m: Decimal,
        omega_lambda: Decimal,
        a_init: Decimal,
    ) -> Self {
        Self {
            h0_f64: h0.to_f64().unwrap_or(0.0),
            omega_m_f64: omega_m.to_f64().unwrap_or(0.0),
            omega_lambda_f64: omega_lambda.to_f64().unwrap_or(0.0),
            a_init_f64: a_init.to_f64().unwrap_or(0.0),
            h0,
            omega_m,
            omega_lambda,
            a_init,
        }
    }
}

/// 1D matter power spectrum P(k) for seeding perturbations.
pub struct PowerSpectrum {
    /// (k, P(k)) pairs.
    pub values: Vec<(f64, f64)>,
}

/// Zel'dovich pancake IC: cold dark matter sheet.
/// f(x,v,t₀) = ρ̄·δ³(v − v₀(x)) where v₀ is the Zel'dovich velocity field.
pub struct ZeldovichIC {
    pub mean_density: Decimal,
    pub h0: Decimal,
    pub omega_m: Decimal,
    pub omega_lambda: Decimal,
    pub scale_factor_init: Decimal,
    pub random_seed: u64,
    // Cached f64 values for hot-path computation
    mean_density_f64: f64,
    h0_f64: f64,
    omega_m_f64: f64,
    omega_lambda_f64: f64,
    scale_factor_init_f64: f64,
}

impl ZeldovichIC {
    /// Create from f64 parameters via CosmologyParams (backward-compatible).
    pub fn new(mean_density: f64, cosmology: CosmologyParams, seed: u64) -> Self {
        Self {
            mean_density: Decimal::from_f64_retain(mean_density).unwrap_or(Decimal::ZERO),
            h0: cosmology.h0,
            omega_m: cosmology.omega_m,
            omega_lambda: cosmology.omega_lambda,
            scale_factor_init: cosmology.a_init,
            random_seed: seed,
            mean_density_f64: mean_density,
            h0_f64: cosmology.h0_f64,
            omega_m_f64: cosmology.omega_m_f64,
            omega_lambda_f64: cosmology.omega_lambda_f64,
            scale_factor_init_f64: cosmology.a_init_f64,
        }
    }

    /// Create from Decimal parameters (exact config).
    pub fn new_decimal(
        mean_density: Decimal,
        h0: Decimal,
        omega_m: Decimal,
        omega_lambda: Decimal,
        scale_factor_init: Decimal,
        seed: u64,
    ) -> Self {
        Self {
            mean_density_f64: mean_density.to_f64().unwrap_or(0.0),
            h0_f64: h0.to_f64().unwrap_or(0.0),
            omega_m_f64: omega_m.to_f64().unwrap_or(0.0),
            omega_lambda_f64: omega_lambda.to_f64().unwrap_or(0.0),
            scale_factor_init_f64: scale_factor_init.to_f64().unwrap_or(0.0),
            mean_density,
            h0,
            omega_m,
            omega_lambda,
            scale_factor_init,
            random_seed: seed,
        }
    }

    /// Create a simple single-mode Zel'dovich pancake for testing.
    ///
    /// v₀(x) = -A·sin(k·x₁)
    /// f(x,v) = (ρ̄/S) · exp(-(v - v₀(x))² / (2σ_v²))
    ///
    /// where S normalises the Gaussian to unity over velocity space.
    pub fn new_single_mode(
        mean_density: f64,
        amplitude: f64,
        wavenumber: f64,
        sigma_v: f64,
    ) -> ZeldovichSingleMode {
        ZeldovichSingleMode {
            mean_density,
            amplitude,
            wavenumber,
            sigma_v,
        }
    }

    /// Generate the Zel'dovich displacement field s(q) from P(k) via FFT.
    ///
    /// Uses `Xorshift64` RNG with Box-Muller transform for Gaussian random
    /// field generation in Fourier space, then IFFT to real space.
    pub fn displacement_field(&self, domain: &Domain) -> [Vec<f64>; 3] {
        use crate::tooling::core::algos::aca::Xorshift64;
        use rustfft::num_complex::Complex64;
        use std::f64::consts::PI;

        let nx = domain.spatial_res.x1 as usize;
        let ny = domain.spatial_res.x2 as usize;
        let nz = domain.spatial_res.x3 as usize;
        let n_total = nx * ny * nz;

        // Box lengths: domain is [-L, L], so total length = 2L
        let lx_half = domain.lx();
        let lx = 2.0 * lx_half[0];
        let ly = 2.0 * lx_half[1];
        let lz = 2.0 * lx_half[2];
        let volume = lx * ly * lz;

        let dx_grid = lx / nx as f64;
        let dy_grid = ly / ny as f64;
        let dz_grid = lz / nz as f64;

        // Nyquist wavenumbers
        let kn_x = PI / dx_grid;
        let kn_y = PI / dy_grid;
        let kn_z = PI / dz_grid;
        let kn2_avg = (kn_x * kn_x + kn_y * kn_y + kn_z * kn_z) / 3.0;

        // Mean spacing and target RMS displacement
        let mean_spacing = (dx_grid * dy_grid * dz_grid).cbrt();
        let target_rms = 0.1 * mean_spacing;

        // RNG + Box-Muller helper
        let mut rng = Xorshift64::new(self.random_seed);

        fn box_muller(rng: &mut Xorshift64) -> (f64, f64) {
            let u1 = (rng.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
            let u2 = (rng.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            (r * theta.cos(), r * theta.sin())
        }

        // Generate δ(k) in Fourier space with Harrison-Zel'dovich-like spectrum
        // P(k) = A * |k| * exp(-|k|² / (2 k_N²))
        // Amplitude A is calibrated after to get target_rms displacement.
        let mut delta_k = vec![Complex64::new(0.0, 0.0); n_total];

        // First pass: generate with A=1, then rescale
        for i0 in 0..nx {
            let kx_idx = if i0 <= nx / 2 {
                i0 as f64
            } else {
                i0 as f64 - nx as f64
            };
            let kx = 2.0 * PI * kx_idx / lx;
            for i1 in 0..ny {
                let ky_idx = if i1 <= ny / 2 {
                    i1 as f64
                } else {
                    i1 as f64 - ny as f64
                };
                let ky = 2.0 * PI * ky_idx / ly;
                for i2 in 0..nz {
                    let kz_idx = if i2 <= nz / 2 {
                        i2 as f64
                    } else {
                        i2 as f64 - nz as f64
                    };
                    let kz = 2.0 * PI * kz_idx / lz;

                    let k2 = kx * kx + ky * ky + kz * kz;
                    let idx = i0 * ny * nz + i1 * nz + i2;

                    if k2 == 0.0 {
                        // Skip DC mode
                        delta_k[idx] = Complex64::new(0.0, 0.0);
                        continue;
                    }

                    let k_mag = k2.sqrt();
                    // P(k) = |k| * exp(-k² / (2 k_N²))  (A=1 initially)
                    let pk = k_mag * (-k2 / (2.0 * kn2_avg)).exp();
                    let amplitude = (pk / volume).sqrt();

                    let (z1, z2) = box_muller(&mut rng);
                    delta_k[idx] = Complex64::new(z1 * amplitude, z2 * amplitude);
                }
            }
        }

        // Enforce Hermitian symmetry: δ(-k) = conj(δ(k))
        for i0 in 0..nx {
            let j0 = if i0 == 0 { 0 } else { nx - i0 };
            for i1 in 0..ny {
                let j1 = if i1 == 0 { 0 } else { ny - i1 };
                for i2 in 0..nz {
                    let j2 = if i2 == 0 { 0 } else { nz - i2 };

                    let idx_pos = i0 * ny * nz + i1 * nz + i2;
                    let idx_neg = j0 * ny * nz + j1 * nz + j2;

                    if idx_pos < idx_neg {
                        // Set conjugate pair
                        let val = delta_k[idx_pos];
                        delta_k[idx_neg] = val.conj();
                    } else if idx_pos == idx_neg {
                        // Self-conjugate mode: must be real
                        delta_k[idx_pos] = Complex64::new(delta_k[idx_pos].re, 0.0);
                    }
                }
            }
        }

        // Compute displacement in k-space: s_k = i·k / |k|² · δ(k)
        let mut sx_k = vec![Complex64::new(0.0, 0.0); n_total];
        let mut sy_k = vec![Complex64::new(0.0, 0.0); n_total];
        let mut sz_k = vec![Complex64::new(0.0, 0.0); n_total];

        for i0 in 0..nx {
            let kx_idx = if i0 <= nx / 2 {
                i0 as f64
            } else {
                i0 as f64 - nx as f64
            };
            let kx = 2.0 * PI * kx_idx / lx;
            for i1 in 0..ny {
                let ky_idx = if i1 <= ny / 2 {
                    i1 as f64
                } else {
                    i1 as f64 - ny as f64
                };
                let ky = 2.0 * PI * ky_idx / ly;
                for i2 in 0..nz {
                    let kz_idx = if i2 <= nz / 2 {
                        i2 as f64
                    } else {
                        i2 as f64 - nz as f64
                    };
                    let kz = 2.0 * PI * kz_idx / lz;

                    let k2 = kx * kx + ky * ky + kz * kz;
                    let idx = i0 * ny * nz + i1 * nz + i2;

                    if k2 == 0.0 {
                        continue;
                    }

                    // s_k = i·k/|k|² · δ(k)
                    let dk = delta_k[idx];
                    let i_dk = Complex64::new(-dk.im, dk.re); // i * δ(k)
                    sx_k[idx] = i_dk * (kx / k2);
                    sy_k[idx] = i_dk * (ky / k2);
                    sz_k[idx] = i_dk * (kz / k2);
                }
            }
        }

        // 3D IFFT each component
        let sx = ifft_3d(&sx_k, [nx, ny, nz]);
        let sy = ifft_3d(&sy_k, [nx, ny, nz]);
        let sz = ifft_3d(&sz_k, [nx, ny, nz]);

        // Rescale to achieve target RMS displacement
        let rms = {
            let sum_sq: f64 = sx
                .iter()
                .chain(sy.iter())
                .chain(sz.iter())
                .map(|v| v * v)
                .sum();
            (sum_sq / (3.0 * n_total as f64)).sqrt()
        };

        if rms > 0.0 {
            let scale = target_rms / rms;
            let sx: Vec<f64> = sx.iter().map(|v| v * scale).collect();
            let sy: Vec<f64> = sy.iter().map(|v| v * scale).collect();
            let sz: Vec<f64> = sz.iter().map(|v| v * scale).collect();
            [sx, sy, sz]
        } else {
            [sx, sy, sz]
        }
    }

    /// Zel'dovich velocity v₀(q) = a·H(a)·f(a)·s(q) (growing mode).
    ///
    /// f(a) ≈ Ω_m(a)^{0.55} is the logarithmic growth rate.
    pub fn velocity_field(&self, domain: &Domain) -> [Vec<f64>; 3] {
        let [sx, sy, sz] = self.displacement_field(domain);

        let a = self.scale_factor_init_f64;
        let a3 = a * a * a;

        // E(a) = H(a)/H0
        let e2 = self.omega_m_f64 / a3 + self.omega_lambda_f64;
        let h_a = self.h0_f64 * e2.sqrt(); // H(a)

        // Ω_m(a) at this scale factor
        let om_a = self.omega_m_f64 / (a3 * e2);

        // Logarithmic growth rate f ≈ Ω_m(a)^{0.55}
        let f_growth = om_a.powf(0.55);

        // v(q) = a · H(a) · f(a) · s(q)
        let factor = a * h_a * f_growth;

        let vx: Vec<f64> = sx.iter().map(|s| s * factor).collect();
        let vy: Vec<f64> = sy.iter().map(|s| s * factor).collect();
        let vz: Vec<f64> = sz.iter().map(|s| s * factor).collect();

        [vx, vy, vz]
    }

    /// Sample onto 6D grid as a thin Gaussian in velocity centred on v₀(x).
    ///
    /// f(x,v) = ρ̄ / ((2π)^{3/2} σ_v³) · exp(-|v - v₀(x)|² / (2σ_v²))
    /// where σ_v = 0.3 · max(Δv) keeps the distribution cold but resolved.
    pub fn sample_on_grid(
        &self,
        domain: &Domain,
        progress: Option<&crate::tooling::core::progress::StepProgress>,
    ) -> PhaseSpaceSnapshot {
        use std::f64::consts::PI;

        let [vx0, vy0, vz0] = self.velocity_field(domain);

        let nx1 = domain.spatial_res.x1 as usize;
        let nx2 = domain.spatial_res.x2 as usize;
        let nx3 = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;

        let dv = domain.dv();
        let lv = domain.lv();

        // Velocity spread: cold but resolved
        let sigma = 0.3 * dv[0].max(dv[1]).max(dv[2]);
        let norm = self.mean_density_f64 / ((2.0 * PI).sqrt() * sigma).powi(3);

        // Strides for row-major 6D layout [x1, x2, x3, v1, v2, v3]
        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
        let mut data = vec![0.0f64; total];

        let inv_2sig2 = 1.0 / (2.0 * sigma * sigma);

        let counter = std::sync::atomic::AtomicU64::new(0);
        let report_interval = (nx1 / 100).max(1) as u64;

        // Establish 0% baseline so the TUI doesn't jump to a non-zero first value
        if let Some(p) = progress {
            p.set_intra_progress(0, nx1 as u64);
        }

        // Parallelize over ix1 slabs — each slab is independent
        data.par_chunks_mut(s_x1)
            .enumerate()
            .for_each(|(ix1, chunk)| {
                for ix2 in 0..nx2 {
                    for ix3 in 0..nx3 {
                        let spatial_idx = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                        let v0x = vx0[spatial_idx];
                        let v0y = vy0[spatial_idx];
                        let v0z = vz0[spatial_idx];

                        let base = ix2 * s_x2 + ix3 * s_x3;

                        for iv1 in 0..nv1 {
                            let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                            let dv1 = v1 - v0x;
                            for iv2 in 0..nv2 {
                                let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                                let dv2 = v2 - v0y;
                                for iv3 in 0..nv3 {
                                    let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                    let dv3 = v3 - v0z;
                                    let v2sq = dv1 * dv1 + dv2 * dv2 + dv3 * dv3;
                                    let f = norm * (-v2sq * inv_2sig2).exp();
                                    chunk[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f;
                                }
                            }
                        }
                    }
                }

                if let Some(p) = progress {
                    let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, nx1 as u64);
                    }
                }
            });

        PhaseSpaceSnapshot {
            data,
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time: 0.0,
        }
    }
}

/// 3D inverse FFT of a complex array, returning real part.
/// Same pattern as `fft_3d_inverse` in `poisson/tensor_poisson.rs`.
fn ifft_3d(data: &[rustfft::num_complex::Complex64], shape: [usize; 3]) -> Vec<f64> {
    use rustfft::FftPlanner;
    use rustfft::num_complex::Complex64;

    let [n0, n1, n2] = shape;
    let n_total = n0 * n1 * n2;
    assert_eq!(data.len(), n_total);

    let mut buf = data.to_vec();
    let scale = 1.0 / n_total as f64;
    let mut planner = FftPlanner::new();

    // IFFT along axis 2 (fastest varying)
    let ifft2 = planner.plan_fft_inverse(n2);
    for i0 in 0..n0 {
        for i1 in 0..n1 {
            let start = i0 * n1 * n2 + i1 * n2;
            ifft2.process(&mut buf[start..start + n2]);
        }
    }

    // IFFT along axis 1
    let ifft1 = planner.plan_fft_inverse(n1);
    let mut line = vec![Complex64::new(0.0, 0.0); n1];
    for i0 in 0..n0 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                line[i1] = buf[i0 * n1 * n2 + i1 * n2 + i2];
            }
            ifft1.process(&mut line);
            for i1 in 0..n1 {
                buf[i0 * n1 * n2 + i1 * n2 + i2] = line[i1];
            }
        }
    }

    // IFFT along axis 0
    let ifft0 = planner.plan_fft_inverse(n0);
    let mut line0 = vec![Complex64::new(0.0, 0.0); n0];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            for i0 in 0..n0 {
                line0[i0] = buf[i0 * n1 * n2 + i1 * n2 + i2];
            }
            ifft0.process(&mut line0);
            for i0 in 0..n0 {
                buf[i0 * n1 * n2 + i1 * n2 + i2] = line0[i0];
            }
        }
    }

    buf.iter().map(|c| c.re * scale).collect()
}

/// Simple single-mode Zel'dovich pancake IC for validation tests.
pub struct ZeldovichSingleMode {
    pub mean_density: f64,
    pub amplitude: f64,
    pub wavenumber: f64,
    pub sigma_v: f64,
}

impl ZeldovichSingleMode {
    /// Sample f(x,v) = ρ̄/(√(2π)σ_v)³ · exp(-(v - v₀(x))²/(2σ_v²))
    /// where v₀(x) = (-A·sin(k·x₁), 0, 0)
    pub fn sample_on_grid(
        &self,
        domain: &Domain,
        progress: Option<&crate::tooling::core::progress::StepProgress>,
    ) -> PhaseSpaceSnapshot {
        use std::f64::consts::PI;

        let nx1 = domain.spatial_res.x1 as usize;
        let nx2 = domain.spatial_res.x2 as usize;
        let nx3 = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;

        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
        let mut data = vec![0.0f64; total];

        let sigma = self.sigma_v;
        let norm = self.mean_density / ((2.0 * PI).sqrt() * sigma).powi(3);

        let counter = std::sync::atomic::AtomicU64::new(0);
        let report_interval = (nx1 / 100).max(1) as u64;

        // Establish 0% baseline so the TUI doesn't jump to a non-zero first value
        if let Some(p) = progress {
            p.set_intra_progress(0, nx1 as u64);
        }

        for ix1 in 0..nx1 {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            let v0x = -self.amplitude * (self.wavenumber * x1).sin();

            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    let base = ix1 * s_x1 + ix2 * s_x2 + ix3 * s_x3;

                    for iv1 in 0..nv1 {
                        let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let dv1 = v1 - v0x;
                                let dv2 = v2;
                                let dv3 = v3;
                                let v2sq = dv1 * dv1 + dv2 * dv2 + dv3 * dv3;
                                let f = norm * (-v2sq / (2.0 * sigma * sigma)).exp();
                                data[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f;
                            }
                        }
                    }
                }
            }

            if let Some(p) = progress {
                let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c.is_multiple_of(report_interval) {
                    p.set_intra_progress(c, nx1 as u64);
                }
            }
        }

        PhaseSpaceSnapshot {
            data,
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time: 0.0,
        }
    }
}

/// Growth factor D(a) for flat ΛCDM cosmology.
///
/// Uses the Carroll-Press-Turner (1992) approximation:
///   D(a) ≈ (5/2) Ω_m(a) / [Ω_m(a)^{4/7} - Ω_Λ(a) + (1 + Ω_m(a)/2)(1 + Ω_Λ(a)/70)]
///
/// where Ω_m(a) = Ω_m / (Ω_m + Ω_Λ a³) and Ω_Λ(a) = Ω_Λ a³ / (Ω_m + Ω_Λ a³).
/// Normalized so D(a) → a in the matter-dominated era.
pub fn growth_factor(a: f64, params: &CosmologyParams) -> f64 {
    let om = params.omega_m_f64;
    let ol = params.omega_lambda_f64;

    // E²(a) = Ω_m/a³ + Ω_Λ  (flat universe, Ω_k = 0)
    let a3 = a * a * a;
    let e2 = om / a3 + ol;

    // Ω_m(a) and Ω_Λ(a) at scale factor a
    let om_a = om / (a3 * e2);
    let ol_a = ol / e2;

    // Carroll-Press-Turner approximation
    let d_unnorm =
        2.5 * om_a / (om_a.powf(4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0));

    // Normalize: in matter domination D ∝ a, so normalize to D(1) and scale
    let e2_1 = om + ol;
    let om_1 = om / e2_1;
    let ol_1 = ol / e2_1;
    let d_1 = 2.5 * om_1 / (om_1.powf(4.0 / 7.0) - ol_1 + (1.0 + om_1 / 2.0) * (1.0 + ol_1 / 70.0));

    a * d_unnorm / d_1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};

    #[test]
    fn zeldovich_multi_mode_displacement_field() {
        let domain = Domain::builder()
            .spatial_extent(10.0)
            .velocity_extent(2.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();
        let cosmo = CosmologyParams::new(70.0, 0.3, 0.7, 0.01);
        let ic = ZeldovichIC::new(1.0, cosmo, 42);
        let [sx, sy, sz] = ic.displacement_field(&domain);
        let n = 8 * 8 * 8;
        assert_eq!(sx.len(), n);
        assert_eq!(sy.len(), n);
        assert_eq!(sz.len(), n);
        // Displacement should be finite and have zero mean (no DC mode)
        assert!(sx.iter().all(|v| v.is_finite()));
        let mean: f64 = sx.iter().sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 1.0,
            "displacement mean should be near zero, got {mean}"
        );
    }

    #[test]
    fn zeldovich_multi_mode_sample() {
        // Use small H0 and late scale factor so velocities stay within grid extent.
        // v ~ a * H(a) * f * s  where s ~ 0.1 * dx ~ 0.25, need v << velocity_extent.
        let domain = Domain::builder()
            .spatial_extent(10.0)
            .velocity_extent(2.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();
        let cosmo = CosmologyParams::new(1.0, 0.3, 0.7, 1.0);
        let ic = ZeldovichIC::new(1.0, cosmo, 42);
        let snap = ic.sample_on_grid(&domain, None);
        assert_eq!(snap.shape, [8, 8, 8, 8, 8, 8]);
        assert!(snap.data.iter().all(|v| v.is_finite() && *v >= 0.0));
        let mass: f64 = snap.data.iter().sum::<f64>();
        assert!(mass > 0.0, "total mass should be positive");
    }
}
