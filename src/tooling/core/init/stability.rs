//! Disk stability initial conditions.
//!
//! Constructs an axisymmetric exponential disk (with optional bulge and dark-matter halo)
//! using the Shu (1969) distribution function f(E, Lz). A seeded azimuthal perturbation
//! mode (e.g. m=2 for bar, m=3 for triangle) allows controlled study of Jeans instability,
//! bar formation, and spiral-arm growth. The Toomre Q diagnostic is provided to assess
//! local gravitational stability before running the simulation.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use super::isolated::IsolatedEquilibrium;
use rayon::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Complete elliptic integral K(m) via Abramowitz & Stegun polynomial approximation (17.3.34).
/// Argument m is the parameter (not the modulus k; m = k²). Valid for 0 ≤ m < 1.
/// Maximum absolute error ~2e-8.
fn elliptic_k(m: f64) -> f64 {
    if m < 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }
    if m >= 1.0 {
        return f64::INFINITY;
    }
    let m1 = 1.0 - m;
    let a0 = 1.386_294_361_12;
    let a1 = 0.096_663_442_59;
    let a2 = 0.035_900_923_83;
    let a3 = 0.037_425_637_13;
    let a4 = 0.014_511_962_12;
    let b0 = 0.5;
    let b1 = 0.124_985_935_97;
    let b2 = 0.068_802_485_76;
    let b3 = 0.033_283_553_46;
    let b4 = 0.004_417_870_12;
    let poly_a = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)));
    let poly_b = b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)));
    poly_a - poly_b * m1.ln()
}

/// Complete elliptic integral E(m) via Abramowitz & Stegun polynomial approximation (17.3.36).
/// Argument m is the parameter. Valid for 0 ≤ m ≤ 1.
/// Maximum absolute error ~2e-8.
fn elliptic_e(m: f64) -> f64 {
    if m <= 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }
    if m >= 1.0 {
        return 1.0;
    }
    let m1 = 1.0 - m;
    let a1 = 0.443_251_414_09;
    let a2 = 0.062_606_012_20;
    let a3 = 0.047_573_835_46;
    let a4 = 0.017_365_064_51;
    let b1 = 0.249_968_730_16;
    let b2 = 0.092_001_800_37;
    let b3 = 0.040_696_975_26;
    let b4 = 0.005_264_496_39;
    let poly_a = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)));
    let poly_b = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)));
    poly_a - poly_b * m1.ln()
}

/// Disk stability initial conditions: Shu (1969) distribution function f(E, Lz)
/// for an axisymmetric disk with optional bulge, halo, and azimuthal perturbation.
pub struct DiskStabilityIC {
    /// Disk surface density Σ(R) as function of cylindrical radius.
    pub disk_surface_density: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Radial velocity dispersion σ_R(R).
    pub disk_velocity_dispersion: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Optional central bulge component.
    pub bulge: Option<Box<dyn IsolatedEquilibrium>>,
    /// Optional fixed dark matter halo potential Φ_halo(x).
    pub halo_potential: Option<Box<dyn Fn([f64; 3]) -> f64 + Send + Sync>>,
    /// Azimuthal mode number m (m=2 = bar, m=3 = triangle).
    pub perturbation_mode_m: u32,
    /// Pattern speed Ω_p in rad/time.
    pub perturbation_pattern_speed: Decimal,
    /// Relative amplitude δΣ/Σ.
    pub perturbation_amplitude: Decimal,
    // Cached f64 values for hot-path computation
    perturbation_pattern_speed_f64: f64,
    perturbation_amplitude_f64: f64,
}

impl DiskStabilityIC {
    /// Create a DiskStabilityIC from f64 parameters (backward-compatible).
    pub fn new(
        disk_surface_density: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        disk_velocity_dispersion: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        perturbation_mode_m: u32,
        perturbation_pattern_speed: f64,
        perturbation_amplitude: f64,
    ) -> Self {
        Self {
            disk_surface_density,
            disk_velocity_dispersion,
            bulge: None,
            halo_potential: None,
            perturbation_mode_m,
            perturbation_pattern_speed: Decimal::from_f64_retain(perturbation_pattern_speed)
                .unwrap_or(Decimal::ZERO),
            perturbation_amplitude: Decimal::from_f64_retain(perturbation_amplitude)
                .unwrap_or(Decimal::ZERO),
            perturbation_pattern_speed_f64: perturbation_pattern_speed,
            perturbation_amplitude_f64: perturbation_amplitude,
        }
    }

    /// Create a DiskStabilityIC from Decimal parameters (exact config).
    pub fn new_decimal(
        disk_surface_density: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        disk_velocity_dispersion: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        perturbation_mode_m: u32,
        perturbation_pattern_speed: Decimal,
        perturbation_amplitude: Decimal,
    ) -> Self {
        Self {
            disk_surface_density,
            disk_velocity_dispersion,
            bulge: None,
            halo_potential: None,
            perturbation_mode_m,
            perturbation_pattern_speed_f64: perturbation_pattern_speed.to_f64().unwrap_or(0.0),
            perturbation_amplitude_f64: perturbation_amplitude.to_f64().unwrap_or(0.0),
            perturbation_pattern_speed,
            perturbation_amplitude,
        }
    }

    /// Compute the circular velocity squared v_c²(R) from the combined disk+bulge+halo
    /// potential using the enclosed-mass approximation for the disk, plus optional
    /// bulge and halo contributions via finite-difference on their potentials.
    ///
    /// G = 1.0 (natural units).
    fn vc_squared(&self, radius: f64) -> f64 {
        let g = 1.0;
        let eps = 1e-6 * radius.max(1e-10);

        // Disk contribution: enclosed-mass approximation
        // v_c²(R) ≈ G * M_enc(R) / R where M_enc = 2π ∫₀^R Σ(R') R' dR'
        let n_quad = 200;
        let dr = radius / n_quad as f64;
        let mut m_enc = 0.0;
        for i in 0..n_quad {
            let r_mid = (i as f64 + 0.5) * dr;
            m_enc += (self.disk_surface_density)(r_mid) * r_mid * dr;
        }
        m_enc *= 2.0 * std::f64::consts::PI;
        let mut vc2 = if radius > 1e-30 {
            g * m_enc / radius
        } else {
            0.0
        };

        // Bulge contribution: v_c² = -R dΦ/dR via finite difference
        if let Some(ref bulge) = self.bulge {
            let phi_plus = bulge.potential(radius + eps);
            let phi_minus = bulge.potential((radius - eps).max(1e-30));
            let dphi_dr = (phi_plus - phi_minus) / (2.0 * eps);
            vc2 += -radius * dphi_dr;
        }

        // Halo contribution: v_c² = -R dΦ/dR via finite difference on Φ(R,0,0)
        if let Some(ref halo_pot) = self.halo_potential {
            let phi_plus = halo_pot([radius + eps, 0.0, 0.0]);
            let phi_minus = halo_pot([(radius - eps).max(1e-30), 0.0, 0.0]);
            let dphi_dr = (phi_plus - phi_minus) / (2.0 * eps);
            vc2 += -radius * dphi_dr;
        }

        vc2.max(0.0)
    }

    /// Compute the total potential Φ(R, z) using the spherical enclosed-mass approximation
    /// for the disk plus optional bulge and halo contributions.
    fn total_potential(&self, r_cyl: f64, z: f64, x: [f64; 3]) -> f64 {
        let g = 1.0;
        let r_sph = (r_cyl * r_cyl + z * z).sqrt();

        // Disk: spherical approximation Φ(r) ≈ -G * M_enc(r) / r
        let n_quad = 200;
        let r_int = r_sph.max(1e-30);
        let dr = r_int / n_quad as f64;
        let mut m_enc = 0.0;
        for i in 0..n_quad {
            let r_mid = (i as f64 + 0.5) * dr;
            m_enc += (self.disk_surface_density)(r_mid) * r_mid * dr;
        }
        m_enc *= 2.0 * std::f64::consts::PI;
        let mut phi = if r_sph > 1e-30 {
            -g * m_enc / r_sph
        } else {
            0.0
        };

        // Bulge contribution
        if let Some(ref bulge) = self.bulge {
            phi += bulge.potential(r_sph);
        }

        // Halo contribution
        if let Some(ref halo_pot) = self.halo_potential {
            phi += halo_pot(x);
        }

        phi
    }

    /// Find the guiding-center radius R_c for a given angular momentum L_z
    /// by bisection: solve L_z = R_c * v_c(R_c).
    fn find_guiding_radius(&self, lz: f64, r_max: f64) -> f64 {
        // L_z = R_c * v_c(R_c) => L_z² = R_c² * v_c²(R_c)
        // Find R_c such that R_c * sqrt(v_c²(R_c)) = L_z
        let mut lo = 1e-10_f64;
        let mut hi = r_max;

        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let vc2 = self.vc_squared(mid);
            let lz_mid = mid * vc2.sqrt();
            if lz_mid < lz {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Toomre Q(R) = σ_R κ / (3.36 G Σ). Q > 1 means locally stable.
    ///
    /// Computes the epicyclic frequency κ(R) = √(R dΩ²/dR + 4Ω²)
    /// from the combined disk+bulge+halo circular velocity curve.
    pub fn toomre_q(&self, radius: f64) -> f64 {
        let g = 1.0;
        let sigma_r = (self.disk_velocity_dispersion)(radius);
        let sigma_surf = (self.disk_surface_density)(radius);

        if sigma_surf.abs() < 1e-30 {
            return f64::INFINITY; // no surface density → trivially stable
        }

        // Compute Ω²(R) = v_c²(R) / R²
        let vc2 = self.vc_squared(radius);
        let omega2 = if radius > 1e-30 {
            vc2 / (radius * radius)
        } else {
            return f64::INFINITY;
        };

        // Epicyclic frequency: κ² = R dΩ²/dR + 4Ω²
        // Compute dΩ²/dR by finite difference
        let eps = 1e-4 * radius.max(1e-8);
        let r_plus = radius + eps;
        let r_minus = (radius - eps).max(1e-30);
        let vc2_plus = self.vc_squared(r_plus);
        let vc2_minus = self.vc_squared(r_minus);
        let omega2_plus = vc2_plus / (r_plus * r_plus);
        let omega2_minus = vc2_minus / (r_minus * r_minus);
        let domega2_dr = (omega2_plus - omega2_minus) / (r_plus - r_minus);

        let kappa2 = radius * domega2_dr + 4.0 * omega2;
        if kappa2 <= 0.0 {
            return 0.0; // negative κ² signals pathological rotation curve
        }
        let kappa = kappa2.sqrt();

        // Q = σ_R * κ / (3.36 * G * Σ)
        sigma_r * kappa / (3.36 * g * sigma_surf)
    }

    /// Sample onto 6D grid: construct f(E, Lz) for disk using the Shu (1969)
    /// distribution function, then superpose an azimuthal perturbation mode.
    ///
    /// Shu DF: f(E, L_z) = [Σ(R_c) Ω(R_c)] / [π κ(R_c) σ_R²(R_c)]
    ///                      × exp[-(E - E_c(L_z)) / σ_R²(R_c)]
    ///
    /// for L_z > 0 (prograde orbits). f = 0 for retrograde (L_z ≤ 0).
    ///
    /// The vertical structure uses a sech²(z/z_0) profile with
    /// z_0 = σ_z / √(4πGΣ), assuming σ_z ≈ σ_R (isotropic approximation).
    pub fn sample_on_grid(
        &self,
        domain: &Domain,
        progress: Option<&crate::tooling::core::progress::StepProgress>,
    ) -> PhaseSpaceSnapshot {
        let g = 1.0;

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

        // Maximum radius for guiding-center search
        let r_max = (lx[0] * lx[0] + lx[1] * lx[1]).sqrt() * 1.5;

        // Row-major strides (same layout as UniformGrid6D)
        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
        let mut data = vec![0.0f64; total];

        let perturbation_m = self.perturbation_mode_m;
        let perturbation_amp = self.perturbation_amplitude_f64;

        let counter = std::sync::atomic::AtomicU64::new(0);
        let report_interval = (nx1 / 100).max(1) as u64;

        // Establish 0% baseline so the TUI doesn't jump to a non-zero first value
        if let Some(p) = progress {
            p.set_intra_progress(0, nx1 as u64);
        }

        data.par_chunks_mut(s_x1)
            .enumerate()
            .for_each(|(ix1, chunk)| {
                let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
                for ix2 in 0..nx2 {
                    let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];

                    let r_cyl = (x1 * x1 + x2 * x2).sqrt();
                    let phi_azimuthal = x2.atan2(x1);

                    // Perturbation factor: 1 + ε cos(m φ)
                    let pert_factor =
                        1.0 + perturbation_amp * (perturbation_m as f64 * phi_azimuthal).cos();

                    for ix3 in 0..nx3 {
                        let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                        let pos = [x1, x2, x3];

                        let phi_total = self.total_potential(r_cyl, x3, pos);

                        // Vertical sech² profile: f_z(z) = sech²(z/z_0) / (2 z_0)
                        // z_0 = σ_z / √(4πGΣ)
                        let sigma_r_here = (self.disk_velocity_dispersion)(r_cyl);
                        let sigma_surf_here = (self.disk_surface_density)(r_cyl);
                        let sigma_z = sigma_r_here; // isotropic approximation
                        let z_0 = if sigma_surf_here > 1e-30 {
                            sigma_z / (4.0 * std::f64::consts::PI * g * sigma_surf_here).sqrt()
                        } else {
                            1e10 // effectively infinite scale height if no surface density
                        };
                        let sech_arg = x3 / z_0;
                        let f_z = if sech_arg.abs() < 50.0 {
                            let cosh_val = sech_arg.cosh();
                            1.0 / (cosh_val * cosh_val * 2.0 * z_0)
                        } else {
                            0.0 // negligible contribution
                        };

                        if f_z < 1e-30 {
                            // Skip velocity loops if vertical contribution is negligible
                            continue;
                        }

                        let base = ix2 * s_x2 + ix3 * s_x3;

                        for iv1 in 0..nv1 {
                            let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                            for iv2 in 0..nv2 {
                                let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                                for iv3 in 0..nv3 {
                                    let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];

                                    // Angular momentum L_z = R * v_φ
                                    // v_φ = (v2*x1 - v1*x2) / R
                                    let lz = if r_cyl > 1e-30 {
                                        v2 * x1 - v1 * x2 // = R * v_φ
                                    } else {
                                        0.0
                                    };

                                    // Prograde disk only
                                    if lz <= 0.0 {
                                        continue;
                                    }

                                    // Find guiding-center radius R_c(L_z)
                                    let r_c = self.find_guiding_radius(lz, r_max);

                                    // Properties at guiding center
                                    let vc2_rc = self.vc_squared(r_c);
                                    let vc_rc = vc2_rc.sqrt();
                                    let omega_rc = if r_c > 1e-30 {
                                        vc_rc / r_c
                                    } else {
                                        continue;
                                    };

                                    let sigma_r_rc = (self.disk_velocity_dispersion)(r_c);
                                    let sigma_surf_rc = (self.disk_surface_density)(r_c);
                                    let sigma_r2_rc = sigma_r_rc * sigma_r_rc;

                                    if sigma_r2_rc < 1e-30 || sigma_surf_rc < 1e-30 {
                                        continue;
                                    }

                                    // Epicyclic frequency at R_c
                                    let eps_fd = 1e-4 * r_c.max(1e-8);
                                    let r_cp = r_c + eps_fd;
                                    let r_cm = (r_c - eps_fd).max(1e-30);
                                    let omega2_rc = omega_rc * omega_rc;
                                    let omega2_plus = self.vc_squared(r_cp) / (r_cp * r_cp);
                                    let omega2_minus = self.vc_squared(r_cm) / (r_cm * r_cm);
                                    let domega2_dr = (omega2_plus - omega2_minus) / (r_cp - r_cm);
                                    let kappa2_rc = r_c * domega2_dr + 4.0 * omega2_rc;
                                    if kappa2_rc <= 0.0 {
                                        continue;
                                    }
                                    let kappa_rc = kappa2_rc.sqrt();

                                    // Circular orbit energy at R_c:
                                    // E_c = ½ v_c(R_c)² + Φ(R_c, 0)
                                    let phi_rc = self.total_potential(r_c, 0.0, [r_c, 0.0, 0.0]);
                                    let e_c = 0.5 * vc2_rc + phi_rc;

                                    // Total energy E = ½ v² + Φ(x)
                                    let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                                    let energy = 0.5 * v2sq + phi_total;

                                    // Shu DF: only for bound orbits near the guiding center
                                    let de = energy - e_c;
                                    if de > 0.0 {
                                        // Unbound relative to circular orbit
                                        continue;
                                    }

                                    // Exponential argument: -(E - E_c) / σ_R²
                                    let arg = -de / sigma_r2_rc;
                                    if arg > 500.0 {
                                        continue; // overflow guard
                                    }

                                    // Shu DF (in-plane part):
                                    // f_shu = Σ(R_c) Ω(R_c) / (π κ(R_c) σ_R²(R_c))
                                    //         × exp(-(E-E_c)/σ_R²(R_c))
                                    let f_shu = sigma_surf_rc * omega_rc
                                        / (std::f64::consts::PI * kappa_rc * sigma_r2_rc)
                                        * arg.exp();

                                    // Combined: f = f_shu(E, Lz) × f_z(z) × perturbation
                                    let f_val = f_shu * f_z * pert_factor;

                                    if f_val > 0.0 {
                                        chunk[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f_val;
                                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};

    #[test]
    fn toomre_q_exponential_disk() {
        // Exponential disk: Σ(R) = Σ₀ exp(-R/Rd), σ_R(R) = σ₀ exp(-R/(2Rd))
        let rd = 1.0;
        let sigma_0 = 0.5;
        let sigma_d0 = 0.3;
        let ic = DiskStabilityIC::new(
            Box::new(move |r: f64| sigma_0 * (-r / rd).exp()),
            Box::new(move |r: f64| sigma_d0 * (-r / (2.0 * rd)).exp()),
            2,
            0.0,
            0.0,
        );
        let q = ic.toomre_q(2.0);
        assert!(q.is_finite(), "Toomre Q should be finite, got {q}");
        assert!(q > 0.0, "Toomre Q should be positive, got {q}");
        println!("Toomre Q at R=2: {q:.4}");
    }

    #[test]
    fn disk_stability_sample_on_grid() {
        let rd = 3.0;
        let sigma_0 = 0.5;
        let sigma_d0 = 0.3;
        let ic = DiskStabilityIC::new(
            Box::new(move |r: f64| sigma_0 * (-r / rd).exp()),
            Box::new(move |r: f64| sigma_d0 * (-r / (2.0 * rd)).exp()),
            2,
            0.1,
            0.05,
        );
        let domain = Domain::builder()
            .spatial_extent(8.0)
            .velocity_extent(2.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();
        let snap = ic.sample_on_grid(&domain, None);
        assert_eq!(snap.shape, [8, 8, 8, 8, 8, 8]);
        assert!(snap.data.iter().all(|v| v.is_finite()));
        assert!(
            snap.data.iter().any(|&v| v > 0.0),
            "should have some nonzero values"
        );
    }
}
