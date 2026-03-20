//! Isolated equilibrium initial conditions: Plummer, King, Hernquist, NFW.
//! All specified as f(E) or f(E,L) — integrals of motion.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use rayon::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Trait for isolated equilibrium models specified as a distribution function.
pub trait IsolatedEquilibrium {
    /// Distribution function f(E, L). E = specific energy, L = specific angular momentum.
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64;
    /// Density profile ρ(r) in real space.
    fn density_profile(&self, r: f64) -> f64;
    /// Gravitational potential Φ(r) (spherically symmetric, includes factor G).
    fn potential(&self, r: f64) -> f64;
}

/// Plummer sphere: f(E) ∝ (−E)^(7/2).
/// Analytic DF; density ρ ∝ (r²+a²)^(−5/2).
pub struct PlummerIC {
    pub mass: Decimal,
    pub scale_radius: Decimal,
    pub g: Decimal,
    // Cached f64 values for hot-path computation
    mass_f64: f64,
    scale_radius_f64: f64,
    g_f64: f64,
}

impl PlummerIC {
    /// Create a Plummer IC from f64 parameters (backward-compatible).
    pub fn new(mass: f64, scale_radius: f64, g: f64) -> Self {
        Self {
            mass: Decimal::from_f64_retain(mass).unwrap(),
            scale_radius: Decimal::from_f64_retain(scale_radius).unwrap(),
            g: Decimal::from_f64_retain(g).unwrap(),
            mass_f64: mass,
            scale_radius_f64: scale_radius,
            g_f64: g,
        }
    }

    /// Create a Plummer IC from Decimal parameters (exact config).
    pub fn new_decimal(mass: Decimal, scale_radius: Decimal, g: Decimal) -> Self {
        Self {
            mass_f64: mass.to_f64().unwrap(),
            scale_radius_f64: scale_radius.to_f64().unwrap(),
            g_f64: g.to_f64().unwrap(),
            mass,
            scale_radius,
            g,
        }
    }
}

impl IsolatedEquilibrium for PlummerIC {
    fn distribution_function(&self, energy: f64, _angular_momentum: f64) -> f64 {
        if energy >= 0.0 {
            return 0.0;
        }
        use std::f64::consts::PI;
        let a = self.scale_radius_f64;
        let m = self.mass_f64;
        let g = self.g_f64;
        // Binney & Tremaine §4.4.3 (Eddington inversion of Plummer density):
        // In natural units G=M=a=1: f(E) = (24√2/7π³) * (-E)^(7/2)
        // In physical units: f(E) = (24√2/7π³) * M * (-E)^(7/2) / (a³ * (GM/a)^5)
        let e0 = g * m / a;
        let prefactor = (24.0 * 2.0_f64.sqrt()) / (7.0 * PI.powi(3));
        prefactor * m * (-energy).powf(3.5) / (a.powi(3) * e0.powf(5.0))
    }

    fn density_profile(&self, r: f64) -> f64 {
        use std::f64::consts::PI;
        let a = self.scale_radius_f64;
        let m = self.mass_f64;
        3.0 * m / (4.0 * PI * a.powi(3)) * (1.0 + r * r / (a * a)).powf(-2.5)
    }

    fn potential(&self, r: f64) -> f64 {
        -self.g_f64 * self.mass_f64 / (r * r + self.scale_radius_f64 * self.scale_radius_f64).sqrt()
    }
}

/// King (1966) lowered Maxwellian: f(E) ∝ (e^((E₀−E)/σ²) − 1) for E < E₀.
/// Requires solving the Poisson-Boltzmann ODE from r=0 outward to find the
/// tidal radius r_t where Φ(r_t) = E₀.
pub struct KingIC {
    pub mass: Decimal,
    /// Dimensionless King concentration W₀ = Φ(0)/σ².
    pub king_parameter_w0: Decimal,
    pub velocity_dispersion: Decimal,
    pub g: Decimal,
    // Cached f64 values for hot-path computation
    mass_f64: f64,
    king_parameter_w0_f64: f64,
    velocity_dispersion_f64: f64,
    g_f64: f64,
    /// Tabulated radius values from ODE integration.
    r_table: Vec<f64>,
    /// Tabulated density ρ(r) from ODE integration.
    rho_table: Vec<f64>,
    /// Tabulated potential Φ(r) from ODE integration (dimensionless W(r)).
    phi_table: Vec<f64>,
    /// Normalisation constant A for the DF.
    norm_a: f64,
    /// Tidal (boundary) energy E₀ = Φ(r_t).
    e0: f64,
}

/// Single RK4 step for (y, dy/dr) given f(r, y, dy).
fn rk4_step(r: f64, y: f64, dy: f64, h: f64, rhs: &dyn Fn(f64, f64, f64) -> f64) -> (f64, f64) {
    let k1y = dy;
    let k1d = rhs(r, y, dy);
    let k2y = dy + 0.5 * h * k1d;
    let k2d = rhs(r + 0.5 * h, y + 0.5 * h * k1y, dy + 0.5 * h * k1d);
    let k3y = dy + 0.5 * h * k2d;
    let k3d = rhs(r + 0.5 * h, y + 0.5 * h * k2y, dy + 0.5 * h * k2d);
    let k4y = dy + h * k3d;
    let k4d = rhs(r + h, y + h * k3y, dy + h * k3d);
    let y_new = y + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    let dy_new = dy + h / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d);
    (y_new, dy_new)
}

impl KingIC {
    /// Create a King model with the given total mass, concentration W₀, core radius, and G.
    ///
    /// The `scale_radius` sets the physical King core radius r₀. The velocity
    /// dispersion σ is derived self-consistently from the Poisson-Boltzmann ODE
    /// so that 4πGρ₀ r₀² = 9σ².
    pub fn new(mass: f64, king_parameter_w0: f64, scale_radius: f64, g: f64) -> Self {
        use std::f64::consts::PI;
        let w0 = king_parameter_w0;
        let r0 = scale_radius; // physical King core radius

        // Solve the dimensionless Poisson-Boltzmann equation:
        //   W''(ξ) + (2/ξ)W'(ξ) = -9 * king_rho(W)
        // where ξ = r/r₀ is the dimensionless radius,
        // W = (E₀ − Φ)/σ² is the dimensionless potential,
        // and king_rho(W) = e^W erf(√W) − √(4W/π)(1 + 2W/3).
        // Boundary conditions: W(0) = W₀, W'(0) = 0.

        let n_steps = 10000;
        let r_max_guess = 100.0; // will stop when W <= 0
        let h = r_max_guess / n_steps as f64;

        let king_rho = |w: f64| -> f64 {
            if w <= 0.0 {
                return 0.0;
            }
            let sw = w.sqrt();
            // ρ_king(W) ∝ e^W erf(√W) - √(4W/π)(1 + 2W/3)
            let erf_val = erf_approx(sw);
            let term1 = w.exp() * erf_val;
            let term2 = (4.0 * w / PI).sqrt() * (1.0 + 2.0 * w / 3.0);
            (term1 - term2).max(0.0)
        };

        let mut r_table = Vec::with_capacity(n_steps);
        let mut w_table = Vec::with_capacity(n_steps);

        let mut r = 1e-10; // avoid r=0 singularity
        let mut w = w0;
        let mut dw = 0.0; // W'(0) = 0

        r_table.push(0.0);
        w_table.push(w0);

        // ODE: W'' = -2/ξ * W' - 9 * king_rho(W)/king_rho(W₀)
        // The density must be normalized by the central value so that
        // ρ̃(W₀) = 1, giving the standard King dimensionless equation.
        let rho_w0 = king_rho(w0);
        let rhs = |r: f64, w: f64, dw: f64| -> f64 {
            if r < 1e-30 {
                // L'Hôpital: at ξ=0, W'' = -3 × ρ̃(W) = -3 × king_rho(W)/king_rho(W₀)
                return -3.0 * king_rho(w) / rho_w0;
            }
            -2.0 / r * dw - 9.0 * king_rho(w) / rho_w0
        };

        for _ in 0..n_steps {
            let (w_new, dw_new) = rk4_step(r, w, dw, h, &rhs);
            r += h;
            w = w_new;
            dw = dw_new;
            r_table.push(r);
            w_table.push(w.max(0.0));
            if w <= 0.0 {
                break;
            }
        }

        // The ODE was solved in dimensionless units ξ = r/r₀ where r₀ is the
        // King core radius. The coefficient 9 in the ODE comes from choosing
        // ξ = r/r₀ with 4πGρ₀ r₀² = 9σ².
        //
        // Given r₀ = scale_radius (user parameter), we derive σ and ρ₀.

        // Dimensionless mass integral: M_d = ∫₀^{ξ_t} 4πξ² king_rho(W(ξ)) dξ
        let mut mass_integral_d = 0.0;
        for i in 1..r_table.len() {
            let xi_mid = 0.5 * (r_table[i - 1] + r_table[i]);
            let rho_mid = king_rho(0.5 * (w_table[i - 1] + w_table[i]));
            let dxi = r_table[i] - r_table[i - 1];
            mass_integral_d += 4.0 * PI * xi_mid * xi_mid * rho_mid * dxi;
        }

        // Physical central density from mass normalization:
        //   M = (ρ₀/king_rho(W₀)) × r₀³ × M_d
        //   → ρ₀ = M × king_rho(W₀) / (r₀³ × M_d)
        let king_rho_w0 = king_rho(w0);
        let rho_0 = if mass_integral_d > 1e-30 && king_rho_w0 > 1e-30 {
            mass * king_rho_w0 / (r0 * r0 * r0 * mass_integral_d)
        } else {
            1.0
        };

        // Velocity dispersion from self-consistency: 4πGρ₀ r₀² = 9σ²
        let sigma = (4.0 * PI * g * rho_0 * r0 * r0 / 9.0).sqrt();

        // Density scale: ρ(r) = rho_scale × king_rho(W(r/r₀))
        let rho_scale = rho_0 / king_rho_w0;

        // Convert r_table from dimensionless ξ to physical r = ξ × r₀
        let r_table: Vec<f64> = r_table.iter().map(|&xi| xi * r0).collect();

        // Physical density table
        let rho_table: Vec<f64> = w_table.iter().map(|&w| rho_scale * king_rho(w)).collect();

        // Physical potential: Φ(r) = -σ² W(r/r₀), with Φ(r_t) = 0 (tidal boundary)
        let phi_table: Vec<f64> = w_table.iter().map(|&w| -sigma * sigma * w).collect();
        let e0 = 0.0; // E₀ = Φ(r_t) = 0

        // DF normalisation: ρ(r) = (2πσ²)^{3/2} A × king_rho(W)
        // → A = rho_scale / (2πσ²)^{3/2}
        let norm_a = rho_scale / (2.0 * PI * sigma * sigma).powf(1.5);

        Self {
            mass: Decimal::from_f64_retain(mass).unwrap(),
            king_parameter_w0: Decimal::from_f64_retain(king_parameter_w0).unwrap(),
            velocity_dispersion: Decimal::from_f64_retain(sigma).unwrap(),
            g: Decimal::from_f64_retain(g).unwrap(),
            mass_f64: mass,
            king_parameter_w0_f64: king_parameter_w0,
            velocity_dispersion_f64: sigma,
            g_f64: g,
            r_table,
            rho_table,
            phi_table,
            norm_a,
            e0,
        }
    }

    /// Create a King model from Decimal parameters (exact config).
    pub fn new_decimal(
        mass: Decimal,
        king_parameter_w0: Decimal,
        scale_radius: Decimal,
        g: Decimal,
    ) -> Self {
        Self::new(
            mass.to_f64().unwrap(),
            king_parameter_w0.to_f64().unwrap(),
            scale_radius.to_f64().unwrap(),
            g.to_f64().unwrap(),
        )
    }
}

/// Approximate error function (Abramowitz & Stegun 7.1.26, max error ~1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

impl IsolatedEquilibrium for KingIC {
    fn distribution_function(&self, energy: f64, _angular_momentum: f64) -> f64 {
        if energy >= self.e0 {
            return 0.0;
        }
        let sigma2 = self.velocity_dispersion_f64 * self.velocity_dispersion_f64;
        let arg = (self.e0 - energy) / sigma2;
        if arg > 500.0 {
            return 0.0; // overflow guard
        }
        self.norm_a * (arg.exp() - 1.0)
    }

    fn density_profile(&self, r: f64) -> f64 {
        interpolate_table(&self.r_table, &self.rho_table, r)
    }

    fn potential(&self, r: f64) -> f64 {
        interpolate_table(&self.r_table, &self.phi_table, r)
    }
}

/// Linear interpolation on a sorted table.
fn interpolate_table(x_table: &[f64], y_table: &[f64], x: f64) -> f64 {
    if x_table.is_empty() {
        return 0.0;
    }
    if x <= x_table[0] {
        return y_table[0];
    }
    if x >= *x_table.last().unwrap() {
        return *y_table.last().unwrap();
    }
    // Binary search for interval
    let mut lo = 0;
    let mut hi = x_table.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x_table[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - x_table[lo]) / (x_table[hi] - x_table[lo]);
    y_table[lo] + t * (y_table[hi] - y_table[lo])
}

/// Hernquist (1990): ρ ∝ 1/(r(r+a)³). More realistic cuspy halo profile.
/// Closed-form DF from Hernquist 1990, eq. 17.
pub struct HernquistIC {
    pub mass: Decimal,
    pub scale_radius: Decimal,
    pub g: Decimal,
    // Cached f64 values for hot-path computation
    mass_f64: f64,
    scale_radius_f64: f64,
    g_f64: f64,
}

impl HernquistIC {
    /// Create a Hernquist IC from f64 parameters (backward-compatible).
    pub fn new(mass: f64, scale_radius: f64, g: f64) -> Self {
        Self {
            mass: Decimal::from_f64_retain(mass).unwrap(),
            scale_radius: Decimal::from_f64_retain(scale_radius).unwrap(),
            g: Decimal::from_f64_retain(g).unwrap(),
            mass_f64: mass,
            scale_radius_f64: scale_radius,
            g_f64: g,
        }
    }

    /// Create a Hernquist IC from Decimal parameters (exact config).
    pub fn new_decimal(mass: Decimal, scale_radius: Decimal, g: Decimal) -> Self {
        Self {
            mass_f64: mass.to_f64().unwrap(),
            scale_radius_f64: scale_radius.to_f64().unwrap(),
            g_f64: g.to_f64().unwrap(),
            mass,
            scale_radius,
            g,
        }
    }
}

impl IsolatedEquilibrium for HernquistIC {
    fn distribution_function(&self, energy: f64, _angular_momentum: f64) -> f64 {
        if energy >= 0.0 {
            return 0.0;
        }
        use std::f64::consts::PI;
        let m = self.mass_f64;
        let a = self.scale_radius_f64;
        let g = self.g_f64;
        // Hernquist (1990), eq. 17:
        // f(E) = M / (8√2·π³·(GMa)^{3/2}) · 1/(1-q²)^{5/2}
        //        · [3·arcsin(q) + q√(1-q²)·(1-2q²)·(8q⁴-8q²-3)]
        // where q = √(-E·a/(GM))
        let gm = g * m;
        let q2 = -energy * a / gm;
        if q2 <= 0.0 || q2 >= 1.0 {
            return 0.0;
        }
        let q = q2.sqrt();
        let omq2 = 1.0 - q2; // 1 - q²
        let sqrt_omq2 = omq2.sqrt();
        let prefactor = m / (8.0 * 2.0_f64.sqrt() * PI.powi(3) * (gm * a).powf(1.5));
        let bracket =
            3.0 * q.asin() + q * sqrt_omq2 * (1.0 - 2.0 * q2) * (8.0 * q2 * q2 - 8.0 * q2 - 3.0);
        let f = prefactor * bracket / omq2.powf(2.5);
        f.max(0.0)
    }

    fn density_profile(&self, r: f64) -> f64 {
        let a = self.scale_radius_f64;
        let m = self.mass_f64;
        m * a / (2.0 * std::f64::consts::PI * r * (r + a).powi(3))
    }

    fn potential(&self, r: f64) -> f64 {
        -self.g_f64 * self.mass_f64 / (r + self.scale_radius_f64)
    }
}

/// NFW (Navarro-Frenk-White): ρ ∝ 1/(r/r_s · (1+r/r_s)²).
/// DF computed via numerical Eddington inversion.
pub struct NfwIC {
    pub mass: Decimal,
    pub scale_radius: Decimal,
    pub concentration: Decimal,
    pub g: Decimal,
    // Cached f64 values for hot-path computation
    mass_f64: f64,
    scale_radius_f64: f64,
    concentration_f64: f64,
    g_f64: f64,
    /// Characteristic density ρ_s (derived from mass, r_s, concentration).
    rho_s: f64,
    /// Tabulated DF: (E, f(E)) pairs.
    df_table_e: Vec<f64>,
    df_table_f: Vec<f64>,
}

impl NfwIC {
    /// Create an NFW IC from f64 parameters (backward-compatible).
    pub fn new(mass: f64, scale_radius: f64, concentration: f64, g: f64) -> Self {
        use std::f64::consts::PI;
        let rs = scale_radius;
        let c = concentration;

        // ρ_s from mass within r_vir = c·r_s:
        // M = 4π ρ_s r_s³ [ln(1+c) - c/(1+c)]
        let ln_factor = (1.0 + c).ln() - c / (1.0 + c);
        let rho_s = mass / (4.0 * PI * rs.powi(3) * ln_factor);

        // Build DF table via Eddington inversion:
        // f(E) = 1/(√8·π²) ∫₀ᴱ (d²ρ/dΦ²) / √(E-Φ) dΦ
        // We need ρ(Φ) by inverting Φ(r), then d²ρ/dΦ² numerically.

        // 1. Tabulate r, Φ(r), ρ(r) from r_min to r_max
        let r_vir = c * rs;
        let r_min = 0.001 * rs;
        let r_max = 10.0 * r_vir;
        let n_tab = 2000;

        let phi_of_r =
            |r: f64| -> f64 { -4.0 * PI * g * rho_s * rs.powi(3) * (1.0 + r / rs).ln() / r };
        let rho_of_r = |r: f64| -> f64 {
            let x = r / rs;
            rho_s / (x * (1.0 + x).powi(2))
        };

        // Build (Φ, ρ) table sorted by Φ (Φ increases with r, going from negative to 0)
        let mut phi_rho: Vec<(f64, f64)> = (0..n_tab)
            .map(|i| {
                let log_r =
                    (r_min.ln()) + (r_max.ln() - r_min.ln()) * i as f64 / (n_tab - 1) as f64;
                let r = log_r.exp();
                (phi_of_r(r), rho_of_r(r))
            })
            .collect();
        phi_rho.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Remove duplicates in Φ
        phi_rho.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-30);

        let phi_tab: Vec<f64> = phi_rho.iter().map(|&(p, _)| p).collect();
        let rho_tab: Vec<f64> = phi_rho.iter().map(|&(_, r)| r).collect();

        // 2. Compute d²ρ/dΦ² numerically (central differences)
        let n = phi_tab.len();
        let mut d2rho_dphi2 = vec![0.0f64; n];
        for i in 1..n - 1 {
            let dp1 = phi_tab[i + 1] - phi_tab[i];
            let dp0 = phi_tab[i] - phi_tab[i - 1];
            let drho1 = (rho_tab[i + 1] - rho_tab[i]) / dp1;
            let drho0 = (rho_tab[i] - rho_tab[i - 1]) / dp0;
            d2rho_dphi2[i] = 2.0 * (drho1 - drho0) / (dp1 + dp0);
        }

        // 3. Eddington integral: f(E) = 1/(√8·π²) ∫_{Φ_min}^{E} d²ρ/dΦ² / √(E-Φ) dΦ
        let phi_min = phi_tab[0];
        let phi_max = *phi_tab.last().unwrap();
        let n_e = 200;
        let mut df_table_e = Vec::with_capacity(n_e);
        let mut df_table_f = Vec::with_capacity(n_e);

        for ie in 0..n_e {
            let e = phi_min + (phi_max - phi_min) * (ie as f64 + 0.5) / n_e as f64;
            // Integrate from phi_min to E
            let mut integral = 0.0;
            for i in 1..n {
                if phi_tab[i] > e {
                    break;
                }
                let phi_mid = 0.5 * (phi_tab[i - 1] + phi_tab[i]);
                let d2 = 0.5 * (d2rho_dphi2[i - 1] + d2rho_dphi2[i]);
                let dphi = phi_tab[i] - phi_tab[i - 1];
                let diff = e - phi_mid;
                if diff > 0.0 {
                    integral += d2 / diff.sqrt() * dphi;
                }
            }
            let f_e = integral / (8.0_f64.sqrt() * PI * PI);
            df_table_e.push(e);
            df_table_f.push(f_e.max(0.0));
        }

        Self {
            mass: Decimal::from_f64_retain(mass).unwrap(),
            scale_radius: Decimal::from_f64_retain(scale_radius).unwrap(),
            concentration: Decimal::from_f64_retain(concentration).unwrap(),
            g: Decimal::from_f64_retain(g).unwrap(),
            mass_f64: mass,
            scale_radius_f64: scale_radius,
            concentration_f64: concentration,
            g_f64: g,
            rho_s,
            df_table_e,
            df_table_f,
        }
    }

    /// Create an NFW IC from Decimal parameters (exact config).
    pub fn new_decimal(
        mass: Decimal,
        scale_radius: Decimal,
        concentration: Decimal,
        g: Decimal,
    ) -> Self {
        Self::new(
            mass.to_f64().unwrap(),
            scale_radius.to_f64().unwrap(),
            concentration.to_f64().unwrap(),
            g.to_f64().unwrap(),
        )
    }
}

impl IsolatedEquilibrium for NfwIC {
    fn distribution_function(&self, energy: f64, _angular_momentum: f64) -> f64 {
        if energy >= 0.0 || self.df_table_e.is_empty() {
            return 0.0;
        }
        interpolate_table(&self.df_table_e, &self.df_table_f, energy).max(0.0)
    }

    fn density_profile(&self, r: f64) -> f64 {
        let x = r / self.scale_radius_f64;
        if x < 1e-30 {
            return self.rho_s * 1e30; // diverges at r=0
        }
        self.rho_s / (x * (1.0 + x).powi(2))
    }

    fn potential(&self, r: f64) -> f64 {
        use std::f64::consts::PI;
        if r < 1e-30 {
            return -4.0
                * PI
                * self.g_f64
                * self.rho_s
                * self.scale_radius_f64
                * self.scale_radius_f64
                * 100.0;
        }
        -4.0 * PI
            * self.g_f64
            * self.rho_s
            * self.scale_radius_f64.powi(3)
            * (1.0 + r / self.scale_radius_f64).ln()
            / r
    }
}

/// Sample an isolated equilibrium IC onto the 6D grid.
/// Evaluates f(E(x,v)) = f(½v² + Φ(r)) on every (x,v) grid point.
/// Parallelized over the outer spatial dimension (ix1) using rayon.
pub fn sample_on_grid(
    ic: &(dyn IsolatedEquilibrium + Sync),
    domain: &Domain,
) -> PhaseSpaceSnapshot {
    sample_on_grid_with_progress(ic, domain, None)
}

/// Like `sample_on_grid`, but reports intra-phase progress via the optional
/// `StepProgress` handle so the TUI can show a cell-level progress bar.
pub fn sample_on_grid_with_progress(
    ic: &(dyn IsolatedEquilibrium + Sync),
    domain: &Domain,
    progress: Option<&crate::tooling::core::progress::StepProgress>,
) -> PhaseSpaceSnapshot {
    use std::sync::atomic::{AtomicU64, Ordering};

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

    // Row-major strides (same layout as UniformGrid6D)
    let s_v3 = 1usize;
    let s_v2 = nv3;
    let s_v1 = nv2 * nv3;
    let s_x3 = nv1 * s_v1;
    let s_x2 = nx3 * s_x3;
    let s_x1 = nx2 * s_x2;

    let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
    let mut data = vec![0.0f64; total];

    let counter = AtomicU64::new(0);
    let report_interval = (nx1 / 100).max(1) as u64;

    // Establish 0% baseline so the TUI doesn't jump to a non-zero first value
    if let Some(p) = progress {
        p.set_intra_progress(0, nx1 as u64);
    }

    // Each ix1 slab is independent — parallelize over ix1
    data.par_chunks_mut(s_x1)
        .enumerate()
        .for_each(|(ix1, chunk)| {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            for ix2 in 0..nx2 {
                let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];
                for ix3 in 0..nx3 {
                    let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                    let r = (x1 * x1 + x2 * x2 + x3 * x3).sqrt();
                    let phi = ic.potential(r);
                    let base = ix2 * s_x2 + ix3 * s_x3;

                    for iv1 in 0..nv1 {
                        let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                                let energy = 0.5 * v2sq + phi;
                                let f = ic.distribution_function(energy, 0.0).max(0.0);
                                chunk[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f;
                            }
                        }
                    }
                }
            }

            if let Some(p) = progress {
                let c = counter.fetch_add(1, Ordering::Relaxed);
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
