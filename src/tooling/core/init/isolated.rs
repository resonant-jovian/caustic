//! Spherically symmetric equilibrium initial conditions for isolated systems.
//!
//! Provides [`IsolatedEquilibrium`] implementations for Plummer, King,
//! Hernquist, Isochrone, and NFW profiles.  Each model supplies a
//! distribution function f(E) or f(E,L) that depends only on integrals
//! of motion, together with the analytic density and potential profiles.
//! The [`sample_on_grid`] function evaluates f(1/2 v^2 + Phi(r)) on every
//! 6D grid point (rayon-parallelised) to produce a [`PhaseSpaceSnapshot`]
//! suitable for initialising a [`Simulation`](crate::Simulation).

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use rayon::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Trait for spherically symmetric equilibrium models specified as a
/// distribution function of the integrals of motion.
///
/// Implementors supply f(E, L), rho(r), and Phi(r).  These are consumed by
/// [`sample_on_grid`] to evaluate the IC on the full 6D phase-space grid.
pub trait IsolatedEquilibrium: Send + Sync {
    /// Evaluate the distribution function f(E, L).
    ///
    /// `energy` is the specific energy E = 1/2 v^2 + Phi(r).
    /// `angular_momentum` is the specific angular momentum magnitude |L|.
    /// Returns 0 for unbound orbits (E >= 0 or E >= E_0 for King models).
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64;
    /// Analytic (or tabulated) mass density profile rho(r).
    fn density_profile(&self, r: f64) -> f64;
    /// Gravitational potential Phi(r), including the factor G*M.
    fn potential(&self, r: f64) -> f64;
}

/// Plummer sphere initial conditions.
///
/// An isotropic model with analytic DF f(E) proportional to (-E)^(7/2) and a
/// cored density profile rho proportional to (r^2 + a^2)^(-5/2).  Widely used
/// as a smooth, non-singular test case for equilibrium preservation.
pub struct PlummerIC {
    /// Total mass of the system (Decimal for config precision).
    pub mass: Decimal,
    /// Plummer softening / scale radius `a`.
    pub scale_radius: Decimal,
    /// Gravitational constant G.
    pub g: Decimal,
    // Cached f64 values for hot-path computation
    mass_f64: f64,
    scale_radius_f64: f64,
    g_f64: f64,
}

impl PlummerIC {
    /// Create a Plummer IC from `f64` parameters.
    ///
    /// Internally converts to `Decimal` for config storage and caches `f64` for
    /// hot-path evaluation.
    pub fn new(mass: f64, scale_radius: f64, g: f64) -> Self {
        Self {
            mass: Decimal::from_f64_retain(mass).unwrap_or(Decimal::ZERO),
            scale_radius: Decimal::from_f64_retain(scale_radius).unwrap_or(Decimal::ZERO),
            g: Decimal::from_f64_retain(g).unwrap_or(Decimal::ZERO),
            mass_f64: mass,
            scale_radius_f64: scale_radius,
            g_f64: g,
        }
    }

    /// Create a Plummer IC from Decimal parameters (exact config).
    pub fn new_decimal(mass: Decimal, scale_radius: Decimal, g: Decimal) -> Self {
        Self {
            mass_f64: mass.to_f64().unwrap_or(0.0),
            scale_radius_f64: scale_radius.to_f64().unwrap_or(0.0),
            g_f64: g.to_f64().unwrap_or(0.0),
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

/// King (1966) lowered-Maxwellian model.
///
/// The DF is f(E) proportional to (exp((E_0 - E)/sigma^2) - 1) for E < E_0
/// and zero otherwise.  Construction solves the Poisson-Boltzmann ODE
/// (via RK4) from r = 0 outward to determine the tidal radius r_t, the
/// density profile, and the self-consistent velocity dispersion sigma.
pub struct KingIC {
    /// Total mass of the system.
    pub mass: Decimal,
    /// Dimensionless King concentration parameter W_0 = Phi(0)/sigma^2.
    pub king_parameter_w0: Decimal,
    /// Core velocity dispersion sigma (derived self-consistently from the ODE).
    pub velocity_dispersion: Decimal,
    /// Gravitational constant G.
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
            mass: Decimal::from_f64_retain(mass).unwrap_or(Decimal::ZERO),
            king_parameter_w0: Decimal::from_f64_retain(king_parameter_w0).unwrap_or(Decimal::ZERO),
            velocity_dispersion: Decimal::from_f64_retain(sigma).unwrap_or(Decimal::ZERO),
            g: Decimal::from_f64_retain(g).unwrap_or(Decimal::ZERO),
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
            mass.to_f64().unwrap_or(0.0),
            king_parameter_w0.to_f64().unwrap_or(0.0),
            scale_radius.to_f64().unwrap_or(0.0),
            g.to_f64().unwrap_or(0.0),
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
    if x >= *x_table.last().unwrap_or(&0.0) {
        return *y_table.last().unwrap_or(&0.0);
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

/// Hernquist (1990) cuspy halo model.
///
/// Density profile rho proportional to 1/(r (r+a)^3), yielding a central
/// r^{-1} cusp more realistic than Plummer for galaxy halos.  The DF is
/// available in closed form (Hernquist 1990, eq. 17).
pub struct HernquistIC {
    /// Total mass of the system.
    pub mass: Decimal,
    /// Hernquist scale radius `a`.
    pub scale_radius: Decimal,
    /// Gravitational constant G.
    pub g: Decimal,
    // Cached f64 values for hot-path computation
    mass_f64: f64,
    scale_radius_f64: f64,
    g_f64: f64,
}

impl HernquistIC {
    /// Create a Hernquist IC from `f64` parameters.
    pub fn new(mass: f64, scale_radius: f64, g: f64) -> Self {
        Self {
            mass: Decimal::from_f64_retain(mass).unwrap_or(Decimal::ZERO),
            scale_radius: Decimal::from_f64_retain(scale_radius).unwrap_or(Decimal::ZERO),
            g: Decimal::from_f64_retain(g).unwrap_or(Decimal::ZERO),
            mass_f64: mass,
            scale_radius_f64: scale_radius,
            g_f64: g,
        }
    }

    /// Create a Hernquist IC from Decimal parameters (exact config).
    pub fn new_decimal(mass: Decimal, scale_radius: Decimal, g: Decimal) -> Self {
        Self {
            mass_f64: mass.to_f64().unwrap_or(0.0),
            scale_radius_f64: scale_radius.to_f64().unwrap_or(0.0),
            g_f64: g.to_f64().unwrap_or(0.0),
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

/// Henon isochrone model: Phi(r) = -GM / (sqrt(b^2 + r^2) + b).
///
/// Analytic DF from Henon (1960) / Binney & Tremaine eq. 4.48.  The
/// isochrone is the only non-trivial spherical potential for which all
/// orbits can be solved in closed form (hence the name).
pub struct IsochroneIC {
    /// Total mass of the system.
    pub mass: Decimal,
    /// Isochrone scale parameter `b`.
    pub scale_radius: Decimal,
    /// Gravitational constant G.
    pub g: Decimal,
    // Cached f64 values for hot-path computation
    mass_f64: f64,
    scale_radius_f64: f64,
    g_f64: f64,
}

impl IsochroneIC {
    /// Create an Isochrone IC from `f64` parameters.
    pub fn new(mass: f64, scale_radius: f64, g: f64) -> Self {
        Self {
            mass: Decimal::from_f64_retain(mass).unwrap_or(Decimal::ZERO),
            scale_radius: Decimal::from_f64_retain(scale_radius).unwrap_or(Decimal::ZERO),
            g: Decimal::from_f64_retain(g).unwrap_or(Decimal::ZERO),
            mass_f64: mass,
            scale_radius_f64: scale_radius,
            g_f64: g,
        }
    }

    /// Create an Isochrone IC from Decimal parameters (exact config).
    pub fn new_decimal(mass: Decimal, scale_radius: Decimal, g: Decimal) -> Self {
        Self {
            mass_f64: mass.to_f64().unwrap_or(0.0),
            scale_radius_f64: scale_radius.to_f64().unwrap_or(0.0),
            g_f64: g.to_f64().unwrap_or(0.0),
            mass,
            scale_radius,
            g,
        }
    }
}

impl IsolatedEquilibrium for IsochroneIC {
    fn distribution_function(&self, energy: f64, _angular_momentum: f64) -> f64 {
        if energy >= 0.0 {
            return 0.0;
        }
        use std::f64::consts::PI;
        let m = self.mass_f64;
        let b = self.scale_radius_f64;
        let g = self.g_f64;
        let gm = g * m;

        // Dimensionless energy: e_tilde = -E*b/(G*M), in (0, 1) for bound orbits.
        let e_tilde = -energy * b / gm;
        if e_tilde <= 0.0 || e_tilde >= 1.0 {
            return 0.0;
        }

        // Binney & Tremaine (2008), eq. 4.48 (Henon isochrone DF):
        //
        // f(E) = M / [2^(7/2) * (2*pi)^3 * (G*M*b)^(3/2)]
        //      * sqrt(e_tilde) / (1 - e_tilde)^4
        //      * [ 27 - 66*e_tilde + 320*e_tilde^2 - 240*e_tilde^3 + 64*e_tilde^4
        //          + 3*(16*e_tilde^2 + 28*e_tilde - 9)
        //            * arcsin(sqrt(e_tilde)) / sqrt(e_tilde * (1 - e_tilde)) ]

        let e2 = e_tilde * e_tilde;
        let e3 = e2 * e_tilde;
        let e4 = e3 * e_tilde;
        let one_minus_e = 1.0 - e_tilde;

        // Polynomial part
        let poly = 27.0 - 66.0 * e_tilde + 320.0 * e2 - 240.0 * e3 + 64.0 * e4;

        // Arcsin part: 3*(16*e^2 + 28*e - 9) * arcsin(sqrt(e)) / sqrt(e*(1-e))
        let sqrt_e = e_tilde.sqrt();
        let sqrt_e_one_minus_e = (e_tilde * one_minus_e).sqrt();
        let arcsin_coeff = 3.0 * (16.0 * e2 + 28.0 * e_tilde - 9.0);
        let arcsin_term = arcsin_coeff * sqrt_e.asin() / sqrt_e_one_minus_e;

        let bracket = poly + arcsin_term;

        // Prefactor: M / [2^(7/2) * (2*pi)^3 * (G*M*b)^(3/2)]
        let two_7_2 = 2.0_f64.powf(3.5); // 2^(7/2) = 8*sqrt(2)
        let two_pi_cubed = (2.0 * PI).powi(3);
        let gmb_3_2 = (gm * b).powf(1.5);
        let prefactor = m / (two_7_2 * two_pi_cubed * gmb_3_2);

        let f = prefactor * sqrt_e / one_minus_e.powi(4) * bracket;
        f.max(0.0)
    }

    fn density_profile(&self, r: f64) -> f64 {
        use std::f64::consts::PI;
        let m = self.mass_f64;
        let b = self.scale_radius_f64;
        let r2 = r * r;
        let a = (b * b + r2).sqrt(); // a = sqrt(b^2 + r^2)
        let ba = b + a; // b + a

        // Binney & Tremaine eq. 2.46:
        // rho(r) = M*b / (4*pi) * (3*(b+a)*a^2 - r^2*(b+3*a)) / (a^3 * (b+a)^3)
        let numer = 3.0 * ba * a * a - r2 * (b + 3.0 * a);
        let denom = a.powi(3) * ba.powi(3);
        m * b / (4.0 * PI) * numer / denom
    }

    fn potential(&self, r: f64) -> f64 {
        let b = self.scale_radius_f64;
        let a = (b * b + r * r).sqrt();
        -self.g_f64 * self.mass_f64 / (a + b)
    }
}

/// NFW (Navarro-Frenk-White) dark matter halo model.
///
/// Density profile rho proportional to 1/((r/r_s)(1 + r/r_s)^2).  The DF
/// is computed numerically via Eddington inversion at construction time and
/// stored as a tabulated (E, f(E)) lookup for fast evaluation.
pub struct NfwIC {
    /// Total enclosed mass within the virial radius r_vir = c * r_s.
    pub mass: Decimal,
    /// NFW scale radius r_s.
    pub scale_radius: Decimal,
    /// Halo concentration c = r_vir / r_s.
    pub concentration: Decimal,
    /// Gravitational constant G.
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
    /// Create an NFW IC from `f64` parameters.
    ///
    /// Derives rho_s from the enclosed mass and solves the Eddington inversion
    /// integral numerically, storing the resulting DF as a lookup table.
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
        phi_rho.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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
        let phi_max = *phi_tab.last().unwrap_or(&0.0);
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
            mass: Decimal::from_f64_retain(mass).unwrap_or(Decimal::ZERO),
            scale_radius: Decimal::from_f64_retain(scale_radius).unwrap_or(Decimal::ZERO),
            concentration: Decimal::from_f64_retain(concentration).unwrap_or(Decimal::ZERO),
            g: Decimal::from_f64_retain(g).unwrap_or(Decimal::ZERO),
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
            mass.to_f64().unwrap_or(0.0),
            scale_radius.to_f64().unwrap_or(0.0),
            concentration.to_f64().unwrap_or(0.0),
            g.to_f64().unwrap_or(0.0),
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
///
/// Evaluates f(E(x,v)) = f(1/2 v^2 + Phi(r)) on every (x,v) grid point
/// and returns the result as a flat [`PhaseSpaceSnapshot`].  The outer
/// spatial dimension (ix1) is parallelised with rayon.
pub fn sample_on_grid(
    ic: &(dyn IsolatedEquilibrium + Sync),
    domain: &Domain,
) -> PhaseSpaceSnapshot {
    sample_on_grid_with_progress(ic, domain, None)
}

/// Like [`sample_on_grid`], but reports intra-phase progress via the optional
/// [`StepProgress`](crate::tooling::core::progress::StepProgress) handle so the
/// TUI can show a cell-level progress bar during IC generation.
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
