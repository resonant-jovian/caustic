//! Isolated equilibrium initial conditions: Plummer, King, Hernquist, NFW.
//! All specified as f(E) or f(E,L) — integrals of motion.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;

/// Trait for isolated equilibrium models specified as a distribution function.
pub trait IsolatedEquilibrium {
    /// Distribution function f(E, L). E = specific energy, L = specific angular momentum.
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64;
    /// Density profile ρ(r) in real space.
    fn density_profile(&self, r: f64) -> f64;
    /// Gravitational potential Φ(r) (spherically symmetric).
    fn potential(&self, r: f64) -> f64;
}

/// Plummer sphere: f(E) ∝ (−E)^(7/2).
/// Analytic DF; density ρ ∝ (r²+a²)^(−5/2).
pub struct PlummerIC {
    pub mass: f64,
    pub scale_radius: f64,
}

impl PlummerIC {
    pub fn new(mass: f64, scale_radius: f64) -> Self {
        Self { mass, scale_radius }
    }
}

impl IsolatedEquilibrium for PlummerIC {
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64 {
        todo!("f(E) = (24*sqrt(2))/(7*pi^3) * M*a^(-2) * (-2*E*a^2/(G*M))^(7/2) for E<0")
    }
    fn density_profile(&self, r: f64) -> f64 {
        todo!("rho(r) = 3M/(4*pi*a^3) * (1 + r^2/a^2)^(-5/2)")
    }
    fn potential(&self, r: f64) -> f64 {
        todo!("Phi(r) = -G*M / sqrt(r^2 + a^2)")
    }
}

/// King (1966) lowered Maxwellian: f(E) ∝ (e^((E₀−E)/σ²) − 1) for E < E₀.
pub struct KingIC {
    pub mass: f64,
    /// Dimensionless King concentration W₀ = Φ(0)/σ².
    pub king_parameter_w0: f64,
    pub velocity_dispersion: f64,
}

impl KingIC {
    pub fn new(mass: f64, king_parameter_w0: f64, velocity_dispersion: f64) -> Self {
        Self { mass, king_parameter_w0, velocity_dispersion }
    }
}

impl IsolatedEquilibrium for KingIC {
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64 {
        todo!("numerically integrate King DF via Poisson-Boltzmann")
    }
    fn density_profile(&self, r: f64) -> f64 {
        todo!("numerically integrate King DF for density")
    }
    fn potential(&self, r: f64) -> f64 {
        todo!("numerically integrate King Poisson-Boltzmann for potential")
    }
}

/// Hernquist (1990): ρ ∝ 1/(r(r+a)³). More realistic halo profile.
pub struct HernquistIC {
    pub mass: f64,
    pub scale_radius: f64,
}

impl HernquistIC {
    pub fn new(mass: f64, scale_radius: f64) -> Self {
        Self { mass, scale_radius }
    }
}

impl IsolatedEquilibrium for HernquistIC {
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64 {
        todo!("Hernquist DF via Eddington inversion")
    }
    fn density_profile(&self, r: f64) -> f64 {
        todo!("rho(r) = M*a / (2*pi * r * (r+a)^3)")
    }
    fn potential(&self, r: f64) -> f64 {
        todo!("Phi(r) = -G*M / (r + a)")
    }
}

/// NFW (Navarro-Frenk-White): ρ ∝ 1/(r(r+rs)²).
/// No analytic DF — use Eddington inversion numerically.
pub struct NfwIC {
    pub mass: f64,
    pub scale_radius: f64,
    pub concentration: f64,
}

impl NfwIC {
    pub fn new(mass: f64, scale_radius: f64, concentration: f64) -> Self {
        Self { mass, scale_radius, concentration }
    }
}

impl IsolatedEquilibrium for NfwIC {
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64 {
        todo!("Eddington integral: f(E) = 1/sqrt(8*pi^2) * d/dE int dPhi/sqrt(E-Phi) * d^2rho/dPhi^2")
    }
    fn density_profile(&self, r: f64) -> f64 {
        todo!("rho(r) = rho_s / (r/rs * (1 + r/rs)^2)")
    }
    fn potential(&self, r: f64) -> f64 {
        todo!("Phi(r) = -4*pi*G*rho_s*rs^3/r * ln(1 + r/rs)")
    }
}

/// Sample an isolated equilibrium IC onto the 6D grid.
pub fn sample_on_grid(ic: &dyn IsolatedEquilibrium, domain: &Domain) -> PhaseSpaceSnapshot {
    todo!("evaluate f(E(x,v)) = f(0.5*v^2 + Phi(r)) on every (x,v) grid point")
}
