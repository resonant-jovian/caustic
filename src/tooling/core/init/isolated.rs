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
    /// Gravitational potential Φ(r) (spherically symmetric, includes factor G).
    fn potential(&self, r: f64) -> f64;
}

/// Plummer sphere: f(E) ∝ (−E)^(7/2).
/// Analytic DF; density ρ ∝ (r²+a²)^(−5/2).
pub struct PlummerIC {
    pub mass: f64,
    pub scale_radius: f64,
    pub g: f64,
}

impl PlummerIC {
    pub fn new(mass: f64, scale_radius: f64, g: f64) -> Self {
        Self {
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
        let a = self.scale_radius;
        let m = self.mass;
        let g = self.g;
        // Binney & Tremaine §4.4.3 (Eddington inversion of Plummer density):
        // In natural units G=M=a=1: f(E) = (24√2/7π³) * (-E)^(7/2)
        // In physical units: f(E) = (24√2/7π³) * M * (-E)^(7/2) / (a³ * (GM/a)^5)
        let e0 = g * m / a;
        let prefactor = (24.0 * 2.0_f64.sqrt()) / (7.0 * PI.powi(3));
        prefactor * m * (-energy).powf(3.5) / (a.powi(3) * e0.powf(5.0))
    }

    fn density_profile(&self, r: f64) -> f64 {
        use std::f64::consts::PI;
        let a = self.scale_radius;
        let m = self.mass;
        3.0 * m / (4.0 * PI * a.powi(3)) * (1.0 + r * r / (a * a)).powf(-2.5)
    }

    fn potential(&self, r: f64) -> f64 {
        -self.g * self.mass / (r * r + self.scale_radius * self.scale_radius).sqrt()
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
        Self {
            mass,
            king_parameter_w0,
            velocity_dispersion,
        }
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
    pub g: f64,
}

impl HernquistIC {
    pub fn new(mass: f64, scale_radius: f64, g: f64) -> Self {
        Self {
            mass,
            scale_radius,
            g,
        }
    }
}

impl IsolatedEquilibrium for HernquistIC {
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64 {
        todo!("Hernquist DF via Eddington inversion")
    }

    fn density_profile(&self, r: f64) -> f64 {
        let a = self.scale_radius;
        let m = self.mass;
        m * a / (2.0 * std::f64::consts::PI * r * (r + a).powi(3))
    }

    fn potential(&self, r: f64) -> f64 {
        -self.g * self.mass / (r + self.scale_radius)
    }
}

/// NFW (Navarro-Frenk-White): ρ ∝ 1/(r(r+rs)²).
/// No analytic DF — use Eddington inversion numerically.
pub struct NfwIC {
    pub mass: f64,
    pub scale_radius: f64,
    pub concentration: f64,
    pub g: f64,
}

impl NfwIC {
    pub fn new(mass: f64, scale_radius: f64, concentration: f64, g: f64) -> Self {
        Self {
            mass,
            scale_radius,
            concentration,
            g,
        }
    }
}

impl IsolatedEquilibrium for NfwIC {
    fn distribution_function(&self, energy: f64, angular_momentum: f64) -> f64 {
        todo!(
            "Eddington integral: f(E) = 1/sqrt(8*pi^2) * d/dE int dPhi/sqrt(E-Phi) * d^2rho/dPhi^2"
        )
    }

    fn density_profile(&self, r: f64) -> f64 {
        todo!("rho(r) = rho_s / (r/rs * (1 + r/rs)^2)")
    }

    fn potential(&self, r: f64) -> f64 {
        todo!("Phi(r) = -4*pi*G*rho_s*rs^3/r * ln(1 + r/rs)")
    }
}

/// Sample an isolated equilibrium IC onto the 6D grid.
/// Evaluates f(E(x,v)) = f(½v² + Φ(r)) on every (x,v) grid point.
pub fn sample_on_grid(ic: &dyn IsolatedEquilibrium, domain: &Domain) -> PhaseSpaceSnapshot {
    use rust_decimal::prelude::ToPrimitive;

    let nx1 = domain.spatial_res.x1 as usize;
    let nx2 = domain.spatial_res.x2 as usize;
    let nx3 = domain.spatial_res.x3 as usize;
    let nv1 = domain.velocity_res.v1 as usize;
    let nv2 = domain.velocity_res.v2 as usize;
    let nv3 = domain.velocity_res.v3 as usize;

    let dx = domain.dx();
    let dv = domain.dv();
    let lx = [
        domain.spatial.x1.to_f64().unwrap(),
        domain.spatial.x2.to_f64().unwrap(),
        domain.spatial.x3.to_f64().unwrap(),
    ];
    let lv = [
        domain.velocity.v1.to_f64().unwrap(),
        domain.velocity.v2.to_f64().unwrap(),
        domain.velocity.v3.to_f64().unwrap(),
    ];

    // Row-major strides (same layout as UniformGrid6D)
    let s_v3 = 1usize;
    let s_v2 = nv3;
    let s_v1 = nv2 * nv3;
    let s_x3 = nv1 * s_v1;
    let s_x2 = nx3 * s_x3;
    let s_x1 = nx2 * s_x2;

    let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
    let mut data = vec![0.0f64; total];

    for ix1 in 0..nx1 {
        let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
        for ix2 in 0..nx2 {
            let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];
            for ix3 in 0..nx3 {
                let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                let r = (x1 * x1 + x2 * x2 + x3 * x3).sqrt();
                let phi = ic.potential(r);
                let base = ix1 * s_x1 + ix2 * s_x2 + ix3 * s_x3;

                for iv1 in 0..nv1 {
                    let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                            let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                            let energy = 0.5 * v2sq + phi;
                            let f = ic.distribution_function(energy, 0.0).max(0.0);
                            data[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f;
                        }
                    }
                }
            }
        }
    }

    PhaseSpaceSnapshot {
        data,
        shape: [nx1, nx2, nx3, nv1, nv2, nv3],
        time: 0.0,
    }
}
