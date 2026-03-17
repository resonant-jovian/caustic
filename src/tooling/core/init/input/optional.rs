//! Optional solver configuration parameters with defaults matching the spec (Section 7.2).

use super::super::domain::Timestep;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Operator splitting order.
pub enum SplittingMethod {
    /// 1st-order Lie splitting. Only for testing.
    Lie,
    /// 2nd-order Strang splitting. Default.
    Strang,
    /// 4th-order Yoshida splitting.
    Yoshida4,
}

/// Poisson solver selection.
pub enum PoissonMethod {
    /// FFT with periodic BC. Fastest, O(N³ log N). Default.
    FftPeriodic,
    /// FFT with isolated BC (James zero-padding). For non-periodic domains.
    FftIsolated,
    /// Geometric multigrid. Supports arbitrary BC.
    Multigrid,
    /// Spherical harmonics expansion. For nearly-spherical problems.
    SphericalHarmonics,
    /// Barnes-Hut tree code. For AMR / non-uniform grids.
    Tree,
}

/// Advection scheme selection.
pub enum AdvectionMethod {
    /// Semi-Lagrangian with cubic spline. No CFL constraint. Default.
    SemiLagrangian,
    /// Conservative finite volume.
    FiniteVolume,
    /// Spectral advection in velocity.
    Spectral,
}

/// Phase-space storage strategy selection.
pub enum RepresentationKind {
    /// Brute-force uniform 6D grid. O(N⁶). Default.
    UniformGrid,
    /// Adaptive mesh refinement in 6D.
    Amr,
    /// Lagrangian cold dark matter sheet. O(N³).
    Sheet,
    /// Tensor-train decomposition. O(N³r³).
    TensorTrain,
    /// Spectral-in-velocity representation.
    SpectralV,
    /// Hybrid sheet + grid representation.
    Hybrid,
}

/// External potential callback: Φ_ext(x, t) → f64.
pub type ExternalPotentialFn = Box<dyn Fn([f64; 3], f64) -> f64 + Send + Sync>;

/// Optional solver configuration. All fields have spec-default values.
///
/// Uses `rust_decimal::Decimal` for exact arithmetic on tolerance/timing parameters.
/// Cached f64 values are provided for hot-path access.
pub struct OptionalParams {
    /// Integration timestep. Adaptive by default (delta_t = 0 means adaptive).
    pub dt: Timestep,
    /// CFL safety factor ∈ (0,1). Default 0.5.
    pub cfl_factor: Decimal,
    /// Minimum adaptive timestep before CFL exit. Default 1e-10.
    pub dt_min: Decimal,
    /// Optional time-dependent external potential Φ_ext(x,t). None = self-gravity only.
    pub phi_external: Option<ExternalPotentialFn>,
    /// Operator splitting order. Default Strang (2nd order).
    pub splitting_method: SplittingMethod,
    /// Poisson solver. Default FftPeriodic.
    pub poisson_method: PoissonMethod,
    /// Advection scheme. Default SemiLagrangian.
    pub advection_method: AdvectionMethod,
    /// Phase-space storage strategy. Default UniformGrid.
    pub representation: RepresentationKind,
    /// Time between snapshot outputs. Default 0.01 (overridden to t_final/100 when known).
    pub output_interval: Decimal,
    /// Time between diagnostic rows. Default 0 = every step.
    pub diagnostic_interval: Decimal,
    /// Energy conservation tolerance for exit. Default 1e-6.
    pub epsilon_energy: Decimal,
    /// Mass fraction threshold for exit. Default 0.99.
    pub epsilon_mass: Decimal,
    /// Casimir drift tolerance. Default 1e-4.
    pub epsilon_casimir: Decimal,
    /// Steady-state norm threshold. Default 1e-8.
    pub epsilon_steady: Decimal,
    /// Max wall-clock seconds. None = unlimited.
    pub wall_time_limit: Option<Decimal>,
    /// Seconds between checkpoint saves. Default 3600.
    pub checkpoint_interval_secs: Decimal,
    /// Enable comoving coordinates and expansion. Default false.
    pub cosmological: bool,
    /// Initial scale factor (cosmological mode only). Default 1.0.
    pub a_init: Decimal,
    /// H₀ (cosmological mode only).
    pub hubble_0: Option<Decimal>,
    /// Ω_m (cosmological mode only).
    pub omega_m: Option<Decimal>,
    /// Ω_Λ (cosmological mode only).
    pub omega_lambda: Option<Decimal>,
}

impl OptionalParams {
    /// CFL factor as f64 for computation.
    pub fn cfl_factor_f64(&self) -> f64 {
        self.cfl_factor.to_f64().unwrap_or(0.5)
    }
    /// Minimum timestep as f64.
    pub fn dt_min_f64(&self) -> f64 {
        self.dt_min.to_f64().unwrap_or(1e-10)
    }
    /// Output interval as f64.
    pub fn output_interval_f64(&self) -> f64 {
        self.output_interval.to_f64().unwrap_or(0.01)
    }
    /// Diagnostic interval as f64.
    pub fn diagnostic_interval_f64(&self) -> f64 {
        self.diagnostic_interval.to_f64().unwrap_or(0.0)
    }
    /// Energy tolerance as f64.
    pub fn epsilon_energy_f64(&self) -> f64 {
        self.epsilon_energy.to_f64().unwrap_or(1e-6)
    }
    /// Mass threshold as f64.
    pub fn epsilon_mass_f64(&self) -> f64 {
        self.epsilon_mass.to_f64().unwrap_or(0.99)
    }
    /// Casimir tolerance as f64.
    pub fn epsilon_casimir_f64(&self) -> f64 {
        self.epsilon_casimir.to_f64().unwrap_or(1e-4)
    }
    /// Steady-state threshold as f64.
    pub fn epsilon_steady_f64(&self) -> f64 {
        self.epsilon_steady.to_f64().unwrap_or(1e-8)
    }
    /// Wall time limit as f64.
    pub fn wall_time_limit_f64(&self) -> Option<f64> {
        self.wall_time_limit.as_ref().and_then(|d| d.to_f64())
    }
    /// Initial scale factor as f64.
    pub fn a_init_f64(&self) -> f64 {
        self.a_init.to_f64().unwrap_or(1.0)
    }
}

/// Helper to create a Decimal from f64 for default values.
fn dec(v: f64) -> Decimal {
    Decimal::from_f64_retain(v).unwrap_or(Decimal::ZERO)
}

impl Default for OptionalParams {
    fn default() -> Self {
        Self {
            dt: Timestep {
                delta_t: Decimal::ZERO,
            }, // adaptive
            cfl_factor: dec(0.5),
            dt_min: dec(1e-10),
            phi_external: None,
            splitting_method: SplittingMethod::Strang,
            poisson_method: PoissonMethod::FftPeriodic,
            advection_method: AdvectionMethod::SemiLagrangian,
            representation: RepresentationKind::UniformGrid,
            output_interval: dec(0.01),
            diagnostic_interval: Decimal::ZERO,
            epsilon_energy: dec(1e-6),
            epsilon_mass: dec(0.99),
            epsilon_casimir: dec(1e-4),
            epsilon_steady: dec(1e-8),
            wall_time_limit: None,
            checkpoint_interval_secs: dec(3600.0),
            cosmological: false,
            a_init: Decimal::ONE,
            hubble_0: None,
            omega_m: None,
            omega_lambda: None,
        }
    }
}
