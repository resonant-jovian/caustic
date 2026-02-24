//! Optional solver configuration parameters with defaults matching the spec (Section 7.2).

use super::super::domain::Timestep;

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

/// Optional solver configuration. All fields have spec-default values.
pub struct OptionalParams {
    /// Integration timestep. Adaptive by default.
    pub dt: Timestep,
    /// CFL safety factor ∈ (0,1). Default 0.5.
    pub cfl_factor: f64,
    /// Minimum adaptive timestep before CFL exit. Default 1e-10 × t_dyn.
    pub dt_min: f64,
    /// Optional time-dependent external potential Φ_ext(x,t). None = self-gravity only.
    pub phi_external: Option<Box<dyn Fn([f64; 3], f64) -> f64 + Send + Sync>>,
    /// Operator splitting order. Default Strang (2nd order).
    pub splitting_method: SplittingMethod,
    /// Poisson solver. Default FftPeriodic.
    pub poisson_method: PoissonMethod,
    /// Advection scheme. Default SemiLagrangian.
    pub advection_method: AdvectionMethod,
    /// Phase-space storage strategy. Default UniformGrid.
    pub representation: RepresentationKind,
    /// Time between snapshot outputs. Default t_final/100.
    pub output_interval: f64,
    /// Time between diagnostic rows. Default every step.
    pub diagnostic_interval: f64,
    /// Energy conservation tolerance for exit. Default 1e-6.
    pub epsilon_energy: f64,
    /// Mass fraction threshold for exit. Default 0.99.
    pub epsilon_mass: f64,
    /// Casimir drift tolerance. Default 1e-4.
    pub epsilon_casimir: f64,
    /// Steady-state norm threshold. Default 1e-8.
    pub epsilon_steady: f64,
    /// Max wall-clock seconds. None = unlimited.
    pub wall_time_limit: Option<f64>,
    /// Seconds between checkpoint saves. Default 3600.
    pub checkpoint_interval_secs: f64,
    /// Enable comoving coordinates and expansion. Default false.
    pub cosmological: bool,
    /// Initial scale factor (cosmological mode only). Default 1.0.
    pub a_init: f64,
    /// H₀ (cosmological mode only).
    pub hubble_0: Option<f64>,
    /// Ω_m (cosmological mode only).
    pub omega_m: Option<f64>,
    /// Ω_Λ (cosmological mode only).
    pub omega_lambda: Option<f64>,
}

impl Default for OptionalParams {
    fn default() -> Self {
        todo!("fill with spec defaults")
    }
}
