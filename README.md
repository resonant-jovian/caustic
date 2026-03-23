# caustic

**A 6D Vlasov–Poisson solver framework for collisionless gravitational dynamics.**

[![Crates.io](https://img.shields.io/crates/v/caustic.svg)](https://crates.io/crates/caustic)
[![docs.rs](https://docs.rs/caustic/badge.svg)](https://docs.rs/caustic)
[![CI](https://github.com/resonant-jovian/caustic/actions/workflows/test.yml/badge.svg)](https://github.com/resonant-jovian/caustic/actions/workflows/test.yml)
[![Clippy](https://github.com/resonant-jovian/caustic/actions/workflows/clippy.yml/badge.svg)](https://github.com/resonant-jovian/caustic/actions/workflows/clippy.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> [!IMPORTANT]
> Pre-0.1.0 — the API is unstable, features may be incomplete or change without notice, and it is not yet intended for general use. Even after 0.1.0, until version 1.0.0 it should not be relied upon for production workloads or serious research.

---

### Highlights

- **8 phase-space representations** — from brute-force 6D grids to HT tensor compression
- **10 Poisson solvers** — FFT, multigrid, tree, spectral harmonics, tensor decomposition
- **14 time integrators** — 1st through 6th order splitting, unsplit RK, BUG, exponential
- **10 IC generators** — Plummer, Hernquist, King, NFW, Zel'dovich, mergers, tidal, disk, custom
- **LoMaC conservation** — machine-precision mass/momentum/energy preservation
- **188 validation tests** — equilibrium, instability, convergence, solver cross-validation
- **Pluggable trait architecture** — swap any component independently

---

## Contents

- [For Everyone](#for-everyone) — What caustic does, quick demo, install
- [For Researchers](#for-researchers) — Physics, validation, conservation, citations
- [For Developers](#for-developers) — Architecture, traits, API, code quality
- [Feature Flags](#feature-flags) | [Development](#development) | [License](#license)

---

## For Everyone

Most astrophysical simulations of dark matter, galaxies, and stellar systems use **N-body methods** — they scatter millions of particles and let gravity do its work. But particles are a lie. They introduce artificial collisions and sampling noise that destroy exactly the structures you care about: razor-thin streams of stars torn from satellite galaxies, the velocity distribution at any point in a dark matter halo, and the caustic surfaces where phase-space sheets fold.

**caustic** takes a fundamentally different approach. It solves the Vlasov–Poisson equations directly on a full 6D grid (3 spatial + 3 velocity dimensions), evolving the distribution function f(x, v, t) without any particles at all. No sampling noise. No artificial two-body relaxation. The phase-space structure you see is the phase-space structure that's there.

### Install

```toml
[dependencies]
caustic = "0.0.12"
```

### How it works

```
                        ┌─────────────────────────────────────────────┐
                        │           TimeIntegrator                    │
                        │   (orchestrates the full timestep cycle)    │
                        └──────────────────┬──────────────────────────┘
                                           │
          ┌────────────────────────────────┼────────────────────────────────┐
          │                                │                                │
          ▼                                ▼                                ▼
   ┌─────────────┐              ┌──────────────────┐              ┌─────────────┐
   │  Advector   │              │  PhaseSpaceRepr  │              │PoissonSolver│
   │             │              │                  │              │             │
   │ advances f  │◄─────────────│  stores f(x,v)   │─────────────►│ ∇²Φ = 4πGρ  │
   │ by Δt       │ acceleration │  computes ρ(x)   │   density    │ returns Φ,g │
   └─────────────┘              └──────────────────┘              └─────────────┘
                                        ▲
                                        │
                                ┌───────┴────────┐
                                │ Initial        │
                                │ Conditions     │
                                │ (ICs)          │
                                └────────────────┘
```

Each box is a **swappable trait** — pick the phase-space representation, Poisson solver, and time integrator that fit your problem. The library provides multiple implementations of each.

### Quick start

```rust
use caustic::prelude::*;
use caustic::{
    FftPoisson, PlummerIC, SemiLagrangian, StrangSplitting,
    SpatialBoundType, VelocityBoundType, sample_on_grid,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Domain::builder()
        .spatial_extent(20.0)      // [-20, 20]^3 in natural units
        .velocity_extent(3.0)      // [-3, 3]^3
        .spatial_resolution(32)    // 32^3 spatial grid
        .velocity_resolution(32)   // 32^3 velocity grid
        .t_final(50.0)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()?;

    // Plummer sphere: mass=1, scale_radius=1, G=1
    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let snap = sample_on_grid(&ic, &domain);

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(1.0))
        .initial_conditions(snap)
        .time_final(50.0)
        .build()?;

    let exit = sim.run()?;
    exit.print_summary();
    Ok(())
}
```

### What you can simulate

- **Dark matter halos** — Plummer, Hernquist, King, and NFW equilibria
- **Galaxy mergers** — two-body encounters with configurable mass ratios and orbits
- **Tidal streams** — satellite disruption in an external host potential
- **Disk dynamics** — bar and spiral instabilities with Toomre Q parameter control
- **Cosmological structure** — Zel'dovich pancake formation and caustic collapse
- **Plasma physics analogues** — Jeans instability, Landau damping, two-stream instability

### Companion: phasma

[phasma](https://github.com/resonant-jovian/phasma) is a ratatui-based terminal UI that consumes caustic as a library dependency. It provides interactive parameter editing, live diagnostics rendering, density/phase-space heatmaps, energy conservation plots, radial profiles, and rank monitoring — all from the terminal. phasma contains no solver logic; it delegates entirely to caustic.

> [!TIP]
> **Researchers** — jump to [For Researchers](#for-researchers) for the physics, validation suite, and citation info.
> **Developers** — jump to [For Developers](#for-developers) for the trait architecture, builder API, and code quality details.

---

## For Researchers

### Governing equations

caustic evolves the 6D distribution function f(x, v, t) under the coupled Vlasov–Poisson system:

```
∂f/∂t + v · ∇ₓf − ∇ₓΦ · ∇ᵥf = 0        (Vlasov equation)
ρ(x,t) = ∫ f(x,v,t) dv³                (density coupling)
∇²Φ(x,t) = 4πGρ(x,t)                   (Poisson equation)
```

**Conserved quantities** (monitored as validation diagnostics): total mass M, total energy E = T + W, total momentum P, total angular momentum L, Casimir invariants C[s] = ∫s(f) dx³dv³ (including C₂ = ∫f² dx³dv³ and entropy S = −∫f ln f dx³dv³).

**Operator splitting**: the Vlasov equation splits into spatial drift (∂f/∂t + v·∇ₓf = 0) and velocity kick (∂f/∂t − ∇Φ·∇ᵥf = 0). Each sub-step is a pure translation. Strang splitting (drift–kick–drift) is second-order and symplectic; Yoshida gives fourth-order accuracy.

### Solver capabilities

#### Phase-space representations

| Representation | Memory | Description |
|---|---|---|
| `UniformGrid6D` | O(N⁶) | Brute-force 6D grid, rayon-parallelized. Reference implementation. |
| `HtTensor` | O(dNk² + dk³) | Hierarchical Tucker decomposition. Black-box construction via HTACA. SLAR advection. |
| `SheetTracker` | O(N³) | Lagrangian cold dark matter sheet. CIC density deposit. Caustic detection. |

<details>
<summary>All 8 representations</summary>

| Representation | Memory | Description |
|---|---|---|
| `UniformGrid6D` | O(N⁶) | Brute-force 6D grid, rayon-parallelized. Reference implementation. |
| `HtTensor` | O(dNk² + dk³) | Hierarchical Tucker tensor decomposition. Black-box construction via HTACA. SLAR advection. |
| `TensorTrain` | O(dNr²) | TT-SVD decomposition with cross approximation advection. |
| `SheetTracker` | O(N³) | Lagrangian cold dark matter sheet. CIC density deposit. Caustic detection. |
| `SpectralV` | O(N³M³) | Hermite spectral basis in velocity; finite-difference in space. |
| `AmrGrid` | adaptive | Adaptive mesh refinement in 6D with gradient-based refinement. |
| `HybridRepr` | adaptive | Sheet/grid hybrid with caustic-aware interface switching. |
| `SphericalRepr` | O(N_r N_l²) | Spherical harmonic basis with radial grid. |

</details>

#### Poisson solvers

| Solver | BC | Complexity | Description |
|---|---|---|---|
| `FftPoisson` | Periodic | O(N³ log N) | Real-to-complex FFT via `realfft`, rayon-parallelized. |
| `VgfPoisson` | Isolated | O(N³ log N) | Spectral-accuracy isolated BC via Vico-Greengard-Ferrando method. |
| `Multigrid` | Periodic/Isolated | O(N³) | V-cycle with red-black Gauss-Seidel smoothing, rayon-parallelized. |

<details>
<summary>All 10 Poisson solvers</summary>

| Solver | BC | Complexity | Description |
|---|---|---|---|
| `FftPoisson` | Periodic | O(N³ log N) | Real-to-complex FFT via `realfft`, rayon-parallelized. |
| `FftIsolated` *(deprecated)* | Isolated | O(N³ log N) | Hockney-Eastwood zero-padding on (2N)³ grid. Deprecated in favor of `VgfPoisson`. |
| `VgfPoisson` | Isolated | O(N³ log N) | Spectral-accuracy isolated BC via Vico-Greengard-Ferrando method. |
| `TensorPoisson` | Isolated | O(N³ log N) | Braess-Hackbusch exponential sum Green's function + dense 3D FFT. 2nd-order near-field correction. |
| `HtPoisson` | Isolated | O(R_G·r·N log N) | HT-format Poisson: exp-sum Green's function in HT tensor format with rank re-compression. |
| `Multigrid` | Periodic/Isolated | O(N³) | V-cycle with red-black Gauss-Seidel smoothing, rayon-parallelized. |
| `SphericalHarmonicsPoisson` | Isolated | O(l²_max N) | Legendre decomposition + radial ODE integration. |
| `TreePoisson` | Isolated | O(N³ log N³) | Barnes-Hut octree with multipole expansion, rayon-parallelized. |
| `MultipoleExpansion` | Isolated | O(l²_max N) | Multipole expansion gravity solver. |
| `Spherical1DPoisson` | Spherical | O(N_r) | 1D radial Poisson solver for spherically symmetric problems. |

</details>

#### Time integrators

| Integrator | Order | Description |
|---|---|---|
| `StrangSplitting` | 2 | Drift(Δt/2) → kick(Δt) → drift(Δt/2). Symplectic. |
| `YoshidaSplitting` | 4 | 3-substep Yoshida coefficients, 7 sub-steps total. Symplectic. |
| `BugIntegrator` | varies | Basis Update & Galerkin (BUG) for HT tensors. |

<details>
<summary>All 14 time integrators</summary>

| Integrator | Order | Description |
|---|---|---|
| `StrangSplitting` | 2 | Drift(Δt/2) → kick(Δt) → drift(Δt/2). Symplectic. |
| `YoshidaSplitting` | 4 | 3-substep Yoshida coefficients, 7 sub-steps total. Symplectic. |
| `LieSplitting` | 1 | Drift(Δt) → kick(Δt). For testing/comparison only. |
| `UnsplitIntegrator` | 2/3/4 | Method-of-lines RK on full Vlasov PDE. No splitting error. Re-solves Poisson at each stage. |
| `RkeiIntegrator` | 3 | RKEI (Runge-Kutta Exponential Integrator). SSP-RK3 with unsplit characteristics. |
| `InstrumentedStrangSplitting` | 2 | Strang splitting with per-sub-step rank diagnostics. |
| `AdaptiveStrangSplitting` | 2 | Strang with adaptive timestep control. |
| `BlanesMoanSplitting` | 4 | Blanes-Moan optimized splitting coefficients. |
| `Rkn6Splitting` | 6 | 6th-order Runge-Kutta-Nyström splitting. |
| `BugIntegrator` | varies | Basis Update & Galerkin (BUG) for HT tensors. |
| `RkBugIntegrator` | varies | Runge-Kutta BUG variant. |
| `ParallelBugIntegrator` | varies | Parallelized BUG. |
| `CosmologicalStrangSplitting` | 2 | Strang with cosmological scale factor. |
| `LawsonRkIntegrator` | varies | Lawson Runge-Kutta exponential integrator. |

</details>

### HT tensor compression

A uniform 6D grid at N=64 per dimension requires 64⁶ ~ 7x10^10 cells. The `HtTensor` representation exploits the balanced binary tree structure of the x-v split to store f(x,v) in O(dNk² + dk³) memory, where k is the representation rank.

**Construction:**
- `HtTensor::from_full()` — compress a full 6D array via hierarchical SVD (HSVD)
- `HtTensor::from_function_aca()` — black-box construction via HTACA (Ballani & Grasedyck 2013), sampling O(dNk) fibers instead of all N⁶ points

**Operations** (all in compressed format, never expanding to full):
- `compute_density()` — O(Nk²) velocity integration via tree contraction
- `truncate(eps)` — rank-adaptive recompression (orthogonalize + top-down SVD)
- `add()` — rank-concatenation, then `truncate()` to compress
- `inner_product()` / `frobenius_norm()` — O(dk⁴) via recursive Gram matrices
- `advect_x()` / `advect_v()` — SLAR (Semi-Lagrangian Adaptive Rank) via HTACA reconstruction

### Initial conditions

**Isolated equilibria** (via `sample_on_grid()`):
- **`PlummerIC`** — Plummer sphere via analytic f(E)
- **`KingIC`** — King model via Poisson-Boltzmann ODE (RK4)
- **`HernquistIC`** — Hernquist profile via closed-form f(E)
- **`NfwIC`** — NFW profile via numerical Eddington inversion

**Cosmological:**
- **`ZeldovichSingleMode`** — single-mode Zel'dovich pancake
- **`ZeldovichIC`** — multi-mode from Gaussian random field (Harrison-Zel'dovich spectrum, FFT-based)

**Disk dynamics:**
- **`DiskStabilityIC`** — exponential disk with Shu (1969) f(E, L_z), Toomre Q, azimuthal perturbation modes

**Multi-body and custom:**
- **`MergerIC`** — two-body superposition with configurable offsets
- **`TidalIC`** — progenitor in an external host potential
- **`CustomIC`** / **`CustomICArray`** — user-provided callable or pre-computed array

### Conservation framework (LoMaC)

The LoMaC (Local Macroscopic Conservation) framework restores exact conservation of mass, momentum, and energy after each time step. Enable via `.lomac(true)` on the simulation builder.

> [!NOTE]
> LoMaC guarantees machine-precision conservation of macroscopic quantities (mass, momentum, energy) regardless of the underlying phase-space representation or its truncation tolerance. This is critical for long-time simulations where accumulated drift can corrupt results.

**Components:**
- **KFVS** — Kinetic Flux Vector Splitting macroscopic solver with half-Maxwellian fluxes
- **Conservative SVD** — Moment-preserving projection via Gram matrix inversion
- **Rank-adaptive controller** — Conservation-aware truncation tolerance with budget management

### Validation suite

> [!IMPORTANT]
> **188 tests**, all passing. Run with `cargo test --release -- --test-threads=1`.

#### Physics validation

| Test | Validates |
|---|---|
| `free_streaming` | Spatial advection accuracy (G=0, f shifts as f(x-vt, v, 0)) |
| `uniform_acceleration` | Velocity advection accuracy |
| `jeans_instability` | Growth rate matches analytic dispersion relation (periodic BC) |
| `jeans_instability_isolated` | Jeans instability with FftIsolated BCs |
| `jeans_stability` | Sub-Jeans perturbation does not grow |
| `plummer_equilibrium` | Long-term equilibrium preservation |
| `king_equilibrium` | King model (W_0=5) equilibrium preservation |
| `nfw_equilibrium` | NFW profile cusp preservation over 5 t_dyn |
| `zeldovich_pancake` | Caustic position matches analytic Zel'dovich solution |
| `spherical_collapse` | Spherical overdensity collapse dynamics |
| `cold_collapse_1d` | Cold slab collapse, phase-space spiral formation |
| `conservation_laws` | Energy, momentum, C_2 conservation to tolerance |
| `landau_damping` | Damping rate matches analytic Landau rate |
| `nonlinear_landau_damping` | Large perturbation, phase-space vortex, conservation over 50 bounce times |
| `two_stream_instability` | Perturbation growth and saturation |
| `sheet_1d_density_comparison` | 1D sheet model vs exact Eldridge-Feix dynamics |

#### Convergence tests

| Test | Validates |
|---|---|
| `density_integration_convergence` | Spatial convergence of density integration |
| `free_streaming_convergence` | Error decreases with resolution |
| `convergence_table_structure` | Richardson extrapolation framework |

#### Solver cross-validation

| Test | Validates |
|---|---|
| `multigrid_vs_fft` | Multigrid matches FftPoisson on periodic problem |
| `multigrid_convergence_order` | 2nd-order convergence (double N, error /4) |
| `spherical_vs_fft_isolated` | Spherical harmonics matches FftIsolated |
| `tree_vs_fft_isolated` | Barnes-Hut tree matches FftIsolated |
| `tensor_poisson_vs_fft_isolated` | TensorPoisson matches FftIsolated |

Plus HT tensor/ACA tests (17), conservation framework tests (15), diagnostics tests (10), and solver-specific unit tests.

### Diagnostics

Conserved quantities monitored each timestep via `GlobalDiagnostics`:

- Total energy (kinetic + potential), momentum, angular momentum
- Casimir C_2, Boltzmann entropy
- Virial ratio, total mass in box
- Density profile (radial binning)

**Analysis tools:**
- L1/L2/L_inf field norms and error metrics
- `ConservationSummary` — energy, mass, momentum, C_2 drift tracking
- `convergence_table` — Richardson extrapolation and convergence order estimation
- `CausticDetector` — caustic surface detection, first caustic time

<details>
<summary>Equation-to-module mapping</summary>

| Equation / Method | Module | Reference |
|---|---|---|
| Vlasov equation (operator splitting) | `time/strang.rs`, `time/yoshida.rs`, `time/rkei.rs` | Cheng & Knorr (1976) |
| Poisson equation (FFT) | `poisson/fft.rs`, `poisson/multigrid.rs` | Hockney & Eastwood (1988) |
| Exponential sum 1/r approximation | `poisson/exponential_sum.rs`, `poisson/tensor_poisson.rs` | Braess & Hackbusch, IMA J. Numer. Anal. 25(4) (2005) |
| HT-format Poisson with rank re-compression | `poisson/ht_poisson.rs` | Khoromskij (2011), Braess-Hackbusch (2005) |
| Near-field correction (0th + 2nd order) | `poisson/exponential_sum.rs`, `poisson/tensor_poisson.rs` | Exl, Mauser & Zhang, JCP (2016) |
| Hierarchical Tucker decomposition | `algos/ht.rs` | Hackbusch & Kuhn, JCAM 261 (2009) |
| HTACA (black-box HT construction) | `algos/ht.rs`, `algos/aca.rs` | Ballani & Grasedyck, Numer. Math. 124 (2013) |
| SLAR (semi-Lagrangian adaptive rank) | `algos/ht.rs` | Kormann & Reuter (2024) |
| RKEI (Runge-Kutta exponential integrator) | `time/rkei.rs` | Kormann & Reuter (2024) |
| LoMaC (local macroscopic conservation) | `conservation/lomac.rs`, `conservation/kfvs.rs` | Guo & Qiu, arXiv:2207.00518 (2022) |
| Conservative SVD projection | `conservation/conservative_svd.rs` | Guo & Qiu (2022) |
| KFVS (kinetic flux vector splitting) | `conservation/kfvs.rs` | Mandal & Deshpande, Comput. Fluids 23(4) (1994) |
| Plummer DF | `init/isolated.rs` | Dejonghe (1987), Binney & Tremaine (2008) |
| King model (Poisson-Boltzmann ODE) | `init/isolated.rs` | King, AJ 71 (1966) |
| NFW (numerical Eddington inversion) | `init/isolated.rs` | Navarro, Frenk & White, ApJ 462 (1996) |
| Zel'dovich approximation | `init/cosmological.rs` | Zel'dovich, A&A 5 (1970) |
| Shu disk DF | `init/stability.rs` | Shu, ApJ 160 (1970) |
| VGF isolated Poisson | `poisson/vgf.rs` | Vico, Greengard & Ferrando, JCP 323 (2016) |
| BUG (Basis Update & Galerkin) | `time/bug.rs` | Ceruti, Lubich et al., BIT Numer. Math. (2022) |

</details>

### Citation

```bibtex
@software{caustic,
  title  = {caustic: A 6D Vlasov--Poisson solver framework},
  url    = {https://github.com/resonant-jovian/caustic},
  year   = {2026}
}
```

---

## For Developers

### Trait architecture

Each solver component is a Rust trait; implementations are swapped independently:

| Trait | Role | Key implementations |
|---|---|---|
| `PhaseSpaceRepr` | Store and query f(x,v) | `UniformGrid6D`, `HtTensor`, `SheetTracker` + 5 more |
| `PoissonSolver` | Solve ∇²Φ = 4πGρ | `FftPoisson`, `VgfPoisson`, `Multigrid` + 7 more |
| `Advector` | Advance f by Δt | `SemiLagrangian` (Catmull-Rom + sparse polynomial) |
| `TimeIntegrator` | Orchestrate timestep | `StrangSplitting`, `YoshidaSplitting`, `BugIntegrator` + 11 more |
| `ExitCondition` | Termination criteria | `TimeLimitCondition`, `EnergyDriftCondition` + 7 more |

> [!TIP]
> See [For Researchers > Solver capabilities](#solver-capabilities) for the full implementation tables.

### Builder API

```rust
// Basic simulation
let mut sim = Simulation::builder()
    .domain(domain)
    .poisson_solver(FftPoisson::new(&domain))
    .advector(SemiLagrangian::new())
    .integrator(StrangSplitting::new(1.0))
    .initial_conditions(snap)
    .time_final(50.0)
    .build()?;

let exit = sim.run()?;            // run to completion
// or: sim.step()?;               // single timestep
```

```rust
// Dynamic dispatch with conservation
let mut sim = Simulation::builder()
    .domain(domain)
    .poisson_solver_boxed(Box::new(VgfPoisson::new(&domain)))
    .advector(SemiLagrangian::new())
    .integrator_boxed(Box::new(YoshidaSplitting::new(1.0)))
    .initial_conditions(snap)
    .time_final(100.0)
    .lomac(true)                   // enable LoMaC conservation
    .cfl_factor(0.5)
    .build()?;
```

> [!NOTE]
> Configuration values (t_final, G, cfl_factor) are stored as `rust_decimal::Decimal` for exact user-facing arithmetic. Backward-compatible `f64` setters are provided alongside `_decimal()` variants.

### The `PhaseSpaceRepr` trait

The central abstraction. All phase-space storage strategies implement this interface:

```rust
pub trait PhaseSpaceRepr: Send + Sync {
    fn compute_density(&self) -> DensityField;
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64);
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64);
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor;
    fn total_mass(&self) -> f64;
    fn casimir_c2(&self) -> f64;
    fn entropy(&self) -> f64;
    fn stream_count(&self) -> StreamCountField;
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64>;
    fn total_kinetic_energy(&self) -> f64;       // default: NAN
    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot;  // default: empty
    fn load_snapshot(&mut self, snap: PhaseSpaceSnapshot);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn can_materialize(&self) -> bool;           // default: true
    fn memory_bytes(&self) -> usize;             // default: 0
}
```

### Exit conditions

| Condition | Description |
|---|---|
| `TimeLimitCondition` | Stop at t >= t_final |
| `EnergyDriftCondition` | Stop when \|dE/E\| > tolerance |
| `MassLossCondition` | Stop when mass loss exceeds threshold |
| `CasimirDriftCondition` | Stop when C_2 drift exceeds threshold |
| `WallClockCondition` | Stop after wall-clock time limit |
| `SteadyStateCondition` | Stop when \|df/dt\| < epsilon |
| `CflViolationCondition` | Stop on CFL violation |
| `VirialRelaxedCondition` | Stop when virial ratio stabilizes |
| `CausticFormationCondition` | Stop at first caustic (stream count > 1) |

### Code quality

> [!NOTE]
> - **`warnings = "forbid"`** — all rustc warnings are hard errors
> - **`clippy::unwrap_used`, `expect_used`, `panic` = `"deny"`** — no panicking in non-test code
> - **OOM protection** — `can_materialize()` lets compressed representations (HT, TT) report whether full 6D materialization is safe; large grids gracefully degrade instead of crashing

### I/O and checkpointing

- Binary snapshot save/load (shape + time + data)
- CSV diagnostics (time series of all conserved quantities)
- JSON checkpoint (snapshot + diagnostics history)
- HT tensor checkpoint (tree structure without dense expansion)
- **HDF5** (feature-gated): `save_snapshot_hdf5`, `load_snapshot_hdf5`, `save_ht_checkpoint_hdf5`, `load_ht_checkpoint_hdf5`

---

## Feature Flags

```toml
[dependencies]
caustic = { version = "0.0.12", features = ["hdf5"] }
```

| Flag | Description |
|---|---|
| `hdf5` | HDF5 I/O via `hdf5-metno` (snapshot and HT checkpoint read/write) |
| `mpi` | MPI domain decomposition via the `mpi` crate (requires MPI installation) |
| `jemalloc` | jemalloc global allocator via `tikv-jemallocator` |
| `mimalloc-alloc` | mimalloc global allocator |
| `dhat-heap` | Heap profiling via `dhat` |
| `tracy` | Tracy profiler integration via `tracing-tracy` |

---

## Development

### Performance

- **Parallelism**: rayon data parallelism across `UniformGrid6D`, `FftPoisson`, `TreePoisson`, `SheetTracker`, `Multigrid`
- **Release profile**: fat LTO, `codegen-units = 1`, `target-cpu=native`
- **Benchmarks**: criterion (`cargo bench`), benchmark binary: `solver_kernels`
- **Instrumentation**: `tracing::info_span!` on all hot methods (zero overhead without a subscriber)
- **Profiling profile**: `[profile.profiling]` inherits release with debug symbols for `perf`/`samply`

### `dev.sh`

Unified development script for testing, benchmarking, and profiling:

```bash
./dev.sh doctor                      # check prerequisites, show install commands
./dev.sh test                        # run all tests (release, sequential)
./dev.sh test --ignored              # include expensive #[ignore] tests
./dev.sh test --all                  # test both caustic and phasma
./dev.sh bench                       # criterion benchmarks
./dev.sh bench --save baseline-v1    # save named baseline
./dev.sh profile flamegraph          # generate flamegraph SVG
./dev.sh profile dhat                # heap profiling
./dev.sh profile perf                # perf record + report
./dev.sh profile samply              # samply (browser UI)
./dev.sh build --release             # release build (fat LTO)
./dev.sh build --profiling           # release + debug symbols
./dev.sh lint                        # clippy + fmt --check
./dev.sh info                        # show available tools
```

<details>
<summary>Prerequisites</summary>

```bash
# Cargo tools
cargo install flamegraph samply

# System packages (Arch Linux)
sudo pacman -S --needed perf valgrind heaptrack kcachegrind massif-visualizer

# Optional: Tracy profiler (AUR)
yay -S tracy
```

**Ubuntu / Debian:**

```bash
cargo install flamegraph samply
sudo apt install linux-tools-common linux-tools-$(uname -r) valgrind heaptrack kcachegrind massif-visualizer
```

Run `./dev.sh doctor` to check which tools are installed and get install commands for missing ones.

</details>

---

## Minimum supported Rust version

Rust edition 2024, targeting **stable Rust 1.85+**.

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See [LICENSE](LICENSE) for details.
