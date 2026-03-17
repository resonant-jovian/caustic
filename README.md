# caustic

**A 6D Vlasov–Poisson solver framework for collisionless gravitational dynamics.**

[![Crates.io](https://img.shields.io/crates/v/caustic.svg)](https://crates.io/crates/caustic)
[![docs.rs](https://docs.rs/caustic/badge.svg)](https://docs.rs/caustic)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

caustic is a modular, general-purpose library for solving the Vlasov–Poisson equations in full 6D phase space (3 spatial + 3 velocity dimensions). It targets astrophysical problems — dark matter halo formation, galaxy dynamics, tidal streams, stellar system stability — that are traditionally handled by N-body methods but suffer from artificial collisionality and loss of fine-grained phase-space structure.

The library provides a pluggable architecture where the phase-space representation, Poisson solver, time integrator, and initial condition generator can be swapped independently.

## Why not N-body?

N-body simulations sample the distribution function with discrete particles. This introduces noise and artificial two-body relaxation that destroys exactly the structures a collisionless solver should preserve: caustic surfaces, thin phase-space streams, and the true velocity distribution at any point. caustic solves the governing equation directly — no particles, no sampling noise, no artificial collisionality.

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
caustic = "0.0.9"
```

### Minimal example: Plummer sphere equilibrium

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

    // Set up a Plummer sphere: mass=1, scale_radius=1, G=1
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

### Conservation-aware simulation with LoMaC

```rust
let mut sim = Simulation::builder()
    .domain(domain)
    .poisson_solver(FftPoisson::new(&domain))
    .advector(SemiLagrangian::new())
    .integrator(StrangSplitting::new(1.0))
    .initial_conditions(snap)
    .time_final(50.0)
    .lomac(true)  // enable local macroscopic conservation
    .build()?;
```

## Architecture

Each solver component is a Rust trait; implementations are swapped independently:

| Trait | Role | Implementations |
|---|---|---|
| `PhaseSpaceRepr` | Store and query f(x,v) | `UniformGrid6D`, `HtTensor`, `TensorTrain`, `SheetTracker`, `SpectralV`, `AmrGrid`, `HybridRepr` |
| `PoissonSolver` | Solve ∇²Φ = 4πGρ | `FftPoisson`, `FftIsolated`, `TensorPoisson`, `Multigrid`, `SphericalHarmonicsPoisson`, `TreePoisson` |
| `Advector` | Advance f by Δt | `SemiLagrangian` (Catmull-Rom + sparse polynomial) |
| `TimeIntegrator` | Orchestrate timestep | `StrangSplitting` (2nd), `YoshidaSplitting` (4th), `LieSplitting` (1st), `UnsplitIntegrator` (RK2/3/4) |

### Phase-space representations

| Representation | Memory | Description |
|---|---|---|
| `UniformGrid6D` | O(N⁶) | Brute-force 6D grid, rayon-parallelized. Reference implementation. |
| `HtTensor` | O(dNk² + dk³) | Hierarchical Tucker tensor decomposition. Black-box construction via HTACA. SLAR advection. |
| `TensorTrain` | O(dNr²) | TT-SVD decomposition with cross approximation advection. |
| `SheetTracker` | O(N³) | Lagrangian cold dark matter sheet. CIC density deposit. Caustic detection. |
| `SpectralV` | O(N³M³) | Hermite spectral basis in velocity; finite-difference in space. |
| `AmrGrid` | adaptive | Adaptive mesh refinement in 6D with gradient-based refinement. |
| `HybridRepr` | adaptive | Sheet/grid hybrid with caustic-aware interface switching. |

### Poisson solvers

| Solver | BC | Complexity | Description |
|---|---|---|---|
| `FftPoisson` | Periodic | O(N³ log N) | Real-to-complex FFT via `realfft`, rayon-parallelized. |
| `FftIsolated` | Isolated | O(N³ log N) | Hockney-Eastwood zero-padding on (2N)³ grid. |
| `TensorPoisson` | Isolated | O(N³ log N) | Braess-Hackbusch exponential sum Green's function + dense 3D FFT. 2nd-order near-field correction. |
| `HtPoisson` | Isolated | O(R_G·r·N log N) | HT-format Poisson: exp-sum Green's function in HT tensor format with rank re-compression. |
| `Multigrid` | Periodic/Isolated | O(N³) | V-cycle with red-black Gauss-Seidel smoothing, rayon-parallelized. |
| `SphericalHarmonicsPoisson` | Isolated | O(l²_max N) | Legendre decomposition + radial ODE integration. |
| `TreePoisson` | Isolated | O(N³ log N³) | Barnes-Hut octree with multipole expansion, rayon-parallelized. |

### Time integrators

| Integrator | Order | Description |
|---|---|---|
| `StrangSplitting` | 2 | Drift(Δt/2) → kick(Δt) → drift(Δt/2). Symplectic. |
| `YoshidaSplitting` | 4 | 3-substep Yoshida coefficients, 7 sub-steps total. Symplectic. |
| `LieSplitting` | 1 | Drift(Δt) → kick(Δt). For testing/comparison only. |
| `UnsplitIntegrator` | 2/3/4 | Method-of-lines RK on full Vlasov PDE. No splitting error. Re-solves Poisson at each stage. |
| `RkeiIntegrator` | 3 | RKEI (Runge-Kutta Exponential Integrator). SSP-RK3 with unsplit characteristics. 3 Poisson solves per step. |
| `InstrumentedStrangSplitting` | 2 | Strang splitting with per-sub-step rank diagnostics (drift/kick/Poisson amplification tracking). |

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
    fn total_kinetic_energy(&self) -> f64;
    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot;
    fn load_snapshot(&mut self, snap: PhaseSpaceSnapshot);
    fn as_any(&self) -> &dyn Any;
}
```

## Hierarchical Tucker (HT) tensor compression

A uniform 6D grid at N=64 per dimension requires 64⁶ ≈ 7×10¹⁰ cells. The `HtTensor` representation exploits the balanced binary tree structure of the x-v split to store f(x,v) in O(dNk² + dk³) memory, where k is the representation rank and N is the grid size per dimension.

**Construction methods:**

- `HtTensor::from_full()` — compress a full 6D array via hierarchical SVD (HSVD). O(N⁶).
- `HtTensor::from_function_aca()` — **black-box construction** via the HTACA algorithm (Ballani & Grasedyck 2013). Builds the HT decomposition by sampling O(dNk) fibers instead of evaluating all N⁶ grid points.

**Operations** (all in compressed format, never expanding to full):

- `compute_density()` — O(Nk²) velocity integration via tree contraction
- `truncate(eps)` — rank-adaptive recompression (orthogonalize + top-down SVD)
- `add()` — rank-concatenation addition, followed by `truncate()` to compress
- `inner_product()` / `frobenius_norm()` — O(dk⁴) via recursive Gram matrices
- `evaluate()` — single-point query in O(dk³)
- `advect_x()` / `advect_v()` — SLAR (Semi-Lagrangian Adaptive Rank) via HTACA reconstruction

## Conservation framework (LoMaC)

The LoMaC (Local Macroscopic Conservation) framework restores exact conservation of mass, momentum, and energy after each time step. Enable via `.lomac(true)` on the simulation builder.

**Components:**

- **KFVS** — Kinetic Flux Vector Splitting macroscopic solver with half-Maxwellian fluxes
- **Conservative SVD** — Moment-preserving projection via Gram matrix inversion
- **Rank-adaptive controller** — Conservation-aware truncation tolerance with budget management

## Initial conditions

**Isolated equilibria** (via `sample_on_grid()`):

- **`PlummerIC`** — Plummer sphere via analytic distribution function f(E)
- **`KingIC`** — King model via Poisson-Boltzmann ODE (RK4 integration)
- **`HernquistIC`** — Hernquist profile via closed-form f(E)
- **`NfwIC`** — NFW profile via numerical Eddington inversion

**Cosmological:**

- **`ZeldovichSingleMode`** — single-mode Zel'dovich pancake
- **`ZeldovichIC`** — multi-mode Zel'dovich ICs from Gaussian random field (Harrison-Zel'dovich spectrum, FFT-based, reproducible seeding)

**Disk dynamics:**

- **`DiskStabilityIC`** — exponential disk with Shu (1969) distribution function f(E, L_z), Toomre Q stability parameter, azimuthal perturbation modes (bars, spirals)

**Multi-body and custom:**

- **`MergerIC`** — two-body superposition f = f₁ + f₂ with offsets
- **`TidalIC`** — progenitor equilibrium model in an external host potential
- **`CustomIC`** / **`CustomICArray`** — user-provided callable or pre-computed array

## Diagnostics

Conserved quantities monitored each timestep via `GlobalDiagnostics`:

- Total energy (kinetic + potential), momentum, angular momentum
- Casimir C₂, Boltzmann entropy
- Virial ratio, total mass in box
- Density profile (radial binning)

**Analysis tools:**

- L1/L2/L∞ field norms and error metrics
- `ConservationSummary` — energy, mass, momentum, C₂ drift tracking
- `convergence_table` — Richardson extrapolation and convergence order estimation
- `CausticDetector` — caustic surface detection, first caustic time

## I/O and checkpointing

- Binary snapshot save/load (shape + time + data)
- CSV diagnostics (time series of all conserved quantities)
- JSON checkpoint (snapshot + diagnostics history)
- HT tensor checkpoint (tree structure without dense expansion)
- **HDF5** (feature-gated): `save_snapshot_hdf5`, `load_snapshot_hdf5`, `save_ht_checkpoint_hdf5`, `load_ht_checkpoint_hdf5`

## Exit conditions

Termination is configurable via the builder API:

| Condition | Description |
|---|---|
| `TimeLimitCondition` | Stop at t ≥ t_final |
| `EnergyDriftCondition` | Stop when \|ΔE/E\| > tolerance |
| `MassLossCondition` | Stop when mass loss exceeds threshold |
| `CasimirDriftCondition` | Stop when C₂ drift exceeds threshold |
| `WallClockCondition` | Stop after wall-clock time limit |
| `SteadyStateCondition` | Stop when ∥∂f/∂t∥ < ε |
| `CflViolationCondition` | Stop on CFL violation |
| `VirialRelaxedCondition` | Stop when virial ratio stabilizes |
| `CausticFormationCondition` | Stop at first caustic (stream count > 1) |

## Validation suite

**163+ tests** — run with `cargo test --release -- --test-threads=1`:

### Physics validation

| Test | Validates |
|---|---|
| `free_streaming` | Spatial advection accuracy (G=0, f shifts as f(x−vt, v, 0)) |
| `uniform_acceleration` | Velocity advection accuracy |
| `jeans_instability` | Growth rate matches analytic dispersion relation (periodic BC) |
| `jeans_instability_isolated` | Jeans instability with FftIsolated BCs, growth rate comparison |
| `jeans_stability` | Sub-Jeans perturbation does not grow |
| `plummer_equilibrium` | Long-term equilibrium preservation |
| `king_equilibrium` | King model (W₀=5) equilibrium preservation |
| `nfw_equilibrium` | NFW profile cusp preservation over 5 t_dyn |
| `zeldovich_pancake` | Caustic position matches analytic Zel'dovich solution |
| `spherical_collapse` | Spherical overdensity collapse dynamics |
| `cold_collapse_1d` | Cold slab gravitational collapse, phase-space spiral formation |
| `conservation_laws` | Energy, momentum, C₂ conservation to tolerance |
| `landau_damping` | Damping rate matches analytic Landau rate |
| `nonlinear_landau_damping` | Large perturbation (ε=0.5), phase-space vortex, conservation over 50 bounce times |
| `two_stream_instability` | Two-stream IC, perturbation growth and saturation |
| `sheet_1d_density_comparison` | 1D sheet model vs exact Eldridge-Feix dynamics |

### Convergence tests

| Test | Validates |
|---|---|
| `density_integration_convergence` | Spatial convergence of density integration |
| `free_streaming_convergence` | Error decreases with resolution |
| `convergence_table_structure` | Richardson extrapolation framework |

### Solver cross-validation

| Test | Validates |
|---|---|
| `multigrid_vs_fft` | Multigrid matches FftPoisson on periodic problem |
| `multigrid_convergence_order` | 2nd-order convergence (double N, error /4) |
| `spherical_vs_fft_isolated` | Spherical harmonics matches FftIsolated |
| `tree_vs_fft_isolated` | Barnes-Hut tree matches FftIsolated |
| `tensor_poisson_vs_fft_isolated` | TensorPoisson matches FftIsolated |

Plus integration tests, HT tensor/ACA tests (17), conservation framework tests (15), diagnostics tests (10), and solver-specific unit tests.

## Equation-to-module mapping

| Equation / Method | Module | Reference |
|---|---|---|
| ∂f/∂t + v·∇ₓf − ∇Φ·∇ᵥf = 0 (Vlasov) | `time/strang.rs`, `time/yoshida.rs`, `time/rkei.rs` | Operator splitting: Cheng & Knorr (1976) |
| ∇²Φ = 4πGρ (Poisson) | `poisson/fft.rs`, `poisson/multigrid.rs` | Hockney & Eastwood (1988) |
| 1/r ≈ Σ c_k exp(-α_k r²) (exponential sum) | `poisson/exponential_sum.rs`, `poisson/tensor_poisson.rs` | Braess & Hackbusch, IMA J. Numer. Anal. 25(4) (2005) |
| HT-format Poisson with rank re-compression | `poisson/ht_poisson.rs` | Khoromskij (2011), Braess-Hackbusch (2005) |
| Near-field correction (0th + 2nd order) | `poisson/exponential_sum.rs`, `poisson/tensor_poisson.rs` | Exl, Mauser & Zhang, JCP (2016) |
| Hierarchical Tucker decomposition | `algos/ht.rs` | Hackbusch & Kühn, JCAM 261 (2009) |
| HTACA (black-box HT construction) | `algos/ht.rs`, `algos/aca.rs` | Ballani & Grasedyck, Numer. Math. 124 (2013) |
| SLAR (semi-Lagrangian adaptive rank) | `algos/ht.rs` | Kormann & Reuter (2024) |
| RKEI (Runge-Kutta exponential integrator) | `time/rkei.rs` | Kormann & Reuter (2024) |
| LoMaC (local macroscopic conservation) | `conservation/lomac.rs`, `conservation/kfvs.rs` | Guo & Qiu, arXiv:2207.00518 (2022) |
| Conservative SVD projection | `conservation/conservative_svd.rs` | Guo & Qiu (2022) |
| KFVS (kinetic flux vector splitting) | `conservation/kfvs.rs` | Mandal & Deshpande, Comput. Fluids 23(4) (1994) |
| Plummer DF: f(E) ∝ (−E)^{7/2} | `init/isolated.rs` | Dejonghe (1987), Binney & Tremaine (2008) |
| King model: Poisson-Boltzmann ODE | `init/isolated.rs` | King, AJ 71 (1966) |
| NFW: numerical Eddington inversion | `init/isolated.rs` | Navarro, Frenk & White, ApJ 462 (1996) |
| Zel'dovich approximation | `init/cosmological.rs` | Zel'dovich, A&A 5 (1970) |
| Shu disk DF: f(E, L_z) | `init/stability.rs` | Shu, ApJ 160 (1970) |

## Feature flags

```toml
[dependencies]
caustic = { version = "0.0.6", features = ["hdf5"] }
```

| Flag | Description |
|---|---|
| `hdf5` | HDF5 I/O via `hdf5-metno` (snapshot and HT checkpoint read/write) |
| `mpi` | MPI domain decomposition via the `mpi` crate (requires MPI installation) |
| `jemalloc` | jemalloc global allocator via `tikv-jemallocator` |
| `mimalloc-alloc` | mimalloc global allocator |
| `dhat-heap` | Heap profiling via `dhat` |
| `tracy` | Tracy profiler integration via `tracing-tracy` |

## Performance

- **Parallelism**: rayon data parallelism across `UniformGrid6D` (density, advection), `FftPoisson` (FFT axes), `TreePoisson` (grid walk), `SheetTracker` (particle advection), `Multigrid` (residual, prolongation)
- **Release profile**: fat LTO, `codegen-units = 1`, `target-cpu=native` (via `.cargo/config.toml`)
- **Benchmarks**: criterion benchmarks (`cargo bench`), benchmark binary: `solver_kernels`
- **Instrumentation**: `tracing::info_span!` on all hot methods (zero overhead without a subscriber)
- **Profiling profile**: `[profile.profiling]` inherits release with debug symbols for `perf`/`samply`

## Companion: phasma

[phasma](https://github.com/resonant-jovian/phasma) is a ratatui-based terminal UI that consumes caustic as a library dependency. It provides interactive parameter editing, live diagnostics rendering, density/phase-space heatmaps, energy conservation plots, radial profile charts, rank monitoring, and Poisson solver analysis — all from the terminal. phasma contains no solver logic; it delegates entirely to caustic.

## Minimum supported Rust version

Rust edition 2024, targeting **stable Rust 1.85+**.

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See [LICENSE](LICENSE) for details.

## Citation

If you use caustic in academic work, please cite:

```bibtex
@software{caustic,
  title  = {caustic: A 6D Vlasov--Poisson solver framework},
  url    = {https://github.com/resonant-jovian/caustic},
  year   = {2026}
}
```
