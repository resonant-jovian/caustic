# caustic

Core solver library (v0.0.12) for a 6D Vlasov-Poisson solver targeting collisionless gravitational dynamics (dark matter halos, galaxy dynamics, tidal streams). Trait-based, pluggable architecture. Rust edition 2024.

Sibling project: **phasma** at `../phasma` (ratatui TUI + batch runner that consumes caustic as a path dependency). See `../phasma/CLAUDE.md` for TUI/runner details.

`rust_decimal` is used for exact arithmetic in config/domain structs (Decimal for config, cached f64 for hot-path computation).

## Commands

Both projects share an identical `dev.sh` script. Run from within `caustic/` or `phasma/`.

```bash
# Preferred — dev.sh (project-aware defaults)
./dev.sh test                        # caustic: --release --test-threads=1; phasma: debug
./dev.sh test --all                  # test both projects
./dev.sh lint                        # clippy + fmt (auto-lints sibling)
./dev.sh build --release             # release build
./dev.sh run -- --config run.toml    # build & run phasma
./dev.sh bench                       # criterion benchmarks (caustic only)
./dev.sh profile flamegraph          # profiling with interactive target picker
./dev.sh doctor                      # check prerequisites, show install commands
./dev.sh info                        # project info, features, cargo profiles

# Direct cargo (when needed)
cargo build                                               # debug
cargo build --release                                     # release (required for validation tests)
cargo test --release -- --test-threads=1                   # caustic validation tests (231+ tests)
cargo test                                                # phasma tests
cargo clippy
cargo fmt
```

## Physics and Mathematical Core

The solver evolves the 6D distribution function f(x, v, t) under the coupled system:

```
df/dt + v . grad_x f - grad Phi . grad_v f = 0   (Vlasov equation)
rho(x,t) = integral f(x,v,t) dv^3                 (density coupling)
laplacian Phi(x,t) = 4 pi G rho(x,t)              (Poisson equation)
```

**Conserved quantities** (used as validation diagnostics): total mass M, total energy E = T + W, total momentum P, total angular momentum L, Casimir invariants C[s] = integral s(f) dx^3 dv^3 (including C2 = integral f^2 dx^3 dv^3 and entropy S = -integral f ln f dx^3 dv^3).

**Operator splitting**: the Vlasov equation naturally splits into a spatial drift (df/dt + v . grad_x f = 0) and a velocity kick (df/dt - grad Phi . grad_v f = 0). Each sub-step is a pure translation in its respective coordinates. Strang splitting (drift dt/2, kick dt, drift dt/2) is second-order and symplectic. Yoshida splitting gives fourth-order accuracy.

## Architecture — Pluggable Trait Architecture

Each solver component is a Rust trait; implementations are swapped independently:

| Trait | Role | Implementations |
|---|---|---|
| `PhaseSpaceRepr` | Store/query f(x,v) | `UniformGrid6D`, `HtTensor`, `TensorTrain`, `SheetTracker`, `SpectralV`, `AmrGrid`, `HybridRepr`, `FlowMapRepr`, `MacroMicroRepr`, `SphericalRepr` |
| `PoissonSolver` | laplacian Phi = 4piG rho given rho | `FftPoisson`, `FftIsolated`, `TensorPoisson`, `HtPoisson`, `Multigrid`, `SphericalHarmonicsPoisson`, `TreePoisson`, `RangeSeparatedPoisson`, `Spherical1DPoisson`, `VgfPoisson` |
| `Advector` | Advance f by dt given g = -grad Phi | `SemiLagrangian` (Catmull-Rom cubic) |
| `TimeIntegrator` | Orchestrate: rho->Phi->g->advect | `StrangSplitting`, `YoshidaSplitting`, `LieSplitting`, `UnsplitIntegrator`, `RkeiIntegrator`, `InstrumentedStrangSplitting`, `AdaptiveStrangSplitting`, `BlanesMoanSplitting`, `CosmologicalStrangSplitting`, `LawsonRkIntegrator`, `Rkn6Splitting`, `BugIntegrator`, `ParallelBugIntegrator`, `RkBugIntegrator` |
| `ExitCondition` | Termination criteria | `TimeLimitCondition`, `EnergyDriftCondition`, `MassLossCondition`, `CasimirDriftCondition`, `WallClockCondition`, `SteadyStateCondition`, `CflViolationCondition`, `VirialRelaxedCondition`, `CausticFormationCondition` |
| `IsolatedEquilibrium` | IC distribution functions | `PlummerIC`, `KingIC`, `HernquistIC`, `IsochroneIC`, `NfwIC` |

**Core trait signatures:**
```rust
// PhaseSpaceRepr (Send + Sync)
fn compute_density(&self) -> DensityField;
fn advect_x(&mut self, displacement: &DisplacementField, dt: f64);
fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64);
fn moment(&self, position: &[f64; 3], order: usize) -> Tensor;
fn total_mass(&self) -> f64;
fn casimir_c2(&self) -> f64;
fn entropy(&self) -> f64;
fn stream_count(&self) -> StreamCountField;
fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64>;
fn total_kinetic_energy(&self) -> f64;       // default: panic
fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot;  // default: panic
fn as_any(&self) -> &dyn Any;

// PoissonSolver
fn solve(&self, density: &DensityField, g: f64) -> PotentialField;
fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField;

// TimeIntegrator
fn advance(&mut self, repr: &mut dyn PhaseSpaceRepr, solver: &dyn PoissonSolver, advector: &dyn Advector, dt: f64);
fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64;

// ExitCondition
fn check(&self, diag: &GlobalDiagnostics, initial: &GlobalDiagnostics) -> Option<ExitReason>;
```

**Simulation builder API:**
```rust
Simulation::builder()
    .domain(domain)
    .representation_boxed(repr)      // or .representation(impl)
    .poisson_solver_boxed(poisson)   // or .poisson_solver(impl)
    .advector(SemiLagrangian::new())
    .integrator_boxed(integrator)    // or .integrator(impl)
    .initial_conditions(snapshot)
    .time_final(10.0)               // also _decimal() variants
    .gravitational_constant(1.0)
    .cfl_factor(0.5)
    .lomac(true)                    // optional conservation framework
    .build()?;
sim.run()?;  // returns ExitPackage
sim.step()?; // single step, returns Option<ExitReason>
```

## Module layout — `src/`

```
lib.rs              — public API re-exports, CausticError
serde_helpers.rs    — shared decimal_serde module (re-exported for phasma)
sim/mod.rs          — Simulation, SimulationBuilder
tooling/core/
  phasespace.rs     — PhaseSpaceRepr trait
  solver.rs         — PoissonSolver trait
  advecator.rs      — Advector trait
  integrator.rs     — TimeIntegrator trait, SimState
  conditions.rs     — ExitCondition trait + 9 impls, ExitReason (with Serialize/Deserialize/Display)
  types.rs          — DensityField, PotentialField, AccelerationField, DisplacementField, Tensor, PhaseSpaceSnapshot, StreamCountField
  progress.rs       — ProgressSnapshot, StepPhase, StepProgress (lock-free intra-step progress for TUI)
  diagnostics.rs    — GlobalDiagnostics, Diagnostics
  io.rs             — IOManager, OutputFormat (binary, HDF5)
  mpi.rs            — [feature-gated] slab decomposition, ghost-zone exchange
  algos/            — PhaseSpaceRepr impls: uniform, ht, ht3d, tensor, sheet, spectral, amr, hybrid, flow_map, macro_micro, spherical_repr, lagrangian (SemiLagrangian), aca, global, wpfc
  conservation/     — LoMaC: kfvs, conservative_svd, lomac, rank_adaptive
  init/             — IC generators: isolated (Plummer/King/Hernquist/Isochrone/NFW), cosmological (Zeldovich), mergers, tidal, stability (DiskStability), arbitrary
  init/domain.rs    — Domain, DomainBuilder, SpatialBoundType, VelocityBoundType
  init/input/       — MandatoryParams, OptionalParams
  poisson/          — fft (FftPoisson + FftIsolated), tensor_poisson, ht_poisson, multigrid, spherical, spherical_1d, tree, range_separated, vgf, exponential_sum, green_ht, multipole, utils
  time/             — strang, yoshida, lie, unsplit, rkei, adaptive, blanes_moan, cosmological, lawson, rkn6, bug, parallel_bug, rk_bug, rank_monitor (InstrumentedStrangSplitting)
  time/constraints/ — dynamical, orbital, spatial (CFL), velocity (CFL)
  output/           — global, primary, phasespace, velocity, caustic diagnostics
  output/exit/      — ExitEvaluator, ExitPackage, standard conditions
tooling/validation/ — 33 validation test modules + helpers.rs (shared test builders)
```

## Validation Suite

Tests run sequentially with `./dev.sh test` (or `cargo test --release -- --test-threads=1`):

| Test | Validates |
|---|---|
| `free_streaming` | Spatial advection accuracy (G=0, f shifts as f(x-vt,v,0)) |
| `uniform_acceleration` | Velocity advection accuracy |
| `jeans_instability` | Growth rate matches analytic dispersion relation |
| `jeans_stability` | Stable modes do not grow spuriously |
| `jeans_isolated` | Jeans instability with isolated BC |
| `plummer_equilibrium` | Equilibrium preserved for many t_dyn |
| `king_equilibrium` | King model (W0=5) equilibrium preservation |
| `nfw_equilibrium` | NFW model equilibrium preservation |
| `isochrone` | Isochrone model equilibrium preservation |
| `zeldovich_pancake` | Caustic position matches analytic Zel'dovich solution |
| `conservation` | Energy, momentum, angular momentum, C2 conserved to tolerance |
| `long_conservation` | Long-run conservation stability |
| `landau_damping` | Damping rate matches analytic Landau rate |
| `nonlinear_landau` | Nonlinear Landau damping regime |
| `strong_landau` | Strong Landau damping regime |
| `two_stream` | Two-stream instability |
| `bump_on_tail` | Bump-on-tail instability |
| `bgk` | BGK mode dynamics |
| `plasma_echo` | Plasma echo phenomenon |
| `spherical_collapse` | Spherical collapse density evolution |
| `cold_collapse` | Cold collapse dynamics |
| `sine_wave_collapse` | Sine-wave collapse dynamics |
| `triaxial_collapse` | Triaxial collapse dynamics |
| `violent_relaxation` | Violent relaxation process |
| `sheet_1d_density_comparison` | 1D sheet model vs exact Eldridge-Feix dynamics |
| `fujiwara` | Fujiwara sheet dynamics |
| `convergence` | Free-streaming convergence, density integration convergence, convergence tables |
| `stability` | Disk stability analysis |
| `mixing` | Phase-space mixing diagnostics |
| `mms` | Method of manufactured solutions verification |
| `casimir_higher` | Higher-order Casimir invariant conservation |
| `waterbag` | Waterbag distribution dynamics |
| `plummer_perturbation` | Plummer perturbation response |
| `uniform` | Uniform distribution tests |

Plus unit/integration tests across all modules. **231+ total tests.** Shared test helpers in `tooling/validation/helpers.rs` (build_standard_sim, relative_drift, snapshot_density, density_mass, assert_valid_output).

## Current Implementation State

**Phase space** (10 impls): `UniformGrid6D` (rayon), `HtTensor` (Hierarchical Tucker -- HSVD, truncation, SLAR advection via HTACA), `TensorTrain` (TT-SVD), `SheetTracker` (Lagrangian sheet, CIC deposit, rayon), `SpectralV` (Hermite velocity basis), `AmrGrid` (adaptive mesh refinement), `HybridRepr` (sheet/grid hybrid with caustic-aware interface), `FlowMapRepr` (flow-map advection), `MacroMicroRepr` (macro-micro decomposition), `SphericalRepr` (spherical coordinates)

**Poisson solvers** (10 impls): `FftPoisson` (periodic, realfft R2C, rayon), `FftIsolated` (Hockney-Eastwood zero-padding), `TensorPoisson` (Braess-Hackbusch exp-sum + dense 3D FFT), `HtPoisson` (exp-sum Green's in HT format), `Multigrid` (V-cycle, red-black GS, rayon), `SphericalHarmonicsPoisson` (Legendre + radial ODE), `TreePoisson` (Barnes-Hut octree, rayon), `RangeSeparatedPoisson` (split near/far field), `Spherical1DPoisson` (1D radial), `VgfPoisson` (variable Green's function)

**Time integrators** (14 impls): `StrangSplitting` (2nd-order symplectic), `YoshidaSplitting` (4th-order, 7 sub-steps), `LieSplitting` (1st-order), `UnsplitIntegrator` (RK2/3/4), `RkeiIntegrator` (SSP-RK3 exponential), `InstrumentedStrangSplitting` (Strang + per-step rank diagnostics), `AdaptiveStrangSplitting`, `BlanesMoanSplitting` (4th-order optimized), `CosmologicalStrangSplitting`, `LawsonRkIntegrator`, `Rkn6Splitting` (6th-order), `BugIntegrator`, `ParallelBugIntegrator`, `RkBugIntegrator`

**Advection**: `SemiLagrangian` (Catmull-Rom + sparse polynomial interpolation)

**ICs** (11 types): PlummerIC, HernquistIC, KingIC, IsochroneIC, NfwIC, ZeldovichSingleMode, ZeldovichIC, DiskStabilityIC, MergerIC, TidalIC, CustomIC/CustomICArray

**Conservation**: LoMaC framework (KFVS + conservative SVD projection), rank-adaptive truncation controller

**Exit conditions** (9 types): time limit, energy drift, mass loss, Casimir drift, wall clock, steady state, CFL violation, virial relaxed, caustic formation

**IO**: IOManager (binary + CSV + JSON checkpoint). Feature-gated HDF5 (`--features hdf5`).

**MPI**: Feature-gated slab domain decomposition (`--features mpi`)

**Performance**: rayon parallelism throughout, criterion benchmarks, tracing spans, feature-gated allocators (jemalloc/mimalloc/dhat/tracy), fat LTO release profile, `target-cpu=native`

**Key dependencies**: rust_decimal, rayon, rustfft, realfft, faer (SVD/QR), serde, tracing, anyhow/thiserror.
