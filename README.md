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
caustic = "0.0.4"
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

## Architecture

Each solver component is a Rust trait; implementations are swapped independently:

| Trait | Role | Implementations |
|---|---|---|
| `PhaseSpaceRepr` | Store and query f(x,v) | `UniformGrid6D` (rayon-parallelized), `HtTensor` (Hierarchical Tucker) |
| `PoissonSolver` | Solve nabla^2 Phi = 4piG rho | `FftPoisson` (periodic, R2C), `FftIsolated` (Hockney-Eastwood zero-padding) |
| `Advector` | Advance f by dt | `SemiLagrangian` (Catmull-Rom interpolation) |
| `TimeIntegrator` | Orchestrate operator splitting | `StrangSplitting` (2nd-order), `LieSplitting` (1st-order), `YoshidaSplitting` (4th-order) |

### The `PhaseSpaceRepr` trait

The central abstraction. All phase-space storage strategies implement this interface:

```rust
pub trait PhaseSpaceRepr: Send + Sync {
    /// Integrate f over all velocities: rho(x) = integral f dv^3.
    fn compute_density(&self) -> DensityField;

    /// Drift sub-step: advect f in spatial coordinates by dx = v*dt.
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64);

    /// Kick sub-step: advect f in velocity coordinates by dv = g*dt.
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64);

    /// Compute velocity moment of order n at given spatial position.
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor;

    /// Total mass M = integral f dx^3 dv^3.
    fn total_mass(&self) -> f64;

    /// Casimir invariant C_2 = integral f^2 dx^3 dv^3.
    fn casimir_c2(&self) -> f64;

    /// Boltzmann entropy S = -integral f ln f dx^3 dv^3.
    fn entropy(&self) -> f64;

    /// Number of distinct velocity streams at each spatial point.
    fn stream_count(&self) -> StreamCountField;

    /// Extract the local velocity distribution f(v|x) at a given position.
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64>;

    /// Total kinetic energy T = 1/2 integral f v^2 dx^3 dv^3.
    fn total_kinetic_energy(&self) -> f64;

    /// Extract a full 6D snapshot of the current state.
    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot;
}
```

### Hierarchical Tucker (HT) tensor compression

A uniform 6D grid at N=64 per dimension requires 64^6 ~ 7x10^10 cells. The `HtTensor` representation exploits the balanced binary tree structure of the x-v split to store f(x,v) in O(dnk^2 + dk^3) memory, where k is the representation rank and n is the grid size per dimension.

**Construction methods:**

- `HtTensor::from_full()` — compress a full 6D array via hierarchical SVD (HSVD). O(N^6) — diagnostic use only.
- `HtTensor::from_function()` — evaluate a callable on the grid, then compress. O(N^6).
- `HtTensor::from_function_aca()` — **black-box construction** via the HTACA algorithm (Ballani & Grasedyck 2013). Builds the HT decomposition by sampling O(dNk) fibers instead of evaluating all N^6 grid points. Leaf frames are computed via fiber sampling + column-pivoted QR; interior transfer tensors via projected SVD.

**Operations** (all in compressed format, never expanding to full):

- `compute_density()` — O(Nk^2) velocity integration via tree contraction
- `truncate(eps)` — rank-adaptive recompression (orthogonalize + top-down SVD)
- `add()` — rank-concatenation addition, followed by `truncate()` to compress
- `inner_product()` / `frobenius_norm()` — O(dk^4) via recursive Gram matrices
- `evaluate()` — single-point query in O(dk^3)

### Adaptive Cross Approximation (ACA)

The `aca` module provides a standalone partially-pivoted ACA implementation (Bebendorf 2000) for low-rank matrix approximation. It builds a rank-k factorization A ~ U V^T by querying only O((m+n)k) entries. Used internally by HTACA for black-box tensor construction.

```rust
use caustic::tooling::core::algos::aca::{aca_partial_pivot, FnMatrix};

let mat = FnMatrix::new(100, 80, |i, j| {
    let d = (i as f64 / 100.0) - (j as f64 / 80.0);
    (-d * d / 0.1).exp()
});
let result = aca_partial_pivot(&mat, 1e-8, 20);
println!("ACA rank: {}", result.rank);
```

## Initial conditions

All implemented ICs satisfy the `IsolatedEquilibrium` trait and can be sampled onto a grid with `sample_on_grid()`:

- **`PlummerIC`** — Plummer sphere via analytic distribution function f(E)
- **`KingIC`** — King model via Poisson-Boltzmann ODE (RK4 integration)
- **`HernquistIC`** — Hernquist profile via closed-form f(E)
- **`NfwIC`** — NFW profile via numerical Eddington inversion
- **`ZeldovichSingleMode`** — single-mode Zel'dovich pancake (cosmological)
- **`MergerIC`** — two-body superposition f = f_1 + f_2 with offsets
- **`TidalIC`** — progenitor equilibrium model in an external host potential
- **`CustomIC`** / **`CustomICArray`** — user-provided callable or pre-computed array

## Diagnostics

Conserved quantities monitored each timestep via `GlobalDiagnostics`:

- Total energy (kinetic + potential), momentum, angular momentum
- Casimir C_2, Boltzmann entropy
- Virial ratio, total mass in box
- Density profile (radial binning)

Additional output modules: `VelocityMoments` (surface density, J-factor), `PhaseSpaceDiagnostics` (power spectrum, growth rates), `CausticDetector` (caustic surface detection, first caustic time).

## Validation suite

Run with `cargo test --release -- --test-threads=1`:

| Test | Validates |
|---|---|
| `free_streaming` | Spatial advection accuracy (G=0, f shifts as f(x-vt, v, 0)) |
| `uniform_acceleration` | Velocity advection accuracy |
| `jeans_instability` | Growth rate matches analytic dispersion relation |
| `jeans_stability` | Sub-Jeans perturbation does not grow |
| `plummer_equilibrium` | Long-term equilibrium preservation |
| `zeldovich_pancake` | Caustic position matches analytic Zel'dovich solution |
| `spherical_collapse` | Spherical overdensity collapse dynamics |
| `conservation_laws` | Energy, momentum, C_2 conservation to tolerance |
| `landau_damping` | Damping rate matches analytic Landau rate |

Plus 2 integration tests (`smoke_test`, `end_to_end_run`) exercising the full pipeline from `Domain` through `Simulation::run()` to `ExitPackage`.

### HT tensor and ACA tests

| Test | Validates |
|---|---|
| `round_trip_rank1` | Rank-1 tensor survives HSVD round-trip |
| `gaussian_blob` | 6D Gaussian compression ratio and accuracy |
| `addition` | Rank-concatenation addition correctness |
| `density_integration` | `compute_density()` matches direct summation |
| `inner_product_and_norm` | Gram-matrix inner product accuracy |
| `truncation_accuracy` | Rank-adaptive recompression error bounds |
| `htaca_separable` | Separable f(x,v) = g(x)h(v) achieves rank 1 at root |
| `htaca_gaussian` | HTACA vs HSVD accuracy on 6D Gaussian |
| `htaca_plummer` | HTACA on Plummer-like DF |
| `htaca_scaling` | Evaluation count at multiple grid sizes |
| `htaca_density_consistency` | `compute_density()` matches between HSVD and HTACA |
| `htaca_rank_convergence` | Error decreases monotonically with max_rank |
| `aca_rank1_exact` | Exact rank-1 matrix recovery |
| `aca_rank3` | Known rank-3 matrix convergence |
| `aca_low_rank_plus_noise` | Tolerance separates signal from noise |
| `aca_gaussian_kernel` | Gaussian kernel rapid convergence |
| `cur_output` | U*V^T reproduces ACA approximation |
| `convergence_criterion` | Frobenius norm estimate tracks correctly |
| `zero_matrix` | Graceful handling of zero input |

## Feature flags

```toml
[dependencies]
caustic = { version = "0.0.4", features = ["jemalloc"] }
```

| Flag | Description |
|---|---|
| `jemalloc` | jemalloc global allocator via `tikv-jemallocator` |
| `mimalloc-alloc` | mimalloc global allocator |
| `dhat-heap` | Heap profiling via `dhat` |
| `tracy` | Tracy profiler integration via `tracing-tracy` |

## Performance

- **Parallelism**: rayon data parallelism across all hot paths (`compute_density`, `advect_x`, `advect_v`, FFT axes)
- **Release profile**: fat LTO, `codegen-units = 1`, `target-cpu=native` (via `.cargo/config.toml`)
- **Benchmarks**: criterion benchmarks (`cargo bench`), benchmark binary: `solver_kernels`
- **Instrumentation**: `tracing::info_span!` on all hot methods (zero overhead without a subscriber)
- **Profiling profile**: `[profile.profiling]` inherits release with debug symbols for `perf`/`samply`

## Roadmap

- [x] Uniform 6D grid with rayon parallelism
- [x] FFT Poisson (periodic + Hockney-Eastwood isolated)
- [x] Semi-Lagrangian advection (Catmull-Rom) + Strang/Lie/Yoshida splitting
- [x] Isolated equilibrium ICs (Plummer, King, Hernquist, NFW)
- [x] Cosmological, merger, tidal, and custom ICs
- [x] Conservation diagnostics + validation suite (30 tests)
- [x] Criterion benchmarks + tracing instrumentation
- [x] Binary snapshot I/O, CSV diagnostics, JSON checkpoints
- [x] Hierarchical Tucker (HT) tensor decomposition with HSVD compression
- [x] Adaptive Cross Approximation (ACA) for black-box low-rank matrix construction
- [x] HTACA black-box HT construction via fiber sampling (Ballani & Grasedyck 2013)
- [ ] SLAR advection in HT format (semi-Lagrangian with rank-adaptive recompression)
- [ ] Tensor-format Poisson solver
- [ ] LoMaC conservative truncation
- [ ] Lagrangian sheet tracker for cold dark matter
- [ ] Multigrid / spherical harmonics Poisson solvers
- [ ] Adaptive mesh refinement
- [ ] GPU acceleration
- [ ] MPI domain decomposition

## Companion: phasma

[phasma](https://github.com/resonant-jovian/phasma) is a ratatui-based terminal UI that consumes caustic as a library dependency. It provides interactive parameter editing, live diagnostics rendering, density/phase-space heatmaps, energy conservation plots, and radial profile charts — all from the terminal. phasma contains no solver logic; it delegates entirely to caustic.

## Minimum supported Rust version

Rust edition 2024, targeting **stable Rust 1.75+**.

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
