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

## Architecture

| Layer | Components |
|---|---|
| **Orchestration** | `TimeIntegrator` — Strang splitting, Yoshida, RK4 |
| **Core solvers** | `Advector` (semi-Lagrangian, spectral, finite volume) · `PoissonSolver` (FFT, multigrid, tree) · `PhaseSpaceRepresentation` (grid, sheet, tensor, spectral, hybrid) |
| **Supporting** | `InitialConditions` · `Diagnostics` · `IOManager` |

### Core modules

| Module | Purpose |
|---|---|
| `caustic::representation` | Phase-space storage and queries. Trait-based — implement `PhaseSpaceRepr` for your own representation. Ships with `UniformGrid6D` and (WIP) `TensorTrain`, `SheetTracker`. |
| `caustic::poisson` | 3D Poisson solvers. `FftPoisson` (periodic), `FftIsolated` (zero-padded), `Multigrid`. |
| `caustic::advect` | 6D advection step. `SemiLagrangian`, `SpectralAdvector`. |
| `caustic::integrate` | Time-stepping orchestration. `StrangSplitting`, `YoshidaSplitting`. |
| `caustic::initial` | Initial condition library. Plummer, King, Hernquist, NFW, Zeldovich cosmological, disk equilibria, custom callables. |
| `caustic::diagnostics` | Conservation monitors (energy, momentum, Casimir C₂, entropy), moment computation, stream counting. |
| `caustic::io` | HDF5 snapshot I/O and checkpoint/restart. |

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
caustic = "0.1"
```

### Minimal example: Plummer sphere equilibrium

```rust
use caustic::prelude::*;

fn main() -> Result<(), CausticError> {
    // Define the computational domain
    let domain = Domain::builder()
        .spatial_extent(20.0)      // [-20, 20]^3 in natural units
        .velocity_extent(3.0)      // [-3, 3]^3
        .spatial_resolution(64)    // 64^3 spatial grid
        .velocity_resolution(64)   // 64^3 velocity grid
        .boundary(Boundary::Isolated)
        .build()?;

    // Set up a Plummer sphere equilibrium
    let ic = PlummerModel::new()
        .mass(1.0)
        .scale_radius(1.0)
        .build();

    // Choose solver components
    let repr = UniformGrid6D::new(&domain);
    let poisson = FftIsolated::new(&domain);
    let advector = SemiLagrangian::new();
    let integrator = StrangSplitting::new();

    // Build the simulation
    let mut sim = Simulation::builder()
        .domain(domain)
        .representation(repr)
        .poisson_solver(poisson)
        .advector(advector)
        .integrator(integrator)
        .initial_conditions(ic)
        .time_final(50.0)          // 50 dynamical times
        .output_interval(1.0)
        .exit_on_energy_drift(1e-4)
        .build()?;

    // Run
    sim.run()?;

    Ok(())
}
```

### Custom initial conditions

```rust
use caustic::prelude::*;

let ic = CustomIC::from_fn(|x, v| {
    // Two overlapping Gaussians — a simple merger setup
    let r1_sq = (x[0] - 3.0).powi(2) + x[1].powi(2) + x[2].powi(2);
    let r2_sq = (x[0] + 3.0).powi(2) + x[1].powi(2) + x[2].powi(2);
    let v_sq = v[0].powi(2) + v[1].powi(2) + v[2].powi(2);
    let sigma = 0.5;

    let f1 = (-r1_sq / 2.0).exp() * (-v_sq / (2.0 * sigma * sigma)).exp();
    let f2 = (-r2_sq / 2.0).exp() * (-v_sq / (2.0 * sigma * sigma)).exp();
    f1 + f2
});
```

## Representations

The central algorithmic challenge is representing a 6D function efficiently. caustic treats this as a trait so that novel representations can be developed and benchmarked within the same framework:

```rust
pub trait PhaseSpaceRepr: Send + Sync {
    /// Integrate f over all velocities to produce the 3D density field.
    fn compute_density(&self) -> DensityField;

    /// Advect f in spatial coordinates by displacement field.
    fn advect_x(&mut self, displacement: &DisplacementField, dt: f64);

    /// Advect f in velocity coordinates by acceleration field.
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64);

    /// Compute a velocity moment of order `n` at spatial position.
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor;

    /// Total mass (integral of f over all phase space).
    fn total_mass(&self) -> f64;

    /// Casimir invariant C_2 = ∫ f² dx³ dv³.
    fn casimir_c2(&self) -> f64;
}
```

### Shipped representations

- **`UniformGrid6D`** — brute-force uniform grid. Simple, correct, useful for validation. Memory: O(N⁶).
- **`TensorTrain`** *(work in progress)* — low-rank tensor decomposition exploiting compressibility. Memory: O(N³ r³) where r is the rank.
- **`SheetTracker`** *(work in progress)* — Lagrangian sheet representation for cold (dark matter) initial conditions. Memory: O(N³).

## Validation

caustic includes a built-in test suite against known analytic solutions:

```bash
cargo test --release -- --test-threads=1
```

| Test | What it validates |
|---|---|
| `free_streaming` | Spatial advection accuracy (G=0) |
| `uniform_acceleration` | Velocity advection accuracy |
| `jeans_instability` | Self-consistent gravitational coupling |
| `plummer_equilibrium` | Long-term equilibrium preservation |
| `zeldovich_pancake` | Cosmological dynamics, caustic formation |
| `conservation` | Energy, momentum, Casimir invariant conservation |

## Feature flags

```toml
[dependencies]
caustic = { version = "0.1", features = ["hdf5", "mpi", "cosmological"] }
```

| Flag | Description |
|---|---|
| `hdf5` | Enable HDF5 snapshot I/O (requires libhdf5) |
| `mpi` | Distributed-memory parallelism via MPI |
| `cosmological` | Comoving coordinates, expansion factor, Zel'dovich ICs |
| `simd` | Explicit SIMD vectorization for advection kernels |

## Roadmap

- [ ] Uniform 6D grid representation
- [ ] FFT Poisson solver (periodic + isolated)
- [ ] Semi-Lagrangian advection with Strang splitting
- [ ] Plummer, King, Hernquist initial conditions
- [ ] Energy/momentum/Casimir conservation diagnostics
- [ ] Tensor-train representation
- [ ] Sheet-tracking representation for cold ICs
- [ ] Hybrid sheet/grid representation
- [ ] Spectral velocity-space (Hermite) representation
- [ ] Multigrid Poisson solver
- [ ] Adaptive time-stepping
- [ ] MPI domain decomposition
- [ ] Cosmological (comoving) mode
- [ ] GPU acceleration (wgpu compute shaders)

## Minimum supported Rust version

caustic targets **stable Rust 1.75+**.

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
