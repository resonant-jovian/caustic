//! Criterion benchmarks for hot solver kernels.
//!
//! Covers: density computation, advection, Poisson solvers (FFT periodic,
//! FFT isolated, TensorPoisson, Multigrid, TreePoisson, SphericalHarmonics),
//! spectral acceleration, full timesteps (Strang, Yoshida, RKEI),
//! conservation (KFVS, LoMaC, extract_moments),
//! and HT tensor operations at multiple grid sizes.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use caustic::prelude::*;
use caustic::tooling::core::algos::lagrangian::sl_shift_1d;
use caustic::tooling::core::algos::uniform::UniformGrid6D;
#[allow(deprecated)]
use caustic::{
    FftIsolated, FftPoisson, KfvsSolver, LoMaC, Multigrid, PlummerIC, RkeiIntegrator,
    SemiLagrangian, SphericalHarmonicsPoisson, StrangSplitting, TensorPoisson, TreePoisson,
    YoshidaSplitting,
};

fn make_plummer_grid(nx: i128, nv: i128) -> (UniformGrid6D, Domain) {
    let domain = Domain::builder()
        .spatial_extent(10.0)
        .velocity_extent(5.0)
        .spatial_resolution(nx)
        .velocity_resolution(nv)
        .t_final(1.0)
        .spatial_bc(caustic::SpatialBoundType::Periodic)
        .velocity_bc(caustic::VelocityBoundType::Open)
        .build()
        .unwrap();
    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let snap = caustic::tooling::core::init::isolated::sample_on_grid(&ic, &domain);
    let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
    (grid, domain)
}

fn make_isolated_domain(nx: i128, nv: i128) -> Domain {
    Domain::builder()
        .spatial_extent(10.0)
        .velocity_extent(5.0)
        .spatial_resolution(nx)
        .velocity_resolution(nv)
        .t_final(1.0)
        .spatial_bc(caustic::SpatialBoundType::Isolated)
        .velocity_bc(caustic::VelocityBoundType::Open)
        .build()
        .unwrap()
}

/// Create a Maxwellian 6D distribution for conservation benchmarks.
fn make_maxwellian_6d(
    spatial_shape: [usize; 3],
    velocity_shape: [usize; 3],
    dv: [f64; 3],
    v_min: [f64; 3],
) -> Vec<f64> {
    let [nx, ny, nz] = spatial_shape;
    let [nv1, nv2, nv3] = velocity_shape;
    let n_spatial = nx * ny * nz;
    let n_vel = nv1 * nv2 * nv3;

    let mut f = vec![0.0; n_spatial * n_vel];
    for ix in 0..n_spatial {
        for iv1 in 0..nv1 {
            for iv2 in 0..nv2 {
                for iv3 in 0..nv3 {
                    let iv = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                    let v1 = v_min[0] + (iv1 as f64 + 0.5) * dv[0];
                    let v2 = v_min[1] + (iv2 as f64 + 0.5) * dv[1];
                    let v3 = v_min[2] + (iv3 as f64 + 0.5) * dv[2];
                    let v2_total = v1 * v1 + v2 * v2 + v3 * v3;
                    f[ix * n_vel + iv] = (-v2_total / 2.0).exp();
                }
            }
        }
    }
    f
}

// ─── Density computation ────────────────────────────────────────────────────

fn bench_compute_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_density");
    for &(nx, nv) in &[(8i128, 4i128), (8, 8), (16, 8), (16, 16)] {
        let (grid, _) = make_plummer_grid(nx, nv);
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &grid,
            |b, grid| {
                b.iter(|| grid.compute_density());
            },
        );
    }
    group.finish();
}

// ─── Advection kernels ──────────────────────────────────────────────────────

fn bench_advect_x(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("advect_x");
    for &(nx, nv) in &[(8i128, 8i128), (16, 8), (16, 16)] {
        let (mut grid, domain) = make_plummer_grid(nx, nv);
        let dummy = DisplacementField {
            dx: vec![],
            dy: vec![],
            dz: vec![],
            shape: [0, 0, 0],
        };
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &poisson,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.01,
            g: 1.0,
        };
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &(),
            |b, _| {
                b.iter(|| grid.advect_x(&dummy, &ctx));
            },
        );
    }
    group.finish();
}

fn bench_advect_v(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("advect_v");
    for &(nx, nv) in &[(8i128, 8i128), (16, 8), (16, 16)] {
        let (mut grid, domain) = make_plummer_grid(nx, nv);
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &poisson,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.01,
            g: 1.0,
        };
        let density = grid.compute_density();
        let potential = poisson.solve(&density, &ctx);
        let accel = poisson.compute_acceleration(&potential);
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &accel,
            |b, accel| {
                b.iter(|| grid.advect_v(accel, &ctx));
            },
        );
    }
    group.finish();
}

// ─── Poisson solvers ────────────────────────────────────────────────────────

fn bench_fft_poisson(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("fft_poisson_periodic");
    for &n in &[8i128, 16, 32] {
        let (grid, domain) = make_plummer_grid(n, 4);
        let density = grid.compute_density();
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &poisson,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.0,
            g: 1.0,
        };
        group.bench_with_input(
            BenchmarkId::new("N", n),
            &(),
            |b, _| {
                b.iter(|| poisson.solve(&density, &ctx));
            },
        );
    }
    group.finish();
}

fn bench_fft_isolated(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("fft_poisson_isolated");
    for &n in &[8i128, 16, 32] {
        let domain = make_isolated_domain(n, 4);
        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = caustic::tooling::core::init::isolated::sample_on_grid(&ic, &domain);
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
        let density = grid.compute_density();
        let poisson = FftIsolated::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &poisson,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.0,
            g: 1.0,
        };
        group.bench_with_input(
            BenchmarkId::new("N", n),
            &(),
            |b, _| {
                b.iter(|| poisson.solve(&density, &ctx));
            },
        );
    }
    group.finish();
}

fn bench_tensor_poisson(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("tensor_poisson");
    for &n in &[8i128, 16] {
        let domain = make_isolated_domain(n, 4);
        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = caustic::tooling::core::init::isolated::sample_on_grid(&ic, &domain);
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
        let density = grid.compute_density();
        let dx = domain.dx();
        let shape = [n as usize, n as usize, n as usize];
        let solver = TensorPoisson::new(shape, dx, 1e-4, 1e-4, 15);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &solver,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.0,
            g: 1.0,
        };
        group.bench_with_input(BenchmarkId::new("N", n), &(), |b, _| {
            b.iter(|| solver.solve(&density, &ctx));
        });
    }
    group.finish();
}

fn bench_multigrid(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("multigrid");
    group.sample_size(10);
    for &n in &[8i128, 16, 32] {
        let (grid, domain) = make_plummer_grid(n, 4);
        let density = grid.compute_density();
        let mg = Multigrid::new(&domain, 4, 3);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &mg,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.0,
            g: 1.0,
        };
        group.bench_with_input(BenchmarkId::new("N", n), &(), |b, _| {
            b.iter(|| mg.solve(&density, &ctx));
        });
    }
    group.finish();
}

fn bench_tree_poisson(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("tree_poisson");
    group.sample_size(10);
    for &n in &[8i128, 16] {
        let domain = make_isolated_domain(n, 4);
        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = caustic::tooling::core::init::isolated::sample_on_grid(&ic, &domain);
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
        let density = grid.compute_density();
        let tree = TreePoisson::new(domain, 0.7);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &tree,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.0,
            g: 1.0,
        };
        group.bench_with_input(BenchmarkId::new("N", n), &(), |b, _| {
            b.iter(|| tree.solve(&density, &ctx));
        });
    }
    group.finish();
}

fn bench_spherical_poisson(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("spherical_poisson");
    group.sample_size(10);
    for &n in &[8i128, 16] {
        let (grid, domain) = make_plummer_grid(n, 4);
        let density = grid.compute_density();
        let dx = domain.dx();
        let shape = [n as usize; 3];
        let solver = SphericalHarmonicsPoisson::new(4, 32, shape, dx);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let poisson_for_ctx = FftPoisson::new(&domain);
        let ctx = SimContext {
            solver: &poisson_for_ctx,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.0,
            g: 1.0,
        };
        group.bench_with_input(BenchmarkId::new("N", n), &(), |b, _| {
            b.iter(|| solver.solve(&density, &ctx));
        });
    }
    group.finish();
}

// ─── Spectral acceleration (compute_acceleration) ───────────────────────────

fn bench_compute_acceleration(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("compute_acceleration");
    for &n in &[8i128, 16, 32] {
        let (grid, domain) = make_plummer_grid(n, 4);
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let ctx = SimContext {
            solver: &poisson,
            advector: &advector,
            emitter: &emitter,
            progress: &progress,
            step: 0,
            time: 0.0,
            dt: 0.0,
            g: 1.0,
        };
        let density = grid.compute_density();
        let potential = poisson.solve(&density, &ctx);
        group.bench_with_input(
            BenchmarkId::new("N", n),
            &(potential, poisson),
            |b, (pot, p)| {
                b.iter(|| p.compute_acceleration(pot));
            },
        );
    }
    group.finish();
}

// ─── Full timestep ──────────────────────────────────────────────────────────

fn bench_full_timestep(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("full_timestep");
    for &(nx, nv) in &[(8i128, 8i128), (16, 8), (16, 16)] {
        let (grid, domain) = make_plummer_grid(nx, nv);
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let mut integrator = StrangSplitting::new();
        let mut grid = grid;
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &(),
            |b, _| {
                b.iter(|| {
                    let ctx = SimContext {
                        solver: &poisson,
                        advector: &advector,
                        emitter: &emitter,
                        progress: &progress,
                        step: 0,
                        time: 0.0,
                        dt: 0.01,
                        g: 1.0,
                    };
                    integrator.advance(&mut grid, &ctx).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_yoshida_timestep(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("yoshida_timestep");
    group.sample_size(10);
    for &(nx, nv) in &[(8i128, 8i128)] {
        let (mut grid, domain) = make_plummer_grid(nx, nv);
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let mut integrator = YoshidaSplitting::new();
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &(),
            |b, _| {
                b.iter(|| {
                    let ctx = SimContext {
                        solver: &poisson,
                        advector: &advector,
                        emitter: &emitter,
                        progress: &progress,
                        step: 0,
                        time: 0.0,
                        dt: 0.01,
                        g: 1.0,
                    };
                    integrator.advance(&mut grid, &ctx).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_rkei_timestep(c: &mut Criterion) {
    use caustic::tooling::core::context::SimContext;
    use caustic::tooling::core::events::EventEmitter;
    use caustic::tooling::core::progress::StepProgress;

    let mut group = c.benchmark_group("rkei_timestep");
    group.sample_size(10);
    for &(nx, nv) in &[(8i128, 8i128)] {
        let (mut grid, domain) = make_plummer_grid(nx, nv);
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let mut integrator = RkeiIntegrator::new();
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &(),
            |b, _| {
                b.iter(|| {
                    let ctx = SimContext {
                        solver: &poisson,
                        advector: &advector,
                        emitter: &emitter,
                        progress: &progress,
                        step: 0,
                        time: 0.0,
                        dt: 0.01,
                        g: 1.0,
                    };
                    integrator.advance(&mut grid, &ctx).unwrap();
                });
            },
        );
    }
    group.finish();
}

// ─── 1D interpolation kernel ────────────────────────────────────────────────

fn bench_catmull_rom(c: &mut Criterion) {
    let mut group = c.benchmark_group("catmull_rom_1d");
    for &n in &[64, 256] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let dx = 1.0 / n as f64;
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| sl_shift_1d(data, 0.3, dx, n, n as f64 * dx * 0.5, true));
        });
    }
    group.finish();
}

// ─── HT tensor operations ───────────────────────────────────────────────────

fn bench_ht_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ht_construction");
    group.sample_size(10);
    let n = 8i128;
    let domain = Domain::builder()
        .spatial_extent(10.0)
        .velocity_extent(5.0)
        .spatial_resolution(n)
        .velocity_resolution(n)
        .t_final(1.0)
        .spatial_bc(caustic::SpatialBoundType::Periodic)
        .velocity_bc(caustic::VelocityBoundType::Open)
        .build()
        .unwrap();
    let _ic = PlummerIC::new(1.0, 1.0, 1.0);
    group.bench_function("aca_8x8", |b| {
        b.iter(|| {
            caustic::HtTensor::from_function_aca(
                |x, v| {
                    let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                    (-r2 / 2.0 - v2 / 2.0).exp()
                },
                &domain,
                1e-4,
                15,
                None,
                None,
            )
        });
    });
    group.finish();
}

fn bench_ht_truncation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ht_truncation");
    group.sample_size(10);
    let n = 8i128;
    let domain = Domain::builder()
        .spatial_extent(10.0)
        .velocity_extent(5.0)
        .spatial_resolution(n)
        .velocity_resolution(n)
        .t_final(1.0)
        .spatial_bc(caustic::SpatialBoundType::Periodic)
        .velocity_bc(caustic::VelocityBoundType::Open)
        .build()
        .unwrap();
    let ht = caustic::HtTensor::from_function_aca(
        |x, v| {
            let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            (-r2 / 2.0 - v2 / 2.0).exp()
        },
        &domain,
        1e-4,
        15,
        None,
        None,
    );
    group.bench_function("truncate_8x8", |b| {
        b.iter(|| {
            let mut clone = ht.clone();
            clone.truncate(1e-3);
        });
    });
    group.finish();
}

// ─── Conservation framework ─────────────────────────────────────────────────

fn bench_extract_moments(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_moments");
    for &(ns, nv) in &[(8usize, 4usize), (16, 4)] {
        let spatial_shape = [ns, ns, ns];
        let velocity_shape = [nv, nv, nv];
        let dv = [1.0; 3];
        let v_min = [-2.0; 3];
        let f = make_maxwellian_6d(spatial_shape, velocity_shape, dv, v_min);
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", ns, nv)),
            &f,
            |b, f| {
                b.iter(|| {
                    caustic::conservative_svd::extract_moments(
                        f,
                        spatial_shape,
                        velocity_shape,
                        dv,
                        v_min,
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_kfvs_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("kfvs_step");
    for &ns in &[8usize, 16] {
        let shape = [ns, ns, ns];
        let dx = [0.5; 3];
        let n = ns * ns * ns;
        let mut solver = KfvsSolver::new(shape, dx);
        // Initialize uniform thermal state
        for s in &mut solver.state {
            s.density = 1.0;
            s.momentum = [0.1, 0.0, 0.0];
            s.energy = 1.5;
        }
        let zero = vec![0.0; n];
        group.bench_with_input(BenchmarkId::new("N", ns), &zero, |b, z| {
            b.iter(|| solver.step(0.01, z, z, z));
        });
    }
    group.finish();
}

fn bench_lomac_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("lomac_projection");
    let spatial = [4usize, 4, 4];
    let velocity = [4usize, 4, 4];
    let dx = [0.5; 3];
    let dv = [1.0; 3];
    let v_min = [-2.0; 3];
    let f = make_maxwellian_6d(spatial, velocity, dv, v_min);
    let mut lomac = LoMaC::new(spatial, velocity, dx, dv, v_min);
    lomac.initialize_from_kinetic(&f);

    // Perturb f to simulate truncation
    let f_damaged: Vec<f64> = f
        .iter()
        .enumerate()
        .map(|(i, &v)| v * (1.0 + 0.2 * (i as f64 * 1.3).sin()))
        .collect();

    group.bench_function("4x4", |b| {
        b.iter(|| lomac.project(&f_damaged));
    });
    group.finish();
}

// ─── Criterion group registration ───────────────────────────────────────────

criterion_group!(
    benches,
    bench_compute_density,
    bench_advect_x,
    bench_advect_v,
    bench_fft_poisson,
    bench_fft_isolated,
    bench_tensor_poisson,
    bench_multigrid,
    bench_tree_poisson,
    bench_spherical_poisson,
    bench_compute_acceleration,
    bench_full_timestep,
    bench_yoshida_timestep,
    bench_rkei_timestep,
    bench_catmull_rom,
    bench_ht_construction,
    bench_ht_truncation,
    bench_extract_moments,
    bench_kfvs_step,
    bench_lomac_projection,
);
criterion_main!(benches);
