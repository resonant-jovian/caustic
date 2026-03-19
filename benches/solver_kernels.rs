//! Criterion benchmarks for hot solver kernels.
//!
//! Covers: density computation, advection, Poisson solvers (FFT periodic,
//! FFT isolated, TensorPoisson), spectral acceleration, full timesteps,
//! and HT tensor operations at multiple grid sizes.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use caustic::prelude::*;
use caustic::tooling::core::algos::lagrangian::sl_shift_1d;
use caustic::tooling::core::algos::uniform::UniformGrid6D;
use caustic::{
    FftIsolated, FftPoisson, PlummerIC, SemiLagrangian, StrangSplitting, TensorPoisson,
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
    let mut group = c.benchmark_group("advect_x");
    for &(nx, nv) in &[(8i128, 8i128), (16, 8)] {
        let (mut grid, _) = make_plummer_grid(nx, nv);
        let dummy = DisplacementField {
            dx: vec![],
            dy: vec![],
            dz: vec![],
            shape: [0, 0, 0],
        };
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &(),
            |b, _| {
                b.iter(|| grid.advect_x(&dummy, 0.01));
            },
        );
    }
    group.finish();
}

fn bench_advect_v(c: &mut Criterion) {
    let mut group = c.benchmark_group("advect_v");
    for &(nx, nv) in &[(8i128, 8i128), (16, 8)] {
        let (mut grid, domain) = make_plummer_grid(nx, nv);
        let poisson = FftPoisson::new(&domain);
        let density = grid.compute_density();
        let potential = poisson.solve(&density, 1.0);
        let accel = poisson.compute_acceleration(&potential);
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &accel,
            |b, accel| {
                b.iter(|| grid.advect_v(accel, 0.01));
            },
        );
    }
    group.finish();
}

// ─── Poisson solvers ────────────────────────────────────────────────────────

fn bench_fft_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_poisson_periodic");
    for &n in &[8i128, 16, 32] {
        let (grid, domain) = make_plummer_grid(n, 4);
        let density = grid.compute_density();
        let poisson = FftPoisson::new(&domain);
        group.bench_with_input(BenchmarkId::new("N", n), &(density, poisson), |b, (d, p)| {
            b.iter(|| p.solve(d, 1.0));
        });
    }
    group.finish();
}

fn bench_fft_isolated(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_poisson_isolated");
    for &n in &[8i128, 16, 32] {
        let domain = make_isolated_domain(n, 4);
        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = caustic::tooling::core::init::isolated::sample_on_grid(&ic, &domain);
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
        let density = grid.compute_density();
        let poisson = FftIsolated::new(&domain);
        group.bench_with_input(BenchmarkId::new("N", n), &(density, poisson), |b, (d, p)| {
            b.iter(|| p.solve(d, 1.0));
        });
    }
    group.finish();
}

fn bench_tensor_poisson(c: &mut Criterion) {
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
        group.bench_with_input(
            BenchmarkId::new("N", n),
            &(density, solver),
            |b, (d, s)| {
                b.iter(|| s.solve(d, 1.0));
            },
        );
    }
    group.finish();
}

// ─── Spectral acceleration (compute_acceleration) ───────────────────────────

fn bench_compute_acceleration(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_acceleration");
    for &n in &[8i128, 16, 32] {
        let (grid, domain) = make_plummer_grid(n, 4);
        let poisson = FftPoisson::new(&domain);
        let density = grid.compute_density();
        let potential = poisson.solve(&density, 1.0);
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
    let mut group = c.benchmark_group("full_timestep");
    for &(nx, nv) in &[(8i128, 8i128), (16, 8)] {
        let (grid, domain) = make_plummer_grid(nx, nv);
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut integrator = StrangSplitting::new(1.0);
        let mut grid = grid;
        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", nx, nv)),
            &(),
            |b, _| {
                b.iter(|| {
                    integrator.advance(&mut grid, &poisson, &advector, 0.01);
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

criterion_group!(
    benches,
    bench_compute_density,
    bench_advect_x,
    bench_advect_v,
    bench_fft_poisson,
    bench_fft_isolated,
    bench_tensor_poisson,
    bench_compute_acceleration,
    bench_full_timestep,
    bench_catmull_rom,
    bench_ht_construction,
    bench_ht_truncation,
);
criterion_main!(benches);
