//! Criterion benchmarks for hot solver kernels.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use caustic::prelude::*;
use caustic::tooling::core::algos::lagrangian::sl_shift_1d;
use caustic::tooling::core::algos::uniform::UniformGrid6D;
use caustic::{FftPoisson, PlummerIC, SemiLagrangian, StrangSplitting};

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

fn bench_compute_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_density");
    for &(nx, nv) in &[(8i128, 4i128), (8, 8)] {
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

fn bench_advect_x(c: &mut Criterion) {
    let (mut grid, _) = make_plummer_grid(8, 8);
    let dummy = DisplacementField {
        dx: vec![],
        dy: vec![],
        dz: vec![],
        shape: [0, 0, 0],
    };
    c.bench_function("advect_x_8x8", |b| {
        b.iter(|| grid.advect_x(&dummy, 0.01));
    });
}

fn bench_advect_v(c: &mut Criterion) {
    let (mut grid, domain) = make_plummer_grid(8, 8);
    let poisson = FftPoisson::new(&domain);
    let density = grid.compute_density();
    let potential = poisson.solve(&density, 1.0);
    let accel = poisson.compute_acceleration(&potential);
    c.bench_function("advect_v_8x8", |b| {
        b.iter(|| grid.advect_v(&accel, 0.01));
    });
}

fn bench_fft_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_poisson");
    for &n in &[8i128, 16] {
        let domain = Domain::builder()
            .spatial_extent(10.0)
            .velocity_extent(5.0)
            .spatial_resolution(n)
            .velocity_resolution(4)
            .t_final(1.0)
            .spatial_bc(caustic::SpatialBoundType::Periodic)
            .velocity_bc(caustic::VelocityBoundType::Open)
            .build()
            .unwrap();
        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = caustic::tooling::core::init::isolated::sample_on_grid(&ic, &domain);
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
        let density = grid.compute_density();
        let poisson = FftPoisson::new(&domain);
        group.bench_with_input(
            BenchmarkId::new("N", n),
            &(density, poisson),
            |b, (density, poisson)| {
                b.iter(|| poisson.solve(density, 1.0));
            },
        );
    }
    group.finish();
}

fn bench_full_timestep(c: &mut Criterion) {
    let (grid, domain) = make_plummer_grid(8, 8);
    let poisson = FftPoisson::new(&domain);
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new(1.0);
    let mut grid = grid;
    c.bench_function("full_timestep_8x8", |b| {
        b.iter(|| {
            integrator.advance(&mut grid, &poisson, &advector, 0.01);
        });
    });
}

fn bench_catmull_rom(c: &mut Criterion) {
    let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("sl_shift_1d_64", |b| {
        b.iter(|| sl_shift_1d(&data, 0.3, 0.15625, 64, 5.0, true));
    });
}

criterion_group!(
    benches,
    bench_compute_density,
    bench_advect_x,
    bench_advect_v,
    bench_fft_poisson,
    bench_full_timestep,
    bench_catmull_rom
);
criterion_main!(benches);
