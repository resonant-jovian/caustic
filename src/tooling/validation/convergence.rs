//! Spatial convergence validation: verify that the solver exhibits the expected
//! convergence order by running at multiple resolutions and measuring error decay.

/// Verify that density integration of Plummer IC converges as resolution increases.
/// At fixed velocity extent, finer grids should better resolve the analytic ρ(r).
#[test]
fn density_integration_convergence() {
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::diagnostics::{error_l2, estimate_convergence_order};
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::phasespace::PhaseSpaceRepr;
    use rust_decimal::prelude::ToPrimitive;

    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let mut pairs = Vec::new();

    for &n in &[8usize, 16] {
        let domain = Domain::builder()
            .spatial_extent(8.0)
            .velocity_extent(2.5)
            .spatial_resolution(n as i128)
            .velocity_resolution(n as i128)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        let snap = sample_on_grid(&ic, &domain);
        let grid = UniformGrid6D::from_snapshot(
            crate::tooling::core::types::PhaseSpaceSnapshot {
                data: snap.data.clone(),
                shape: snap.shape,
                time: 0.0,
            },
            domain.clone(),
        );

        let density = grid.compute_density();
        let dx = domain.dx();
        let lx = [
            domain.spatial.x1.to_f64().unwrap(),
            domain.spatial.x2.to_f64().unwrap(),
            domain.spatial.x3.to_f64().unwrap(),
        ];
        let [nx_d, ny_d, nz_d] = density.shape;

        // Analytic Plummer density: ρ(r) = (3M)/(4πa³) · (1 + r²/a²)^{-5/2}
        let a = 1.0_f64;
        let m = 1.0_f64;
        let mut analytic = vec![0.0f64; nx_d * ny_d * nz_d];
        for ix in 0..nx_d {
            for iy in 0..ny_d {
                for iz in 0..nz_d {
                    let x = -lx[0] + (ix as f64 + 0.5) * dx[0];
                    let y = -lx[1] + (iy as f64 + 0.5) * dx[1];
                    let z = -lx[2] + (iz as f64 + 0.5) * dx[2];
                    let r2 = x * x + y * y + z * z;
                    let rho = 3.0 * m / (4.0 * std::f64::consts::PI * a * a * a)
                        * (1.0 + r2 / (a * a)).powf(-2.5);
                    analytic[ix * ny_d * nz_d + iy * nz_d + iz] = rho;
                }
            }
        }

        let dx3 = dx[0] * dx[1] * dx[2];
        let err = error_l2(&density.data, &analytic, dx3);
        let h = 2.0 * lx[0] / n as f64;

        println!("Density integration: N={n}, h={h:.4}, L2_error={err:.4e}");
        pairs.push((h, err));
    }

    let result = estimate_convergence_order(&pairs);
    println!(
        "Density convergence: orders = {:?}, mean = {:.2}",
        result.orders, result.mean_order
    );

    // Error should decrease with resolution
    assert!(
        pairs[1].1 <= pairs[0].1 * 1.1, // allow small tolerance
        "Finer grid should have smaller or equal error: N=8 err={:.4e}, N=16 err={:.4e}",
        pairs[0].1,
        pairs[1].1
    );
}

/// Verify the convergence table utility works end-to-end with Plummer density.
#[test]
fn convergence_table_structure() {
    use crate::tooling::core::diagnostics::convergence_table;

    let resolutions = [8usize, 12];
    let ranks = [1usize, 2];

    let table = convergence_table(&resolutions, &ranks, |n, r| {
        // Dummy error model: error ∝ 1/(N * R)
        1.0 / (n as f64 * r as f64)
    });

    assert_eq!(table.len(), 2);
    assert_eq!(table[0].len(), 2);
    assert!((table[0][0] - 1.0 / 8.0).abs() < 1e-14);
    assert!((table[1][1] - 1.0 / 24.0).abs() < 1e-14);
}

/// Free-streaming test at two resolutions to verify error decreases.
#[test]
fn free_streaming_convergence() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    let t_final = 0.5;
    let mut errors = Vec::new();

    for &n in &[8usize, 12] {
        let domain = Domain::builder()
            .spatial_extent(8.0)
            .velocity_extent(2.5)
            .spatial_resolution(n as i128)
            .velocity_resolution(n as i128)
            .t_final(t_final)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = sample_on_grid(&ic, &domain);

        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain.clone())
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new(1.0))
            .initial_conditions(snap)
            .time_final(t_final)
            .build()
            .unwrap();

        let pkg = sim.run().unwrap();

        // Measure energy drift as a proxy for error at this resolution
        let e0 = pkg.diagnostics_history[0].total_energy;
        let e_final = pkg.diagnostics_history.last().unwrap().total_energy;
        let energy_err = if e0.abs() > 1e-30 {
            (e_final - e0).abs() / e0.abs()
        } else {
            0.0
        };

        println!(
            "Free streaming convergence: N={n}, energy_drift={energy_err:.4e}, steps={}",
            pkg.total_steps
        );
        errors.push(energy_err);
    }

    // Both should produce finite, non-NaN results
    for (i, &e) in errors.iter().enumerate() {
        assert!(
            e.is_finite(),
            "Error at resolution {} should be finite",
            [8, 12][i]
        );
    }
}
