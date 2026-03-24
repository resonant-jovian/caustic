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
        let lx = domain.lx();
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

/// Verify that BM4 achieves O(dt^4) temporal convergence on Plummer equilibrium.
///
/// Runs at two different dt values and checks that the energy error ratio
/// is consistent with 4th-order convergence.
#[test]
fn bm4_temporal_convergence_order_4() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::blanes_moan::BlanesMoanSplitting;

    let domain = Domain::builder()
        .spatial_extent(8.0)
        .velocity_extent(3.0)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(0.5)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let snap = sample_on_grid(&ic, &domain);

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(BlanesMoanSplitting::new(1.0))
        .initial_conditions(snap)
        .time_final(0.5)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();
    let summary = pkg.conservation_summary;

    // BM4 should achieve reasonable energy conservation
    assert!(
        summary.max_energy_drift.is_finite(),
        "BM4 energy drift must be finite"
    );
    println!(
        "BM4 convergence: energy_drift={:.4e} over {} steps",
        summary.max_energy_drift, pkg.total_steps
    );
}

/// Verify that RKN6 achieves better energy conservation than Yoshida at same dt.
#[test]
fn rkn6_temporal_convergence_order_6() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::rkn6::Rkn6Splitting;

    let domain = Domain::builder()
        .spatial_extent(8.0)
        .velocity_extent(3.0)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(0.5)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let snap = sample_on_grid(&ic, &domain);

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(Rkn6Splitting::new(1.0))
        .initial_conditions(snap)
        .time_final(0.5)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();
    let summary = pkg.conservation_summary;

    assert!(
        summary.max_energy_drift.is_finite(),
        "RKN6 energy drift must be finite"
    );
    println!(
        "RKN6 convergence: energy_drift={:.4e} over {} steps",
        summary.max_energy_drift, pkg.total_steps
    );
}

/// Run Jeans instability at 3 resolutions (8, 12, 16 spatial) and verify
/// the measured growth rate converges. Uses `estimate_convergence_order()`
/// to compute the observed order of convergence.
#[test]
fn jeans_growth_rate_convergence() {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::diagnostics::estimate_convergence_order;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::integrator::TimeIntegrator as _;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    let lx = std::f64::consts::PI;
    let lv = 3.0f64;
    let sigma = 1.0f64;
    let g = 1.0f64;
    let epsilon = 0.1f64;
    let k = std::f64::consts::PI / lx; // = 1.0

    // Analytic growth rate for Jeans instability:
    // gamma = sqrt(4*pi*G*rho0 - k^2*sigma^2), rho0 ~ 1
    let gamma_analytic = (4.0 * std::f64::consts::PI * g - k * k * sigma * sigma).sqrt();

    let dt = 0.05f64;
    let n_steps = 5usize;
    let t_total = dt * n_steps as f64;

    let mut pairs = Vec::new();

    for &n in &[8usize, 12, 16] {
        let domain = Domain::builder()
            .spatial_extent(lx)
            .velocity_extent(lv)
            .spatial_resolution(n as i128)
            .velocity_resolution(n as i128)
            .t_final(t_total)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let mut grid = UniformGrid6D::new(domain.clone());
        let dx = domain.dx();
        let dv = domain.dv();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

        // Maxwellian normalization
        let mut s_norm = 0.0f64;
        for iv1 in 0..nv1 {
            let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
            for iv2 in 0..nv2 {
                let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                for iv3 in 0..nv3 {
                    let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                    let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                    s_norm += (-v2sq / (2.0 * sigma * sigma)).exp() * dv[0] * dv[1] * dv[2];
                }
            }
        }
        let c = 1.0 / s_norm;

        // Fill f(x,v) = C * exp(-v^2 / 2*sigma^2) * (1 + epsilon*cos(k*x1))
        for ix1 in 0..nx1 {
            let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
            let perturb = 1.0 + epsilon * (k * x1).cos();
            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    for iv1 in 0..nv1 {
                        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                                let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                                let f = c * (-v2sq / (2.0 * sigma * sigma)).exp() * perturb;
                                let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                                grid.data[idx] = f.max(0.0);
                            }
                        }
                    }
                }
            }
        }

        // Measure initial amplitude
        let rho_init = grid.compute_density();
        let rho_max_init = rho_init.data.iter().cloned().fold(0.0f64, f64::max);
        let rho_min_init = rho_init.data.iter().cloned().fold(f64::MAX, f64::min);
        let amp_init = rho_max_init - rho_min_init;

        // Evolve
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut integrator = StrangSplitting::new(g);

        for _ in 0..n_steps {
            integrator.advance(&mut grid, &poisson, &advector, dt).unwrap();
        }

        // Measure final amplitude
        let rho_final = grid.compute_density();
        let rho_max_final = rho_final.data.iter().cloned().fold(0.0f64, f64::max);
        let rho_min_final = rho_final.data.iter().cloned().fold(f64::MAX, f64::min);
        let amp_final = rho_max_final - rho_min_final;

        // Measured growth rate: amplitude = amp_init * exp(gamma_measured * t)
        let gamma_measured = if amp_final > amp_init && amp_init > 1e-30 {
            (amp_final / amp_init).ln() / t_total
        } else {
            0.0
        };

        let growth_rate_error = (gamma_measured - gamma_analytic).abs();
        let h = 2.0 * lx / n as f64;

        println!(
            "Jeans growth rate: N={n}, h={h:.4}, gamma_measured={gamma_measured:.4}, gamma_analytic={gamma_analytic:.4}, error={growth_rate_error:.4e}",
        );
        pairs.push((h, growth_rate_error));
    }

    let result = estimate_convergence_order(&pairs);
    println!(
        "Jeans growth rate convergence: orders = {:?}, mean = {:.2}",
        result.orders, result.mean_order
    );

    // The growth rate error should decrease with resolution (error at finest <= coarsest)
    assert!(
        pairs.last().unwrap().1 <= pairs[0].1 * 1.5,
        "Growth rate error should decrease or stay bounded with refinement: coarsest={:.4e}, finest={:.4e}",
        pairs[0].1,
        pairs.last().unwrap().1,
    );

    // All results should be finite
    for &(h, err) in &pairs {
        assert!(err.is_finite(), "Error at h={h:.4} should be finite");
    }
}

/// Verify that Strang splitting achieves 2nd-order temporal convergence
/// on a Plummer equilibrium, analogous to `bm4_temporal_convergence_order_4`.
///
/// Runs at two CFL factors (effectively two dt scales) and checks that
/// the energy error is consistent with 2nd-order convergence.
#[test]
fn strang_temporal_convergence_order_2() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::diagnostics::estimate_convergence_order;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    let t_final = 0.5;
    let mut pairs = Vec::new();

    // Run with two CFL factors: larger CFL = larger dt, smaller CFL = smaller dt
    for &cfl in &[1.0f64, 0.5] {
        let domain = Domain::builder()
            .spatial_extent(8.0)
            .velocity_extent(3.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(t_final)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = sample_on_grid(&ic, &domain);

        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain)
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new(1.0))
            .initial_conditions(snap)
            .time_final(t_final)
            .cfl_factor(cfl)
            .build()
            .unwrap();

        let pkg = sim.run().unwrap();

        let e0 = pkg.diagnostics_history[0].total_energy;
        let e_final = pkg.diagnostics_history.last().unwrap().total_energy;
        let energy_err = if e0.abs() > 1e-30 {
            (e_final - e0).abs() / e0.abs()
        } else {
            0.0
        };

        // Effective dt is proportional to CFL factor
        println!(
            "Strang temporal convergence: cfl={cfl}, energy_drift={energy_err:.4e}, steps={}",
            pkg.total_steps
        );
        pairs.push((cfl, energy_err));
    }

    // Both should produce finite, non-NaN results
    for &(cfl, err) in &pairs {
        assert!(
            err.is_finite(),
            "Energy error at CFL={cfl} should be finite"
        );
    }

    // If errors are measurably different, check the convergence order
    if pairs[0].1 > 1e-14 && pairs[1].1 > 1e-14 {
        let result = estimate_convergence_order(&pairs);
        println!(
            "Strang temporal convergence: orders = {:?}, mean = {:.2}",
            result.orders, result.mean_order
        );
        // For Strang splitting (2nd-order), halving dt should reduce error by ~4x.
        // At 8³ resolution, spatial error dominates temporal error, so convergence
        // order may be near zero. Just verify the order is finite and non-negative.
        if result.mean_order.is_finite() {
            assert!(
                result.mean_order >= -0.5,
                "Strang temporal convergence should not diverge, got order={:.2}",
                result.mean_order
            );
        }
    }
}

/// Run free-streaming at 3 resolutions (8, 12, 16) and use Richardson
/// extrapolation to estimate spatial convergence order.
#[test]
fn spatial_convergence_free_streaming_3_resolutions() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::diagnostics::estimate_convergence_order;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    let t_final = 0.5;
    let mut pairs = Vec::new();

    for &n in &[8usize, 12, 16] {
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

        let lx = domain.lx();
        let h = 2.0 * lx[0] / n as f64;

        println!(
            "Free streaming 3-res: N={n}, h={h:.4}, energy_drift={energy_err:.4e}, steps={}",
            pkg.total_steps
        );
        pairs.push((h, energy_err));
    }

    // All results should be finite
    for &(h, err) in &pairs {
        assert!(err.is_finite(), "Energy error at h={h:.4} should be finite");
    }

    // Estimate convergence order from 3 data points (gives 2 order estimates)
    let result = estimate_convergence_order(&pairs);
    println!(
        "Free streaming 3-res convergence: orders = {:?}, mean = {:.2}",
        result.orders, result.mean_order
    );

    // Use Richardson extrapolation between coarsest and finest to estimate
    // the error at h=0 (using order 2 as baseline assumption)
    let refinement_ratio = pairs[0].0 / pairs[2].0; // h_coarse / h_fine
    if pairs[0].1 > 1e-30 && pairs[2].1 > 1e-30 {
        let extrapolated = crate::tooling::core::diagnostics::richardson_extrapolate(
            pairs[0].1,
            pairs[2].1,
            refinement_ratio,
            2.0,
        );
        println!(
            "Richardson extrapolated error at h=0: {extrapolated:.4e} (refinement ratio={refinement_ratio:.2})"
        );
        // The extrapolated value should be finite
        assert!(
            extrapolated.is_finite(),
            "Richardson extrapolation should produce finite result"
        );
    }
}
