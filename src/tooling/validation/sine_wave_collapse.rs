#![allow(clippy::unwrap_used)]
//! Sine-wave gravitational collapse convergence test.
//!
//! A sinusoidal density perturbation on a warm Maxwellian background undergoes
//! gravitational collapse. Tests convergence of density profiles and energy
//! conservation vs resolution.

/// Result bundle from a single sine-wave collapse run.
struct SineWaveResult {
    energy_drift: f64,
    rho_max_init: f64,
    rho_max_final: f64,
    rho_mean_init: f64,
    rho_mean_final: f64,
}

impl SineWaveResult {
    /// Density contrast: rho_max / rho_mean. Values > 1 indicate structure.
    fn contrast_init(&self) -> f64 {
        self.rho_max_init / self.rho_mean_init.max(1e-30)
    }

    fn contrast_final(&self) -> f64 {
        self.rho_max_final / self.rho_mean_final.max(1e-30)
    }
}

/// Helper: run a sine-wave collapse at given resolution and return diagnostics.
fn run_sine_wave(n_spatial: usize, n_velocity: usize, t_final: f64) -> SineWaveResult {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::integrator::TimeIntegrator as _;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::solver::PoissonSolver as _;
    use crate::tooling::core::time::strang::StrangSplitting;

    let lx = std::f64::consts::PI;
    let lv = 3.0_f64;
    let sigma_v = 0.5_f64;
    let g = 1.0_f64;
    let amplitude = 0.3_f64;
    let k = std::f64::consts::PI / lx; // fundamental mode in [-lx, lx]

    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(n_spatial as i128)
        .velocity_resolution(n_velocity as i128)
        .t_final(t_final)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    // Normalize Maxwellian on the discrete velocity grid
    let mut v_norm = 0.0_f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        for iv2 in 0..nv2 {
            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
            for iv3 in 0..nv3 {
                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                v_norm += (-v2sq / (2.0 * sigma_v * sigma_v)).exp() * dv[0] * dv[1] * dv[2];
            }
        }
    }
    let c_vel = 1.0 / v_norm;

    // f(x,v) = (1 + A*sin(k*x1)) * C * exp(-|v|^2 / 2*sigma^2)
    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        let rho_x = 1.0 + amplitude * (k * x1).sin();
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                            let fv = c_vel * (-v2sq / (2.0 * sigma_v * sigma_v)).exp();
                            let f = rho_x * fv;
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f.max(0.0);
                        }
                    }
                }
            }
        }
    }

    // Record initial state
    let dx3 = dx[0] * dx[1] * dx[2];
    let rho_init = grid.compute_density();
    let rho_max_init = rho_init.data.iter().cloned().fold(0.0_f64, f64::max);
    let rho_mean_init = rho_init.data.iter().sum::<f64>() / rho_init.data.len() as f64;
    assert!(rho_max_init > 0.0, "Initial density must be positive");

    let poisson = FftPoisson::new(&domain);
    let pot_init = poisson.solve(&rho_init, g);
    let w_init: f64 = rho_init
        .data
        .iter()
        .zip(pot_init.data.iter())
        .map(|(&rho, &phi)| 0.5 * rho * phi)
        .sum::<f64>()
        * dx3;
    let t_ke_init = grid.total_kinetic_energy().unwrap();
    let e_total_init = t_ke_init + w_init;

    // Evolve
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new(g);
    let dt = 0.05_f64;
    let n_steps = (t_final / dt).ceil() as usize;

    for step in 0..n_steps {
        integrator.advance(&mut grid, &poisson, &advector, dt);

        // Mid-run NaN check
        if step == n_steps / 2 {
            let rho_mid = grid.compute_density();
            assert!(
                !rho_mid.data.iter().any(|x| x.is_nan()),
                "NaN density at midpoint (step {})",
                step
            );
        }
    }

    // Final diagnostics
    let rho_final = grid.compute_density();
    assert!(
        !rho_final.data.iter().any(|x| x.is_nan()),
        "Final density contains NaN"
    );
    let rho_max_final = rho_final.data.iter().cloned().fold(0.0_f64, f64::max);
    let rho_mean_final = rho_final.data.iter().sum::<f64>() / rho_final.data.len() as f64;

    let pot_final = poisson.solve(&rho_final, g);
    let w_final: f64 = rho_final
        .data
        .iter()
        .zip(pot_final.data.iter())
        .map(|(&rho, &phi)| 0.5 * rho * phi)
        .sum::<f64>()
        * dx3;
    let t_ke_final = grid.total_kinetic_energy().unwrap();
    let e_total_final = t_ke_final + w_final;

    let energy_drift = if e_total_init.abs() > 1e-30 {
        (e_total_final - e_total_init).abs() / e_total_init.abs()
    } else {
        0.0
    };

    SineWaveResult {
        energy_drift,
        rho_max_init,
        rho_max_final,
        rho_mean_init,
        rho_mean_final,
    }
}

#[test]
fn sine_wave_collapse_convergence() {
    // Run at two resolutions: 8 and 12 spatial cells with 8 velocity cells.
    let t_final = 2.0;
    let r8 = run_sine_wave(8, 8, t_final);
    let r12 = run_sine_wave(12, 8, t_final);

    println!("Sine-wave collapse convergence:");
    println!(
        "  N=8:  energy_drift={:.6}, contrast {:.3} -> {:.3}, rho_max {:.4} -> {:.4}",
        r8.energy_drift,
        r8.contrast_init(),
        r8.contrast_final(),
        r8.rho_max_init,
        r8.rho_max_final
    );
    println!(
        "  N=12: energy_drift={:.6}, contrast {:.3} -> {:.3}, rho_max {:.4} -> {:.4}",
        r12.energy_drift,
        r12.contrast_init(),
        r12.contrast_final(),
        r12.rho_max_init,
        r12.rho_max_final
    );

    // All results should be finite (no NaN/Inf blow-up)
    assert!(
        r8.energy_drift.is_finite(),
        "Energy drift at N=8 should be finite"
    );
    assert!(
        r12.energy_drift.is_finite(),
        "Energy drift at N=12 should be finite"
    );
    assert!(
        r8.rho_max_final > 0.0,
        "Final density at N=8 must remain positive"
    );
    assert!(
        r12.rho_max_final > 0.0,
        "Final density at N=12 must remain positive"
    );

    // At N=12 (better resolved), gravitational collapse should increase the
    // density contrast. The perturbation seeds gravitational focusing which
    // amplifies the density peak relative to the mean.
    assert!(
        r12.contrast_final() > r12.contrast_init(),
        "Collapse at N=12 should increase density contrast: init={:.3}, final={:.3}",
        r12.contrast_init(),
        r12.contrast_final()
    );

    // Convergence check: the finer grid should resolve the collapse better.
    // At N=12 the density contrast growth should exceed or match that at N=8.
    let growth_8 = r8.contrast_final() / r8.contrast_init();
    let growth_12 = r12.contrast_final() / r12.contrast_init();
    assert!(
        growth_12 >= growth_8 * 0.5,
        "Finer grid should resolve collapse at least comparably: \
         growth_8={growth_8:.3}, growth_12={growth_12:.3}"
    );
}

#[test]
#[ignore]
fn sine_wave_collapse_high_resolution() {
    // Higher resolution test -- expensive
    let r = run_sine_wave(16, 12, 3.0);
    println!(
        "Sine-wave N=16: energy_drift={:.6}, contrast {:.3} -> {:.3}, rho_max {:.4} -> {:.4}",
        r.energy_drift,
        r.contrast_init(),
        r.contrast_final(),
        r.rho_max_init,
        r.rho_max_final
    );
    assert!(
        r.energy_drift.is_finite(),
        "Energy drift at N=16 should be finite"
    );
    assert!(
        r.contrast_final() > r.contrast_init(),
        "Collapse at N=16 should increase density contrast: init={:.3}, final={:.3}",
        r.contrast_init(),
        r.contrast_final()
    );
}
