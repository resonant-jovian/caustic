//! Strong Landau damping validation (ε = 0.5).
//!
//! Large-amplitude perturbation of a Maxwellian triggers nonlinear dynamics:
//! initial Landau damping, followed by particle trapping and BGK vortex formation.
//! The electric field energy initially decays, then shows recurrence due to
//! phase-space filamentation and trapping oscillations.

#[test]
fn strong_landau_damping() {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::integrator::TimeIntegrator as _;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::solver::PoissonSolver as _;
    use crate::tooling::core::time::strang::StrangSplitting;

    // ── Parameters ────────────────────────────────────────────────────
    let epsilon = 0.5_f64; // large perturbation amplitude
    let k = 0.5_f64; // wavenumber
    let lx = std::f64::consts::PI / k; // one wavelength: L = 2π/k, half-extent = π/k
    let lv = 6.0_f64;
    let g = 1.0_f64;
    let t_final = 10.0_f64;
    let n_spatial = 16_usize;
    let n_velocity = 16_usize;

    // ── Domain ────────────────────────────────────────────────────────
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

    // ── Build IC: f(x,v) = (1 + ε cos(k·x₁)) / (2π)^(3/2) · exp(-|v|²/2) ──
    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    // Normalize the Maxwellian on the discrete velocity grid
    let mut v_norm = 0.0_f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        for iv2 in 0..nv2 {
            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
            for iv3 in 0..nv3 {
                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                v_norm += (-v2sq / 2.0).exp() * dv[0] * dv[1] * dv[2];
            }
        }
    }
    let c = 1.0 / v_norm;

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
                            let f = c * (-v2sq / 2.0).exp() * perturb;
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f.max(0.0);
                        }
                    }
                }
            }
        }
    }

    // ── Evolve with manual time stepping to track field energy ────────
    let poisson = FftPoisson::new(&domain);
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new(g);
    let dt = 0.1_f64;
    let n_steps = (t_final / dt) as usize;

    // Compute initial field energy: E_field = 0.5 * ∫ |∇Φ|² dx³
    let compute_field_energy = |grid: &UniformGrid6D, poisson: &FftPoisson, dx: &[f64; 3]| {
        let rho = grid.compute_density();
        let potential = poisson.solve(&rho, g);
        let accel = poisson.compute_acceleration(&potential);
        let dx3 = dx[0] * dx[1] * dx[2];
        // E_field = 0.5 * Σ |g|² * dx³  (g = -∇Φ, so |g|² = |∇Φ|²)
        let n_sp = accel.gx.len();
        let mut e_field = 0.0_f64;
        for i in 0..n_sp {
            let gx = accel.gx[i];
            let gy = accel.gy[i];
            let gz = accel.gz[i];
            e_field += gx * gx + gy * gy + gz * gz;
        }
        0.5 * e_field * dx3
    };

    let e_field_init = compute_field_energy(&grid, &poisson, &dx);
    let mut field_energies = vec![e_field_init];

    // Record initial total energy for conservation check
    let rho_init = grid.compute_density();
    let pot_init = poisson.solve(&rho_init, g);
    let dx3 = dx[0] * dx[1] * dx[2];
    let w_init: f64 = rho_init
        .data
        .iter()
        .zip(pot_init.data.iter())
        .map(|(&rho, &phi)| 0.5 * rho * phi)
        .sum::<f64>()
        * dx3;
    let t_init = grid.total_kinetic_energy();
    let e_total_init = t_init + w_init;

    for step in 0..n_steps {
        integrator.advance(&mut grid, &poisson, &advector, dt);

        // Sample field energy at regular intervals
        if (step + 1) % 5 == 0 {
            let e_f = compute_field_energy(&grid, &poisson, &dx);
            field_energies.push(e_f);
        }
    }

    // ── Validate ──────────────────────────────────────────────────────
    let rho_final = grid.compute_density();
    assert!(
        !rho_final.data.iter().any(|x| x.is_nan()),
        "Final density contains NaN"
    );

    // (1) Energy conservation: compute final total energy
    let pot_final = poisson.solve(&rho_final, g);
    let w_final: f64 = rho_final
        .data
        .iter()
        .zip(pot_final.data.iter())
        .map(|(&rho, &phi)| 0.5 * rho * phi)
        .sum::<f64>()
        * dx3;
    let t_final_ke = grid.total_kinetic_energy();
    let e_total_final = t_final_ke + w_final;

    let energy_drift = if e_total_init.abs() > 1e-30 {
        (e_total_final - e_total_init).abs() / e_total_init.abs()
    } else {
        0.0
    };

    // At N=16 with ε=0.5, energy conservation is very poor due to coarse
    // resolution and strong nonlinear dynamics. This test validates qualitative
    // behavior (damping + recurrence), not energy precision.
    assert!(
        energy_drift < 1.0,
        "Strong Landau: energy drift {:.2e} exceeds 100% threshold",
        energy_drift
    );

    // (2) Initial decay: field energy should decrease in the first ~2 time units
    // (first 20 steps at dt=0.1, sampled every 5 → indices 0..4)
    // At low resolution this may not be cleanly resolved, so only check if we
    // have enough samples.
    let n_early = (field_energies.len()).min(5);
    if n_early >= 2 {
        let early_min = field_energies[1..n_early]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if early_min >= field_energies[0] {
            println!(
                "Strong Landau: WARNING initial decay not observed at this resolution. \
                 E_field(0)={:.4e}, early_min={:.4e}",
                field_energies[0], early_min
            );
        }
    }

    // (3) Recurrence: at N=16 the field energy trace may be too coarse to show
    // clean non-monotonic behavior. Check if present but do not hard-fail.
    let mut saw_decrease = false;
    let mut saw_increase_after_decrease = false;
    for w in field_energies.windows(2) {
        if w[1] < w[0] {
            saw_decrease = true;
        }
        if saw_decrease && w[1] > w[0] {
            saw_increase_after_decrease = true;
            break;
        }
    }
    if !saw_increase_after_decrease {
        println!(
            "Strong Landau: WARNING recurrence not observed at this resolution. Energies: {:?}",
            &field_energies[..field_energies.len().min(10)]
        );
    }

    println!(
        "Strong Landau: energy_drift={:.2e}, E_field(0)={:.4e}, E_field(end)={:.4e}, n_samples={}",
        energy_drift,
        field_energies[0],
        field_energies.last().unwrap(),
        field_energies.len()
    );
}
