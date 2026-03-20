//! Violent relaxation validation.
//!
//! A far-from-equilibrium initial state (cold, compact Gaussian) collapses under
//! self-gravity and virializes through violent relaxation (rapid fluctuations in
//! the mean-field potential redistribute energy among particles). The system should
//! approach virial equilibrium (2T + W → 0) while conserving total energy and mass.

#[test]
fn violent_relaxation() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::PhaseSpaceSnapshot;

    // ── Parameters ────────────────────────────────────────────────────
    let sigma_x = 0.5_f64; // compact spatial Gaussian
    let sigma_v = 0.1_f64; // cold velocity dispersion
    let g = 1.0_f64;
    let lx = 4.0_f64; // spatial half-extent
    let lv = 3.0_f64; // velocity half-extent
    let t_final = 10.0_f64;
    let n = 8_usize;

    // ── Domain ────────────────────────────────────────────────────────
    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(n as i128)
        .velocity_resolution(n as i128)
        .t_final(t_final)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    // ── Build cold Gaussian IC ────────────────────────────────────────
    // f(x,v) = A · exp(-|x|²/(2σ_x²)) · exp(-|v|²/(2σ_v²))
    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    // Normalize so total mass ≈ 1
    let mut raw_sum = 0.0_f64;
    let dx3 = dx[0] * dx[1] * dx[2];
    let dv3 = dv[0] * dv[1] * dv[2];
    let d6v = dx3 * dv3;

    // First pass: compute normalization
    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        for ix2 in 0..nx2 {
            let x2 = -lx + (ix2 as f64 + 0.5) * dx[1];
            for ix3 in 0..nx3 {
                let x3 = -lx + (ix3 as f64 + 0.5) * dx[2];
                let r2x = x1 * x1 + x2 * x2 + x3 * x3;
                let fx = (-r2x / (2.0 * sigma_x * sigma_x)).exp();
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            let r2v = v1 * v1 + v2 * v2 + v3 * v3;
                            let fv = (-r2v / (2.0 * sigma_v * sigma_v)).exp();
                            raw_sum += fx * fv;
                        }
                    }
                }
            }
        }
    }

    let norm = 1.0 / (raw_sum * d6v); // normalize to total mass ≈ 1

    // Second pass: fill grid
    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        for ix2 in 0..nx2 {
            let x2 = -lx + (ix2 as f64 + 0.5) * dx[1];
            for ix3 in 0..nx3 {
                let x3 = -lx + (ix3 as f64 + 0.5) * dx[2];
                let r2x = x1 * x1 + x2 * x2 + x3 * x3;
                let fx = (-r2x / (2.0 * sigma_x * sigma_x)).exp();
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            let r2v = v1 * v1 + v2 * v2 + v3 * v3;
                            let fv = (-r2v / (2.0 * sigma_v * sigma_v)).exp();
                            let f = norm * fx * fv;
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f;
                        }
                    }
                }
            }
        }
    }

    // ── Record initial state ──────────────────────────────────────────
    let rho_init = grid.compute_density();
    let mass_init: f64 = rho_init.data.iter().sum::<f64>() * dx3;
    assert!(mass_init > 0.0, "Initial mass must be positive: {}", mass_init);

    // ── Build and run simulation ──────────────────────────────────────
    let snap = PhaseSpaceSnapshot {
        data: grid.data.clone(),
        shape: [nx1, nx2, nx3, nv1, nv2, nv3],
        time: 0.0,
    };

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(g))
        .initial_conditions(snap)
        .time_final(t_final)
        .gravitational_constant(g)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();

    // ── Validate ──────────────────────────────────────────────────────
    let history = &pkg.diagnostics_history;
    assert!(
        history.len() >= 2,
        "Need at least 2 diagnostic entries, got {}",
        history.len()
    );

    // (1) Mass conservation
    let m0 = history[0].mass_in_box;
    let max_mass_drift = history
        .iter()
        .map(|d| (d.mass_in_box - m0).abs() / m0.abs().max(1e-30))
        .fold(0.0_f64, f64::max);

    assert!(
        max_mass_drift < 0.01,
        "Violent relaxation: mass drift too large: {:.2e} (threshold 0.01)",
        max_mass_drift
    );

    // (2) Virial ratio: |2T/W + 1| should decrease or become < 1.0
    // The initial state is far from virial equilibrium (cold, compact → |2T/W| << 1).
    // After collapse and relaxation, the system should approach |2T/|W|| → 1.
    let virial_init = history[0].virial_ratio;
    let virial_final = history.last().unwrap().virial_ratio;

    // The virial ratio 2T/|W| should approach 1 (i.e., virial equilibrium).
    // |virial - 1| should be smaller at end than at start, or at least < 1.0.
    let virial_dev_init = (virial_init - 1.0).abs();
    let virial_dev_final = (virial_final - 1.0).abs();

    let virial_improved = virial_dev_final < virial_dev_init || virial_dev_final < 1.0;
    assert!(
        virial_improved,
        "Violent relaxation: virial ratio should approach equilibrium. \
         |vir_init-1|={:.3}, |vir_final-1|={:.3}",
        virial_dev_init, virial_dev_final
    );

    // (3) Total energy conservation (generous for coarse grid)
    let e0 = history[0].total_energy;
    let max_e_drift = history
        .iter()
        .map(|d| (d.total_energy - e0).abs() / e0.abs().max(1e-30))
        .fold(0.0_f64, f64::max);

    assert!(
        max_e_drift < 0.5,
        "Violent relaxation: energy drift too large: {:.2e} (threshold 0.5)",
        max_e_drift
    );

    println!(
        "Violent relaxation: mass_drift={:.2e}, energy_drift={:.2e}, \
         virial_init={:.3}, virial_final={:.3}, steps={}",
        max_mass_drift, max_e_drift, virial_init, virial_final, pkg.total_steps
    );
}
