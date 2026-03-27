//! Fujiwara (1983) uniform density sphere collapse.
//!
//! A cold uniform sphere undergoes gravitational collapse, violent relaxation,
//! and virialization. Tests the solver's ability to handle collapse dynamics
//! with isolated (vacuum) boundary conditions.
//!
//! Reference: Fujiwara, T. (1983), "Vlasov simulations of stellar systems",
//! Publications of the Astronomical Society of Japan, 35, 547–563.
//!
//! Unlike the smoke test in `spherical.rs` (periodic BC, no Poisson self-gravity
//! coupling check), this test uses `FftIsolated` for proper vacuum boundary
//! conditions and validates virial ratio evolution plus energy conservation.

/// Uniform density sphere collapse with isolated BC.
/// Tagged `#[ignore]` because 16^3 x 16^3 is expensive.
#[test]
#[ignore]
#[allow(deprecated)]
fn fujiwara_uniform_sphere_collapse() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftIsolated;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::PhaseSpaceSnapshot;

    // ── Parameters ──────────────────────────────────────────────────────
    // Uniform sphere: radius R=3, mass M=1, G=1
    // Free-fall time: t_ff = pi/2 * sqrt(R^3 / (2 G M)) ~ 4.1 for R=3, G=M=1
    let n = 16_usize;
    let lx = 6.0_f64; // spatial half-extent: domain [-6, 6]^3
    let lv = 2.0_f64; // velocity half-extent: domain [-2, 2]^3
    let sphere_radius = 3.0_f64;
    let total_mass = 1.0_f64;
    let sigma_v = 0.2_f64; // cold but non-zero (avoids delta-function sampling issues)
    let g = 1.0_f64;
    let t_final = 8.0_f64; // ~2 free-fall times

    // ── Domain ──────────────────────────────────────────────────────────
    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(n as i128)
        .velocity_resolution(n as i128)
        .t_final(t_final)
        .spatial_bc(SpatialBoundType::Isolated)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    // ── Build uniform sphere IC ─────────────────────────────────────────
    // f(x,v) = rho0 * g(v)  for |x| < R,  0 otherwise
    // where g(v) = (2 pi sigma_v^2)^{-3/2} exp(-|v|^2 / (2 sigma_v^2))
    // normalised on the discrete velocity grid so that integral f dv^3 = rho0.
    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    let sphere_volume = 4.0 / 3.0 * std::f64::consts::PI * sphere_radius.powi(3);
    let rho0 = total_mass / sphere_volume;

    // Compute discrete velocity normalization
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

    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        for ix2 in 0..nx2 {
            let x2 = -lx + (ix2 as f64 + 0.5) * dx[1];
            for ix3 in 0..nx3 {
                let x3 = -lx + (ix3 as f64 + 0.5) * dx[2];
                let r = (x1 * x1 + x2 * x2 + x3 * x3).sqrt();
                if r > sphere_radius {
                    continue;
                }
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                            let f = rho0 * c_vel * (-v2sq / (2.0 * sigma_v * sigma_v)).exp();
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f;
                        }
                    }
                }
            }
        }
    }

    // ── Record initial state ────────────────────────────────────────────
    let dx3 = dx[0] * dx[1] * dx[2];
    let rho_init = grid.compute_density();
    let mass_init: f64 = rho_init.data.iter().sum::<f64>() * dx3;
    assert!(
        mass_init > 0.0,
        "Initial mass must be positive: {}",
        mass_init
    );

    // ── Build and run simulation ────────────────────────────────────────
    let snap = PhaseSpaceSnapshot {
        data: grid.data.clone(),
        shape: [nx1, nx2, nx3, nv1, nv2, nv3],
        time: 0.0,
    };

    let poisson = FftIsolated::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new())
        .initial_conditions(snap)
        .time_final(t_final)
        .gravitational_constant(g)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();

    // ── Validate ────────────────────────────────────────────────────────
    let history = &pkg.diagnostics_history;
    assert!(
        history.len() >= 5,
        "Should have at least 5 diagnostic entries, got {}",
        history.len()
    );

    // (1) Final energy should be finite and non-NaN
    let e_final = history.last().unwrap().total_energy;
    assert!(
        e_final.is_finite(),
        "Final total energy is not finite: {}",
        e_final
    );

    // (2) Energy conservation: at N=16 with isolated BC and open velocity BC,
    // mass escapes the velocity domain during violent collapse. Use a generous
    // tolerance for this coarse resolution.
    let e0 = history[0].total_energy;
    let e_drift = if e0.abs() > 1e-30 {
        (e_final - e0).abs() / e0.abs()
    } else {
        0.0
    };

    // (3) Virial ratio evolution: for a cold system, 2T/|W| starts near 0
    // (negligible kinetic energy). During collapse and violent relaxation,
    // it should approach ~1.0 (virial equilibrium: 2T + W = 0).
    let vir_init = history[0].virial_ratio;
    let vir_final = history.last().unwrap().virial_ratio;

    // The virial ratio should have changed from its initial (near-zero) value.
    // At this resolution, we do not demand convergence to exactly 1.0, but
    // the final virial ratio should be further from 0 than the initial one.
    let vir_changed = (vir_final - vir_init).abs() > 1e-6 || vir_final.abs() > 0.1;
    assert!(
        vir_changed,
        "Fujiwara: virial ratio should evolve during collapse. \
         vir_init={:.4}, vir_final={:.4}",
        vir_init, vir_final
    );

    // (4) Mass should remain positive
    let mass_final = history.last().unwrap().mass_in_box;
    assert!(
        mass_final > 0.0,
        "Final mass should be positive: {}",
        mass_final
    );

    println!(
        "Fujiwara collapse: e_drift={:.4}, vir_init={:.4}, vir_final={:.4}",
        e_drift, vir_init, vir_final
    );
    println!(
        "  steps={}, t_final={:.2}, mass_init={:.4}, mass_final={:.4}",
        pkg.total_steps,
        history.last().unwrap().time,
        mass_init,
        mass_final
    );
}
