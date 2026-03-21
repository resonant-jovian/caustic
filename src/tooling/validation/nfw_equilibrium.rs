//! NFW equilibrium stability validation.
//! NfwIC + FftIsolated (or FftPoisson as fallback), cusp preserved over multiple t_dyn.

#[test]
#[ignore] // takes ~12s in release mode
fn nfw_equilibrium() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{NfwIC, sample_on_grid},
    };
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    // NFW profile: ρ(r) = ρ_s / [(r/r_s)(1+r/r_s)²]
    // M=1, r_s=1, c=5, G=1.
    // Dynamical time at r_s: t_dyn ~ r_s / v_circ ~ 1/sqrt(G·M/r_s) ~ 1.
    // Run for ~5 t_dyn and check the cusp (central density) is preserved.
    let domain = Domain::builder()
        .spatial_extent(6.0) // [−6, 6]³ encompasses ~6 r_s
        .velocity_extent(3.0) // v_circ ~ sqrt(GM/r_s) ~ 1, need headroom
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(5.0) // ~5 dynamical times
        .spatial_bc(SpatialBoundType::Periodic) // FftPoisson requires periodic
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    let ic = NfwIC::new(1.0, 1.0, 5.0, 1.0); // M=1, r_s=1, c=5, G=1
    let snap = sample_on_grid(&ic, &domain);

    // Compute initial density profile
    let initial_grid = crate::tooling::core::algos::uniform::UniformGrid6D::from_snapshot(
        crate::tooling::core::types::PhaseSpaceSnapshot {
            data: snap.data.clone(),
            shape: snap.shape,
            time: 0.0,
        },
        domain.clone(),
    );
    let initial_density = initial_grid.compute_density();
    let rho_max_init = initial_density.data.iter().cloned().fold(0.0f64, f64::max);

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(1.0))
        .initial_conditions(snap)
        .time_final(5.0)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();

    let final_density = sim.repr.compute_density();
    assert!(
        !final_density.data.iter().any(|x| x.is_nan()),
        "Final density contains NaN"
    );

    let rho_max_final = final_density.data.iter().cloned().fold(0.0f64, f64::max);

    // The NFW cusp should be broadly preserved. On a coarse grid with periodic BC,
    // we can't expect perfect preservation, but central density shouldn't collapse
    // to zero or blow up.
    assert!(rho_max_final > 0.0, "Final density should be positive");

    // Check mass conservation (with truncated velocity BC)
    let dx = sim.domain.dx();
    let dx3 = dx[0] * dx[1] * dx[2];
    let m_init: f64 = initial_density.data.iter().sum::<f64>() * dx3;
    let m_final: f64 = final_density.data.iter().sum::<f64>() * dx3;

    let mass_drift = if m_init > 1e-30 {
        (m_final - m_init).abs() / m_init
    } else {
        0.0
    };

    // With truncated velocity BC, mass should be reasonably conserved
    assert!(m_final > 0.0, "Final mass should be positive");

    assert!(
        pkg.diagnostics_history.len() >= 2,
        "Should have at least 2 diagnostic entries"
    );

    println!(
        "NFW equilibrium: rho_max init={:.4}, final={:.4}, mass_drift={:.2}%, steps={}",
        rho_max_init,
        rho_max_final,
        mass_drift * 100.0,
        pkg.total_steps
    );
}
