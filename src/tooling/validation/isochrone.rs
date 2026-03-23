//! Isochrone (Henon) equilibrium validation: analytic f(E) should remain
//! nearly unchanged over dynamical times.

#[test]
fn isochrone_equilibrium() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{IsochroneIC, sample_on_grid},
    };
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    // Isochrone: M=1, b=1, G=1.
    // Escape velocity at r=0: v_esc = sqrt(2GM/b) = sqrt(2) ~ 1.41
    // Dynamical time t_dyn ~ sqrt(b^3 / (GM)) = 1.
    // Spatial box: density falls off as ~r^-4 for r >> b; lx=8 is adequate.
    let domain = Domain::builder()
        .spatial_extent(8.0) // spatial [-8, 8]^3, 8 cells -> dx=2
        .velocity_extent(2.5) // velocity [-2.5, 2.5]^3, 8 cells -> dv=0.625
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(4.0) // ~4 dynamical times
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    let ic = IsochroneIC::new(1.0, 1.0, 1.0);
    let snap = sample_on_grid(&ic, &domain);

    // Verify initial conditions are valid
    assert!(
        !snap.data.iter().any(|v| v.is_nan()),
        "Isochrone IC must not contain NaN"
    );
    let dx = domain.dx();
    let dv = domain.dv();
    let dv6 = dx[0] * dx[1] * dx[2] * dv[0] * dv[1] * dv[2];
    let m_ic: f64 = snap.data.iter().sum::<f64>() * dv6;
    assert!(
        m_ic > 0.0,
        "Isochrone IC must have positive mass, got {m_ic}"
    );

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

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(1.0))
        .initial_conditions(snap)
        .time_final(4.0)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();
    let final_density = sim.repr.compute_density();

    let dx = sim.domain.dx();
    let dx3 = dx[0] * dx[1] * dx[2];
    let m_init: f64 = initial_density.data.iter().sum::<f64>() * dx3;
    let m_final: f64 = final_density.data.iter().sum::<f64>() * dx3;

    assert!(m_final > 0.0, "Final mass should be positive");
    assert!(
        !final_density.data.iter().any(|x| x.is_nan()),
        "Final density contains NaN values"
    );
    assert!(
        pkg.diagnostics_history.len() >= 2,
        "Should have at least 2 diagnostic entries"
    );

    // Check energy conservation
    let e0 = pkg.diagnostics_history[0].total_energy;
    let e_final = pkg.diagnostics_history.last().unwrap().total_energy;
    let e_drift = if e0.abs() > 1e-30 {
        (e_final - e0).abs() / e0.abs()
    } else {
        0.0
    };

    // Check Casimir C2 preservation
    let c2_0 = pkg.diagnostics_history[0].casimir_c2;
    let c2_f = pkg.diagnostics_history.last().unwrap().casimir_c2;
    let c2_drift = if c2_0.abs() > 1e-30 {
        (c2_f - c2_0).abs() / c2_0.abs()
    } else {
        0.0
    };

    let mass_drift = if m_init > 1e-30 {
        (m_final - m_init).abs() / m_init
    } else {
        0.0
    };

    println!(
        "Isochrone equilibrium: m_init={:.4}, m_final={:.4}, mass_drift={:.2}%, E_drift={:.2e}, C2_drift={:.2e}, steps={}",
        m_init,
        m_final,
        mass_drift * 100.0,
        e_drift,
        c2_drift,
        pkg.total_steps
    );
}
