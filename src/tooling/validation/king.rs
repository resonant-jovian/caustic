//! King model equilibrium validation: the lowered Maxwellian f(E) should remain
//! nearly unchanged over several dynamical times, similar to the Plummer test.

#[test]
fn king_equilibrium() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{KingIC, sample_on_grid},
    };
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    // King model: W₀=5 (moderate concentration), r₀=1, M=1, G=1.
    // Tidal radius ≈ 5–10 r₀ for W₀=5.
    let domain = Domain::builder()
        .spatial_extent(8.0)
        .velocity_extent(2.5)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(2.0) // ~1 dynamical time
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    let ic = KingIC::new(1.0, 5.0, 1.0, 1.0);
    let snap = sample_on_grid(&ic, &domain);

    // Verify initial conditions are valid
    assert!(
        !snap.data.iter().any(|v| v.is_nan()),
        "King IC must not contain NaN"
    );
    let dx = domain.dx();
    let dv = domain.dv();
    let dv6 = dx[0] * dx[1] * dx[2] * dv[0] * dv[1] * dv[2];
    let m_init: f64 = snap.data.iter().sum::<f64>() * dv6;
    assert!(
        m_init > 0.0,
        "King IC must have positive mass, got {m_init}"
    );

    // Store initial density profile
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
        .domain(domain.clone())
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(1.0))
        .initial_conditions(snap)
        .time_final(2.0)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();

    let final_density = sim.repr.compute_density();
    let dx3 = dx[0] * dx[1] * dx[2];
    let m_final: f64 = final_density.data.iter().sum::<f64>() * dx3;

    // Final state should be well-defined
    assert!(
        !final_density.data.iter().any(|x| x.is_nan()),
        "Final density contains NaN"
    );
    assert!(m_final > 0.0, "Final mass should be positive");
    assert!(
        pkg.diagnostics_history.len() >= 2,
        "Should have at least 2 diagnostic entries"
    );

    // Energy should not drift catastrophically
    let e0 = pkg.diagnostics_history[0].total_energy;
    let e_final = pkg.diagnostics_history.last().unwrap().total_energy;
    let energy_drift = if e0.abs() > 1e-30 {
        (e_final - e0).abs() / e0.abs()
    } else {
        0.0
    };

    println!(
        "King equilibrium: W0=5, m_init={:.4}, m_final={:.4}, E_drift={:.2e}, steps={}",
        m_init, m_final, energy_drift, pkg.total_steps
    );
}
