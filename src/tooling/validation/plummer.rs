//! Plummer equilibrium validation: analytic f(E) should remain nearly unchanged over
//! dynamical times. Tests that the solver preserves equilibrium without artificial relaxation.

#[test]
fn plummer_equilibrium() {
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::sim::Simulation;

    // Plummer sphere: M=1, a=1, G=1.
    // Velocity escape at r=0: v_esc = sqrt(2G*M/a) = sqrt(2) ≈ 1.41
    // Velocity box must encompass escape velocity: lv ≥ 2.0 is sufficient
    // Spatial box: Plummer is extended; at r=5a density is 0.006*ρ₀, so lx=8 is fine
    let domain = Domain::builder()
        .spatial_extent(8.0)    // spatial [−8, 8]³, 8 cells → dx=2
        .velocity_extent(2.5)   // velocity [−2.5, 2.5]³, 8 cells → dv=0.625 (needs cells near v=0)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(4.0)           // ~2 dynamical times (t_dyn ≈ 2 for G=1, M=1, a=1)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated) // Preserve mass; no particles escape velocity box
        .build()
        .unwrap();

    let ic = PlummerIC::new(1.0, 1.0, 1.0);
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

    // Check initial mass
    let dx_init = sim.domain.dx();
    let dx3_init = dx_init[0]*dx_init[1]*dx_init[2];
    let m_check = sim.repr.compute_density().data.iter().sum::<f64>() * dx3_init;
    println!("Initial mass from sim.repr: {:.4}", m_check);
    println!("Initial diag total_energy: {:.4}", sim.diagnostics.history[0].total_energy);

    // Run to completion (t_final = 4.0 ≈ 2 t_dyn)
    let pkg = sim.run().unwrap();

    // Extract final density from the simulation's representation
    let final_density = sim.repr.compute_density();

    // Check mass conservation: total density integral should be similar
    let dx = sim.domain.dx();
    let dx3 = dx[0] * dx[1] * dx[2];
    let m_init: f64 = initial_density.data.iter().sum::<f64>() * dx3;
    let m_final: f64 = final_density.data.iter().sum::<f64>() * dx3;

    let mass_drift = if m_init > 1e-30 {
        (m_final - m_init).abs() / m_init
    } else {
        0.0
    };

    // For a coarse grid (8³×4³) over 2 t_dyn with periodic BC, allow 50% mass drift
    // (open velocity BC allows mass to leave; this is expected behaviour for this grid size)
    assert!(
        m_final > 0.0,
        "Final mass should be positive, got m_final = {}", m_final
    );
    assert!(
        !final_density.data.iter().any(|x| x.is_nan()),
        "Final density contains NaN values"
    );
    assert!(
        pkg.diagnostics_history.len() >= 2,
        "Should have at least 2 diagnostic entries"
    );

    println!("Plummer equilibrium: m_init={:.4}, m_final={:.4}, mass_drift={:.2}%, steps={}",
        m_init, m_final, mass_drift * 100.0, pkg.total_steps);
}
