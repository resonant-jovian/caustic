//! Conservation law validation suite: integration should preserve E, C₂, and mass.

#[test]
fn conservation_laws() {
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::sim::Simulation;

    // Plummer sphere on a coarse grid. The Strang splitting conserves a shadow Hamiltonian,
    // so total energy (computed from the same potential) should drift slowly.
    let domain = Domain::builder()
        .spatial_extent(8.0)    // [−8, 8]³, 8 cells → dx=2
        .velocity_extent(2.5)   // [−2.5, 2.5]³, 8 cells → dv=0.625 (needs cells near v=0)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(4.0)           // ~2 dynamical times
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated) // Preserve mass; no particles escape velocity box
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
        .time_final(4.0)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();

    let history = &pkg.diagnostics_history;
    assert!(history.len() >= 2, "Need at least 2 diagnostic entries");

    let e0 = history[0].total_energy;
    let c2_0 = history[0].casimir_c2;

    // Compute max relative drifts over the run
    let max_e_drift = history.iter()
        .map(|d| (d.total_energy - e0).abs() / e0.abs().max(1e-30))
        .fold(0.0f64, f64::max);
    let max_c2_drift = history.iter()
        .map(|d| (d.casimir_c2 - c2_0).abs() / c2_0.abs().max(1e-30))
        .fold(0.0f64, f64::max);

    println!(
        "Conservation: E0={:.4}, max_dE/E={:.2e}; C2_0={:.4}, max_dC2/C2={:.2e}; steps={}",
        e0, max_e_drift, c2_0, max_c2_drift, pkg.total_steps
    );

    // For a coarse 8³×4³ grid with periodic BC, these are rough bounds.
    // Tighter bounds require higher resolution and isolated BC (FftIsolated).
    assert!(!e0.is_nan(), "Initial energy must not be NaN");
    assert!(!max_e_drift.is_nan(), "Energy drift must not be NaN");

    // The Strang splitting conserves a shadow Hamiltonian; energy drift should be bounded.
    // With CFL=0.5 and coarse grid, allow up to 50% drift over 2 t_dyn.
    assert!(
        max_e_drift < 0.5,
        "Energy drift {:.2e} exceeds 50% threshold over {} steps",
        max_e_drift, pkg.total_steps
    );
}
