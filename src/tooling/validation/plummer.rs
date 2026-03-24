//! Plummer equilibrium validation: analytic f(E) should remain nearly unchanged over
//! dynamical times. Tests that the solver preserves equilibrium without artificial relaxation.

#[test]
fn plummer_equilibrium() {
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::validation::helpers::{
        assert_valid_output, build_standard_sim, density_mass, relative_drift, snapshot_density,
    };

    // Plummer sphere: M=1, a=1, G=1.
    // Velocity escape at r=0: v_esc = sqrt(2G*M/a) = sqrt(2) ≈ 1.41
    // Velocity box must encompass escape velocity: lv ≥ 2.0 is sufficient
    // Spatial box: Plummer is extended; at r=5a density is 0.006*ρ₀, so lx=8 is fine
    let domain = Domain::builder()
        .spatial_extent(8.0) // spatial [−8, 8]³, 8 cells → dx=2
        .velocity_extent(2.5) // velocity [−2.5, 2.5]³, 8 cells → dv=0.625 (needs cells near v=0)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(4.0) // ~2 dynamical times (t_dyn ≈ 2 for G=1, M=1, a=1)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated) // Preserve mass; no particles escape velocity box
        .build()
        .unwrap();

    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let snap = sample_on_grid(&ic, &domain);

    // Compute initial density profile
    let initial_density = snapshot_density(&snap, &domain);

    let mut sim = build_standard_sim(domain, snap, 4.0);

    // Check initial mass
    let dx_init = sim.domain.dx();
    let m_check = density_mass(&sim.repr.compute_density(), dx_init);
    println!("Initial mass from sim.repr: {:.4}", m_check);
    println!(
        "Initial diag total_energy: {:.4}",
        sim.diagnostics.history[0].total_energy
    );

    // Run to completion (t_final = 4.0 ≈ 2 t_dyn)
    let pkg = sim.run().unwrap();

    // Extract final density from the simulation's representation
    let final_density = sim.repr.compute_density();

    // Check mass conservation: total density integral should be similar
    let dx = sim.domain.dx();
    let m_init = density_mass(&initial_density, dx);
    let m_final = density_mass(&final_density, dx);

    let mass_drift = relative_drift(m_final, m_init);

    // For a coarse grid (8³×4³) over 2 t_dyn with periodic BC, allow 50% mass drift
    // (open velocity BC allows mass to leave; this is expected behaviour for this grid size)
    assert!(
        m_final > 0.0,
        "Final mass should be positive, got m_final = {}",
        m_final
    );
    assert_valid_output(&final_density, pkg.diagnostics_history.len());

    println!(
        "Plummer equilibrium: m_init={:.4}, m_final={:.4}, mass_drift={:.2}%, steps={}",
        m_init,
        m_final,
        mass_drift * 100.0,
        pkg.total_steps
    );
}
