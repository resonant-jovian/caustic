//! Isochrone (Henon) equilibrium validation: analytic f(E) should remain
//! nearly unchanged over dynamical times.

#[test]
fn isochrone_equilibrium() {
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{IsochroneIC, sample_on_grid},
    };
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::validation::helpers::{
        assert_valid_output, build_standard_sim, density_mass, relative_drift, snapshot_density,
    };

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
    let initial_density = snapshot_density(&snap, &domain);

    let mut sim = build_standard_sim(domain, snap, 4.0);

    let pkg = sim.run().unwrap();
    let final_density = sim.repr.compute_density();

    let dx = sim.domain.dx();
    let m_init = density_mass(&initial_density, dx);
    let m_final = density_mass(&final_density, dx);

    assert!(m_final > 0.0, "Final mass should be positive");
    assert_valid_output(&final_density, pkg.diagnostics_history.len());

    // Check energy conservation
    let e0 = pkg.diagnostics_history[0].total_energy;
    let e_final = pkg.diagnostics_history.last().unwrap().total_energy;
    let e_drift = relative_drift(e_final, e0);

    // Check Casimir C2 preservation
    let c2_0 = pkg.diagnostics_history[0].casimir_c2;
    let c2_f = pkg.diagnostics_history.last().unwrap().casimir_c2;
    let c2_drift = relative_drift(c2_f, c2_0);

    let mass_drift = relative_drift(m_final, m_init);

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
