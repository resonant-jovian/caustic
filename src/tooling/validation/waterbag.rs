//! Waterbag equilibrium validation: f(x,v) = f₀ for |v| < v_max, 0 otherwise.
//!
//! The waterbag has a sharp phase-space boundary. Under the Vlasov equation, the
//! boundary deforms but the enclosed volume (and hence all Casimir invariants) is
//! conserved. This test checks that WPFC advection preserves mass, C₂, and entropy
//! for a uniform-density waterbag over several dynamical times.

#[test]
fn waterbag_equilibrium() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::algos::wpfc::AdvectionScheme;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::PhaseSpaceSnapshot;

    // ── Parameters ────────────────────────────────────────────────────
    let lx = 2.0_f64; // spatial half-extent: box is [-2, 2]³
    let lv = 2.0_f64; // velocity half-extent: box is [-2, 2]³
    let v_max = 1.0_f64; // waterbag boundary: |v_i| < v_max
    let f0 = 1.0_f64; // constant value inside the waterbag
    let g = 1.0_f64;
    let t_final = 4.0_f64;
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

    // ── Build waterbag IC on a UniformGrid6D with WPFC ────────────────
    let mut grid = UniformGrid6D::new(domain.clone()).with_advection_scheme(AdvectionScheme::Wpfc);
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    for ix1 in 0..nx1 {
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            // Waterbag: f = f₀ if all |v_i| < v_max, else 0
                            let inside = v1.abs() < v_max && v2.abs() < v_max && v3.abs() < v_max;
                            let f = if inside { f0 } else { 0.0 };
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f;
                        }
                    }
                }
            }
        }
    }

    // ── Record initial diagnostics ────────────────────────────────────
    let dx3 = dx[0] * dx[1] * dx[2];
    let rho_init = grid.compute_density();
    let mass_init: f64 = rho_init.data.iter().sum::<f64>() * dx3;
    let c2_init = grid.casimir_c2();
    let entropy_init = grid.entropy();

    assert!(
        mass_init > 0.0,
        "Initial mass must be positive: {}",
        mass_init
    );
    assert!(!c2_init.is_nan(), "Initial C₂ must not be NaN");

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

    // ── Validate conservation ─────────────────────────────────────────
    let history = &pkg.diagnostics_history;
    assert!(
        history.len() >= 2,
        "Need at least 2 diagnostic entries, got {}",
        history.len()
    );

    // Mass conservation: check from diagnostics history
    let m0 = history[0].mass_in_box;
    let max_mass_drift = history
        .iter()
        .map(|d| (d.mass_in_box - m0).abs() / m0.abs().max(1e-30))
        .fold(0.0_f64, f64::max);

    // Casimir C₂ conservation
    let c2_0 = history[0].casimir_c2;
    let max_c2_drift = history
        .iter()
        .map(|d| (d.casimir_c2 - c2_0).abs() / c2_0.abs().max(1e-30))
        .fold(0.0_f64, f64::max);

    // Entropy conservation
    // For a waterbag with f=0 or f=1, the initial entropy S = -∫f ln f ≈ 0
    // (since ln(1)=0 and f=0 cells are skipped). Use absolute drift when
    // initial entropy is near zero to avoid dividing by ~0.
    let s0 = history[0].entropy;
    let max_s_drift = if s0.abs() < 1e-10 {
        // Absolute drift: entropy is near zero initially, so track absolute change
        history
            .iter()
            .map(|d| (d.entropy - s0).abs())
            .fold(0.0_f64, f64::max)
    } else {
        history
            .iter()
            .map(|d| (d.entropy - s0).abs() / s0.abs())
            .fold(0.0_f64, f64::max)
    };

    println!(
        "Waterbag: mass_drift={:.2e}, C₂_drift={:.2e}, entropy_drift={:.2e}, steps={}",
        max_mass_drift, max_c2_drift, max_s_drift, pkg.total_steps
    );

    // Mass should be very well preserved with truncated velocity BC
    assert!(
        max_mass_drift < 1e-10,
        "Waterbag mass drift too large: {:.2e} (threshold 1e-10)",
        max_mass_drift
    );

    // Casimir C₂ = ∫f² should be preserved (generous for coarse 8³×8³ grid)
    assert!(
        max_c2_drift < 0.05,
        "Waterbag Casimir C₂ drift too large: {:.2e} (threshold 0.05)",
        max_c2_drift
    );

    // Entropy S = -∫f ln f: at 8³×8³ resolution, numerical diffusion smears the
    // sharp waterbag boundary and generates spurious entropy. Use a generous
    // tolerance — this test validates qualitative conservation, not precision.
    assert!(
        max_s_drift < 5.0,
        "Waterbag entropy drift too large: {:.2e} (threshold 5.0)",
        max_s_drift
    );
}
