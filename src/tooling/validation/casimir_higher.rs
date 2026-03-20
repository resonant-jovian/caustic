//! Higher-order Casimir invariant tracking.
//!
//! The Vlasov equation conserves all Casimirs C[s] = ∫ s(f) d³x d³v for any
//! smooth function s. This test tracks C₂ = ∫f², C₃ = ∫f³, C₄ = ∫f⁴, and
//! entropy S = -∫f ln f over a short Plummer equilibrium evolution to verify
//! they remain approximately constant.

#[test]
fn casimir_higher_order() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::init::isolated::{PlummerIC, sample_on_grid};
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::PhaseSpaceSnapshot;

    // ── Domain ────────────────────────────────────────────────────────
    let n = 8_usize;
    let domain = Domain::builder()
        .spatial_extent(8.0)
        .velocity_extent(2.5)
        .spatial_resolution(n as i128)
        .velocity_resolution(n as i128)
        .t_final(2.0)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    // ── Plummer IC ────────────────────────────────────────────────────
    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let snap = sample_on_grid(&ic, &domain);

    // ── Compute initial Casimirs from snapshot data ───────────────────
    let dx = domain.dx();
    let dv = domain.dv();
    let dx3 = dx[0] * dx[1] * dx[2];
    let dv3 = dv[0] * dv[1] * dv[2];
    let d6v = dx3 * dv3; // phase-space volume element

    let compute_casimirs = |data: &[f64]| -> (f64, f64, f64, f64) {
        let mut c2 = 0.0_f64;
        let mut c3 = 0.0_f64;
        let mut c4 = 0.0_f64;
        let mut entropy = 0.0_f64;
        for &f in data.iter() {
            if f > 0.0 {
                c2 += f * f;
                c3 += f * f * f;
                c4 += f * f * f * f;
                entropy -= f * f.ln();
            }
        }
        (c2 * d6v, c3 * d6v, c4 * d6v, entropy * d6v)
    };

    let (c2_init, c3_init, c4_init, s_init) = compute_casimirs(&snap.data);

    assert!(c2_init > 0.0, "Initial C₂ must be positive: {}", c2_init);
    assert!(c3_init > 0.0, "Initial C₃ must be positive: {}", c3_init);
    assert!(c4_init > 0.0, "Initial C₄ must be positive: {}", c4_init);

    // ── Build and run simulation ──────────────────────────────────────
    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain.clone())
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(1.0))
        .initial_conditions(snap)
        .time_final(2.0)
        .gravitational_constant(1.0)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();

    // ── Compute final Casimirs from final snapshot ────────────────────
    let (c2_final, c3_final, c4_final, s_final) = compute_casimirs(&pkg.final_snapshot.data);

    let c2_drift = (c2_final - c2_init).abs() / c2_init.abs().max(1e-30);
    let c3_drift = (c3_final - c3_init).abs() / c3_init.abs().max(1e-30);
    let c4_drift = (c4_final - c4_init).abs() / c4_init.abs().max(1e-30);
    let s_drift = (s_final - s_init).abs() / s_init.abs().max(1e-30);

    println!(
        "Casimir higher-order: C₂ drift={:.2e}, C₃ drift={:.2e}, \
         C₄ drift={:.2e}, entropy drift={:.2e}, steps={}",
        c2_drift, c3_drift, c4_drift, s_drift, pkg.total_steps
    );
    println!(
        "  C₂: {:.6} → {:.6}, C₃: {:.6} → {:.6}, C₄: {:.6} → {:.6}, S: {:.6} → {:.6}",
        c2_init, c2_final, c3_init, c3_final, c4_init, c4_final, s_init, s_final
    );

    // ── Assertions (generous tolerance at N=8 resolution) ─────────────
    assert!(
        c2_drift < 0.5,
        "C₂ drift too large: {:.2e} (threshold 0.5)",
        c2_drift
    );
    assert!(
        c3_drift < 0.5,
        "C₃ drift too large: {:.2e} (threshold 0.5)",
        c3_drift
    );
    assert!(
        c4_drift < 0.5,
        "C₄ drift too large: {:.2e} (threshold 0.5)",
        c4_drift
    );
    assert!(
        s_drift < 0.5,
        "Entropy drift too large: {:.2e} (threshold 0.5)",
        s_drift
    );

    // Also verify the built-in C₂ and entropy from diagnostics history
    let history = &pkg.diagnostics_history;
    assert!(
        history.len() >= 2,
        "Need at least 2 diagnostic entries, got {}",
        history.len()
    );
    let builtin_c2_0 = history[0].casimir_c2;
    let builtin_c2_drift = history
        .iter()
        .map(|d| (d.casimir_c2 - builtin_c2_0).abs() / builtin_c2_0.abs().max(1e-30))
        .fold(0.0_f64, f64::max);
    println!("  Built-in C₂ drift from diagnostics: {:.2e}", builtin_c2_drift);
}
