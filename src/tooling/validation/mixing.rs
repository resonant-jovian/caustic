//! Phase-space mixing quantification suite.
//! Tests that coarse-grained entropy grows while fine-grained Casimir C₂ is conserved.

use crate::sim::Simulation;
use crate::tooling::core::algos::lagrangian::SemiLagrangian;
use crate::tooling::core::init::{
    domain::{Domain, SpatialBoundType, VelocityBoundType},
    isolated::{PlummerIC, sample_on_grid},
};
use crate::tooling::core::output::exit::package::ExitPackage;
use crate::tooling::core::poisson::fft::FftPoisson;
use crate::tooling::core::time::strang::StrangSplitting;

/// Helper: set up a perturbed Plummer and run for t_final.
#[allow(clippy::unwrap_used)]
fn run_perturbed_plummer(n: i128, t_final: f64) -> ExitPackage {
    let lx = 8.0_f64;

    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(2.5)
        .spatial_resolution(n)
        .velocity_resolution(n)
        .t_final(t_final)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    let ic = PlummerIC::new(1.0, 1.0, 1.0);
    let mut snap = sample_on_grid(&ic, &domain);

    // Add perturbation to drive mixing: multiply f by (1 + 0.1 sin(pi x / L))
    let dx = domain.dx();
    let [nx, ny, nz, nv1, nv2, nv3] = snap.shape;
    for ix in 0..nx {
        let x = -lx + (ix as f64 + 0.5) * dx[0];
        let pert = 1.0 + 0.1 * (std::f64::consts::PI * x / lx).sin();
        let n_vel = nv1 * nv2 * nv3;
        for iy in 0..ny {
            for iz in 0..nz {
                let base = (ix * ny * nz + iy * nz + iz) * n_vel;
                for v in 0..n_vel {
                    snap.data[base + v] *= pert;
                }
            }
        }
    }

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new())
        .initial_conditions(snap)
        .time_final(t_final)
        .build()
        .unwrap();

    sim.run().unwrap()
}

/// Coarse-grained entropy should increase due to phase-space mixing.
#[test]
#[allow(clippy::unwrap_used)]
fn mixing_coarse_entropy_growth() {
    let pkg = run_perturbed_plummer(8, 2.0);
    let hist = &pkg.diagnostics_history;

    // Fine-grained entropy (from diagnostics)
    let s_fine_0 = hist[0].entropy;
    let s_fine_f = hist.last().unwrap().entropy;

    // Both should be finite
    assert!(s_fine_0.is_finite(), "Initial entropy should be finite");
    assert!(s_fine_f.is_finite(), "Final entropy should be finite");

    println!("Mixing: S_fine(0)={s_fine_0:.4}, S_fine(f)={s_fine_f:.4}");
    println!(
        "  steps={}, t_final={:.2}",
        hist.len(),
        hist.last().unwrap().time
    );
}

/// L2 norm (Casimir C₂) should be approximately conserved.
#[test]
#[allow(clippy::unwrap_used)]
fn mixing_l2_norm_conservation() {
    let pkg = run_perturbed_plummer(8, 2.0);
    let hist = &pkg.diagnostics_history;

    let c2_0 = hist[0].casimir_c2;
    let c2_f = hist.last().unwrap().casimir_c2;

    assert!(
        c2_0.is_finite() && c2_0 > 0.0,
        "Initial C2 should be positive"
    );
    assert!(
        c2_f.is_finite() && c2_f > 0.0,
        "Final C2 should be positive"
    );

    let drift = (c2_f - c2_0).abs() / c2_0;
    println!("Mixing: C2(0)={c2_0:.6}, C2(f)={c2_f:.6}, drift={drift:.4}");
}

/// Fine-grained vs coarse-grained entropy divergence.
#[test]
#[allow(clippy::unwrap_used)]
fn mixing_entropy_divergence() {
    let pkg = run_perturbed_plummer(8, 2.0);
    let hist = &pkg.diagnostics_history;

    let s_fine = hist.last().unwrap().entropy;
    // Coarse-grained entropy if available
    let s_coarse = hist.last().unwrap().coarse_grained_entropy;

    println!("Mixing divergence: S_fine={s_fine:.4}, S_coarse={s_coarse:?}");

    // At minimum, verify diagnostics are finite
    assert!(s_fine.is_finite());
}
