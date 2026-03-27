//! Diagnostic test for HT (Hierarchical Tucker) representation.
//!
//! Exercises the HT initialization + one Strang step to verify
//! density is preserved through SLAR advection. Produces tracing
//! output at debug level for diagnosing density collapse bugs.

#[test]
fn ht_plummer_one_step() {
    use crate::tooling::core::algos::ht::HtTensor;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::{
        domain::{Domain, SpatialBoundType, VelocityBoundType},
        isolated::{IsolatedEquilibrium, PlummerIC, sample_on_grid},
    };
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::vgf::VgfPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    // Small grid that matches plummer_ht config style
    // (isolated BC, HT repr, VGF Poisson)
    let domain = Domain::builder()
        .spatial_extent(10.0)
        .velocity_extent(3.0)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(10.0)
        .spatial_bc(SpatialBoundType::Isolated)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    let ic = PlummerIC::new(1.0, 1.0, 1.0);

    // Build HT tensor via ACA (same path as phasma configs)
    let ht = HtTensor::from_function_aca(
        |x, v| {
            let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
            let phi = ic.potential(r);
            let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            let energy = 0.5 * v2 + phi;
            ic.distribution_function(energy, 0.0).max(0.0)
        },
        &domain,
        1e-6,
        50,
        None,
        None,
    );

    // Inspect individual nodes for NaN/zero content
    for (i, node) in ht.nodes.iter().enumerate() {
        match node {
            crate::tooling::core::algos::ht::HtNode::Leaf { dim, frame } => {
                let nrows = frame.nrows();
                let ncols = frame.ncols();
                let mut nan_count = 0usize;
                let mut max_val = 0.0f64;
                for r in 0..nrows {
                    for c in 0..ncols {
                        let v = frame[(r, c)];
                        if v.is_nan() {
                            nan_count += 1;
                        }
                        if v.is_finite() {
                            max_val = max_val.max(v.abs());
                        }
                    }
                }
                println!(
                    "  Node {i}: Leaf dim={dim}, shape=({nrows}x{ncols}), nan={nan_count}, max_abs={max_val:.6e}"
                );
            }
            crate::tooling::core::algos::ht::HtNode::Interior {
                left,
                right,
                transfer,
                ranks,
            } => {
                let nan_count = transfer.iter().filter(|v| v.is_nan()).count();
                let max_val = transfer
                    .iter()
                    .filter(|v| v.is_finite())
                    .map(|v| v.abs())
                    .fold(0.0f64, f64::max);
                println!(
                    "  Node {i}: Interior [{left},{right}], ranks={ranks:?}, len={}, nan={nan_count}, max_abs={max_val:.6e}",
                    transfer.len()
                );
            }
        }
    }

    // Also test: evaluate a single point at center
    let center_val = ht.evaluate([4, 4, 4, 4, 4, 4]);
    println!("HT evaluate([4,4,4,4,4,4]) = {center_val:.6e}");

    // Quick from_full comparison: build HT from the uniform grid snapshot
    let snap_for_ht = sample_on_grid(&ic, &domain);
    let ht_from_full = HtTensor::from_full(&snap_for_ht.data, snap_for_ht.shape, &domain, 1e-6);
    let from_full_density = ht_from_full.compute_density();
    let from_full_rho_max = from_full_density
        .data
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    println!(
        "HT from_full: rho_max={from_full_rho_max:.6e}, rank={}",
        ht_from_full.total_rank()
    );

    // Check initial density
    let init_density = ht.compute_density();
    let init_rho_max = init_density.data.iter().cloned().fold(0.0_f64, f64::max);
    let init_mass: f64 =
        init_density.data.iter().sum::<f64>() * domain.dx().iter().product::<f64>();
    println!(
        "HT init: rho_max={init_rho_max:.6e}, mass={init_mass:.6e}, rank={}",
        ht.total_rank()
    );

    assert!(
        init_rho_max > 1e-6,
        "Initial rho_max should be non-trivial, got {init_rho_max}"
    );
    assert!(
        init_mass > 0.01,
        "Initial mass should be non-trivial, got {init_mass}"
    );

    // Also build uniform grid from same IC for comparison
    let snap = sample_on_grid(&ic, &domain);
    let uniform =
        crate::tooling::core::algos::uniform::UniformGrid6D::from_snapshot(snap, domain.clone());
    let uniform_density = uniform.compute_density();
    let uniform_rho_max = uniform_density.data.iter().cloned().fold(0.0_f64, f64::max);
    let uniform_mass: f64 =
        uniform_density.data.iter().sum::<f64>() * domain.dx().iter().product::<f64>();
    println!("Uniform: rho_max={uniform_rho_max:.6e}, mass={uniform_mass:.6e}");

    // Compare HT vs uniform density
    let rho_max_ratio = init_rho_max / uniform_rho_max.max(1e-30);
    let mass_ratio = init_mass / uniform_mass.max(1e-30);
    println!("HT/Uniform: rho_max_ratio={rho_max_ratio:.4}, mass_ratio={mass_ratio:.4}");

    // Build simulation with HT
    let poisson = VgfPoisson::new(&domain);
    let mut sim = crate::sim::Simulation::builder()
        .domain(domain)
        .representation_boxed(Box::new(ht))
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new())
        .time_final(10.0)
        .gravitational_constant(1.0)
        .cfl_factor(0.3)
        .build()
        .unwrap();

    println!(
        "Initial diag: E={:.6e}, T={:.6e}, W={:.6e}, virial={:.4}, mass={:.6e}",
        sim.diagnostics.history[0].total_energy,
        sim.diagnostics.history[0].kinetic_energy,
        sim.diagnostics.history[0].potential_energy,
        sim.diagnostics.history[0].virial_ratio,
        sim.diagnostics.history[0].mass_in_box,
    );

    // Run exactly one step
    let exit = sim.step().unwrap();
    println!(
        "Step 1: exit={exit:?}, time={:.6e}, step={}",
        sim.time, sim.step
    );

    // Check post-step density
    let post_density = sim.repr.compute_density();
    let post_rho_max = post_density.data.iter().cloned().fold(0.0_f64, f64::max);
    let post_mass: f64 =
        post_density.data.iter().sum::<f64>() * sim.domain.dx().iter().product::<f64>();
    println!("Post step 1: rho_max={post_rho_max:.6e}, mass={post_mass:.6e}");

    let step_mass_ratio = post_mass / init_mass.max(1e-30);
    println!("Mass ratio after step 1: {step_mass_ratio:.6}");

    assert!(
        post_rho_max > 1e-10,
        "Post-step rho_max should not collapse to near-zero, got {post_rho_max:.6e}"
    );
    assert!(
        step_mass_ratio > 0.01,
        "Mass should not collapse after one step, got ratio {step_mass_ratio:.6e}"
    );
}
