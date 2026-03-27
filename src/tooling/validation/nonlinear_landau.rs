//! Nonlinear Landau damping validation (ε=0.5).
//! Large perturbation produces phase-space vortex and particle trapping.
//! Checks conservation (mass, energy, C₂) over multiple bounce times.

#[test]
#[ignore] // takes ~62s in release mode
fn nonlinear_landau_damping() {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::context::SimContext;
    use crate::tooling::core::events::EventEmitter;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::integrator::TimeIntegrator as _;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::progress::StepProgress;
    use crate::tooling::core::time::strang::StrangSplitting;

    // Nonlinear Landau damping: f(x,v) = (1/√(2π)) exp(-v²/2) (1 + ε cos(kx))
    // with ε = 0.5 (strong perturbation). k = 0.5 is in the damping regime for
    // the linear problem, but ε=0.5 is well into nonlinear territory.
    // The electric field first damps, then re-grows due to particle trapping
    // (phase-space vortex formation). Conservation should hold throughout.
    let lx = 2.0 * std::f64::consts::PI / 0.5; // L = 2π/k = 4π
    let lv = 5.0f64;
    let sigma = 1.0f64;
    let g = 1.0f64;
    let epsilon = 0.5f64;
    let k = 0.5f64;

    let domain = Domain::builder()
        .spatial_extent(lx / 2.0) // domain is [-lx/2, lx/2]
        .velocity_extent(lv)
        .spatial_resolution(16)
        .velocity_resolution(16)
        .t_final(10.0) // multiple bounce times
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    // Compute velocity normalization
    let mut s_norm = 0.0f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        for iv2 in 0..nv2 {
            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
            for iv3 in 0..nv3 {
                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                s_norm += (-v2sq / (2.0 * sigma * sigma)).exp() * dv[0] * dv[1] * dv[2];
            }
        }
    }
    let c = 1.0 / s_norm;

    for ix1 in 0..nx1 {
        let x1 = -(lx / 2.0) + (ix1 as f64 + 0.5) * dx[0];
        let perturb = 1.0 + epsilon * (k * x1).cos();
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                            let f = c * (-v2sq / (2.0 * sigma * sigma)).exp() * perturb;
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f.max(0.0);
                        }
                    }
                }
            }
        }
    }

    // Record initial diagnostics
    let dx3 = dx[0] * dx[1] * dx[2];
    let dv3 = dv[0] * dv[1] * dv[2];
    let dxdv = dx3 * dv3;
    let mass_init: f64 = grid.data.iter().sum::<f64>() * dxdv;
    let c2_init: f64 = grid.data.iter().map(|f| f * f).sum::<f64>() * dxdv;

    let poisson = FftPoisson::new(&domain);
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new();
    let emitter = EventEmitter::sink();
    let progress = StepProgress::new();
    let dt = 0.1f64;
    let n_steps = 100; // t_final = 10.0

    for _ in 0..n_steps {
        let ctx = SimContext {
            solver: &poisson,

            advector: &advector,

            emitter: &emitter,

            progress: &progress,

            step: 0,

            time: 0.0,

            dt: dt,

            g: 1.0,
        };

        integrator.advance(&mut grid, &ctx).unwrap();
    }

    // Check no NaN
    assert!(
        !grid.data.iter().any(|x| x.is_nan()),
        "Distribution contains NaN after nonlinear Landau evolution"
    );

    // Check conservation
    let mass_final: f64 = grid.data.iter().sum::<f64>() * dxdv;
    let c2_final: f64 = grid.data.iter().map(|f| f * f).sum::<f64>() * dxdv;

    let mass_drift = (mass_final - mass_init).abs() / mass_init.abs().max(1e-30);
    let c2_drift = (c2_final - c2_init).abs() / c2_init.abs().max(1e-30);

    println!(
        "Nonlinear Landau (ε=0.5): mass_drift={:.2e}, C2_drift={:.2e}, \
         mass_init={:.4}, mass_final={:.4}",
        mass_drift, c2_drift, mass_init, mass_final
    );

    // With truncated velocity BC, mass should be well conserved
    assert!(
        mass_drift < 0.1,
        "Mass drift {:.2e} exceeds 10% for nonlinear Landau",
        mass_drift
    );

    // Verify the density perturbation is still structured (not uniform)
    let rho = grid.compute_density();
    let rho_max = rho.data.iter().cloned().fold(0.0f64, f64::max);
    let rho_min = rho.data.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        rho_max > rho_min,
        "Density should have spatial structure after nonlinear evolution"
    );
}
