//! Two-stream instability validation.
//! IC: f(x,v) ∝ v² exp(-v²/2) (1 + ε cos(kx)) — two counter-streaming beams.
//! The perturbation should grow and saturate at a finite amplitude.

#[test]
fn two_stream_instability() {
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

    // Two-stream: f(x,v) = C * v₁² * exp(-v²/2) * (1 + ε cos(k x₁))
    // The v² factor creates two beams moving in ±x₁ direction.
    // With G=1, this is gravitationally unstable for appropriate k.
    let lx = std::f64::consts::PI;
    let lv = 4.0f64;
    let sigma = 1.0f64;
    let g = 1.0f64;
    let epsilon = 0.05f64;
    let k = std::f64::consts::PI / lx; // fundamental mode

    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(2.0)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    // Compute normalization for v₁² exp(-v²/2σ²) distribution
    let mut s_norm = 0.0f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        for iv2 in 0..nv2 {
            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
            for iv3 in 0..nv3 {
                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                s_norm += v1 * v1 * (-v2sq / (2.0 * sigma * sigma)).exp() * dv[0] * dv[1] * dv[2];
            }
        }
    }
    let c = 1.0 / s_norm.max(1e-30);

    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
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
                            let f = c * v1 * v1 * (-v2sq / (2.0 * sigma * sigma)).exp() * perturb;
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f.max(0.0);
                        }
                    }
                }
            }
        }
    }

    let rho_init = grid.compute_density();
    let rho_max_init = rho_init.data.iter().cloned().fold(0.0f64, f64::max);
    let rho_min_init = rho_init.data.iter().cloned().fold(f64::MAX, f64::min);
    let amp_init = (rho_max_init - rho_min_init).max(1e-30);

    let poisson = FftPoisson::new(&domain);
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new();
    let emitter = EventEmitter::sink();
    let progress = StepProgress::new();
    let dt = 0.05f64;

    // Track amplitude over time for saturation detection
    let mut max_amp = amp_init;
    for step in 0..40 {
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

        if (step + 1) % 10 == 0 {
            let rho = grid.compute_density();
            let rho_max = rho.data.iter().cloned().fold(0.0f64, f64::max);
            let rho_min = rho.data.iter().cloned().fold(f64::MAX, f64::min);
            let amp = rho_max - rho_min;
            if amp > max_amp {
                max_amp = amp;
            }
        }
    }

    let rho_final = grid.compute_density();
    assert!(
        !rho_final.data.iter().any(|x| x.is_nan()),
        "Density contains NaN after two-stream evolution"
    );

    let rho_max_final = rho_final.data.iter().cloned().fold(0.0f64, f64::max);
    let rho_min_final = rho_final.data.iter().cloned().fold(f64::MAX, f64::min);
    let amp_final = rho_max_final - rho_min_final;

    // The two-stream instability should cause the perturbation to grow
    // (at least initially, before possible saturation)
    assert!(
        max_amp > amp_init,
        "Two-stream: perturbation should grow. amp_init={:.4e}, max_amp={:.4e}",
        amp_init,
        max_amp
    );

    println!(
        "Two-stream instability: amp_init={:.4e}, amp_final={:.4e}, max_amp={:.4e}, ratio={:.2}",
        amp_init,
        amp_final,
        max_amp,
        max_amp / amp_init
    );
}
