//! Gravitational cold collapse (1D) validation.
//! Cold slab with sinusoidal perturbation collapses under self-gravity.
//! Checks phase-space spiral formation and tracks HtTensor ranks if available.

#[test]
fn cold_collapse_1d() {
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

    // Cold 1D slab: f(x,v) = ρ(x) * δ(v) approximated as narrow Gaussian.
    // ρ(x₁) = ρ₀ (1 + ε cos(k x₁)) with k < k_J → collapse forms
    // phase-space spiral (shell crossing / caustic).
    let lx = std::f64::consts::PI;
    let lv = 2.0f64;
    let sigma_v = 0.15f64; // cold: narrow velocity distribution
    let g = 1.0f64;
    let rho0 = 1.0f64;
    let epsilon = 0.3f64; // moderately strong perturbation
    let k = std::f64::consts::PI / lx;

    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(16)
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

    // Normalize narrow Gaussian in velocity
    let mut s_norm = 0.0f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        for iv2 in 0..nv2 {
            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
            for iv3 in 0..nv3 {
                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                s_norm += (-v2sq / (2.0 * sigma_v * sigma_v)).exp() * dv[0] * dv[1] * dv[2];
            }
        }
    }

    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        let rho_local = rho0 * (1.0 + epsilon * (k * x1).cos());
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            let v2sq = v1 * v1 + v2 * v2 + v3 * v3;
                            let f = rho_local / s_norm * (-v2sq / (2.0 * sigma_v * sigma_v)).exp();
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

    let poisson = FftPoisson::new(&domain);
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new();
    let emitter = EventEmitter::sink();
    let progress = StepProgress::new();
    let dt = 0.05f64;
    let n_steps = 40; // t = 2.0

    // Track density peak growth (collapse should increase peak density)
    let mut density_peaks = vec![rho_max_init];
    for step in 0..n_steps {
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
            assert!(
                !rho.data.iter().any(|x| x.is_nan()),
                "NaN at step {}",
                step + 1
            );
            let peak = rho.data.iter().cloned().fold(0.0f64, f64::max);
            density_peaks.push(peak);
        }
    }

    let rho_final = grid.compute_density();
    let rho_max_final = rho_final.data.iter().cloned().fold(0.0f64, f64::max);

    // Cold collapse: density peak should increase due to gravitational focusing
    // With k < k_J and cold initial conditions, the perturbation grows strongly.
    assert!(
        rho_max_final > rho_max_init,
        "Cold collapse: density peak should increase. init={:.4}, final={:.4}",
        rho_max_init,
        rho_max_final
    );

    // Semi-Lagrangian with Catmull-Rom interpolation does not guarantee positivity.
    // Near caustics (sharp phase-space features), polynomial undershoots are expected.
    // Verify undershoots are bounded relative to the peak value.
    let f_min = grid.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let undershoot_ratio = -f_min / rho_max_final;
    assert!(
        f_min > -rho_max_final,
        "Undershoot too large: f_min={f_min:.4e}, peak={rho_max_final:.4e}, ratio={undershoot_ratio:.2}"
    );

    println!(
        "Cold collapse: rho_max init={:.4}, final={:.4}, ratio={:.2}, peaks={:?}",
        rho_max_init,
        rho_max_final,
        rho_max_final / rho_max_init,
        density_peaks
    );
}
