//! Jeans instability validation: sinusoidal density perturbation with k < k_J grows.
//! Uses warm Maxwellian IC + cosine perturbation; checks density amplitude increases.

#[test]
fn jeans_instability() {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::integrator::TimeIntegrator as _;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    // Parameters: G=1, sigma=1, spatial_extent=pi → k_fund=1 < k_J≈3.54 → unstable
    // Growth rate γ = sqrt(4πGρ0 − k²σ²) ≈ sqrt(12.57 − 1.0) ≈ 3.4
    // After 5 steps of dt=0.05 (t=0.25): amplitude grows by exp(3.4*0.25) ≈ 2.3x
    let lx = std::f64::consts::PI; // domain half-width → k_fundamental = pi/lx = 1
    let lv = 3.0f64;
    let sigma = 1.0f64;
    let g = 1.0f64;
    let epsilon = 0.1f64;
    let k = std::f64::consts::PI / lx; // = 1.0

    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(0.5)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    // Fill f(x,v) = C * exp(-v²/2σ²) * (1 + ε*cos(k*x1))
    // C chosen so grid density ≈ 1 (computed from Maxwellian sum over v grid)
    // Compute normalization: S = sum_{v} exp(-v²/2σ²) * dv³
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
    let c = 1.0 / s_norm; // normalise so ρ0 ≈ 1

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
                            let f = c * (-v2sq / (2.0 * sigma * sigma)).exp() * perturb;
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f.max(0.0);
                        }
                    }
                }
            }
        }
    }

    let rho0_init = grid.compute_density();
    let rho_max_init = rho0_init.data.iter().cloned().fold(0.0f64, f64::max);
    let rho_min_init = rho0_init.data.iter().cloned().fold(f64::MAX, f64::min);
    let amp_init = rho_max_init - rho_min_init;

    let poisson = FftPoisson::new(&domain);
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new(g);
    let dt = 0.05f64;

    for _ in 0..5 {
        integrator.advance(&mut grid, &poisson, &advector, dt).unwrap();
    }

    let rho_final = grid.compute_density();
    assert!(
        !rho_final.data.iter().any(|x| x.is_nan()),
        "Density contains NaN"
    );

    let rho_max_final = rho_final.data.iter().cloned().fold(0.0f64, f64::max);
    let rho_min_final = rho_final.data.iter().cloned().fold(f64::MAX, f64::min);
    let amp_final = rho_max_final - rho_min_final;

    // k_J ≈ sqrt(4πGρ0)/σ ≈ 3.54 > k=1 → unstable → amplitude should grow
    assert!(
        amp_final > amp_init,
        "Jeans instability: amplitude should grow. init={:.4}, final={:.4}",
        amp_init,
        amp_final
    );
    println!(
        "Jeans instability: k={:.3}, k_J≈{:.3}, amp_init={:.4}, amp_final={:.4}, ratio={:.2}",
        k,
        (4.0 * std::f64::consts::PI * g).sqrt() / sigma,
        amp_init,
        amp_final,
        amp_final / amp_init
    );
}
