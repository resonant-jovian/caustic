//! Jeans stability validation: perturbation with k > k_J oscillates, does not grow.

#[test]
fn jeans_stability() {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::integrator::TimeIntegrator as _;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    // Large sigma=3.0, small box lx=0.5:
    // k_J = sqrt(4πGρ0)/σ ≈ sqrt(4π)/3 ≈ 1.18 (ρ0≈1, G=1)
    // k_fund = π/lx = π/0.5 ≈ 6.28 >> k_J → Jeans-stable
    let lx = 0.5f64;
    let lv = 5.0f64;
    let sigma = 3.0f64;
    let g = 1.0f64;
    let epsilon = 0.1f64;
    let k = std::f64::consts::PI / lx; // ≈ 6.28

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

    // Maxwellian normalization on the grid
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

    let rho_init = grid.compute_density();
    let rho_max_init = rho_init.data.iter().cloned().fold(0.0f64, f64::max);

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
    assert!(
        rho_final.data.iter().all(|x| x.is_finite()),
        "Density contains Inf"
    );

    let rho_max_final = rho_final.data.iter().cloned().fold(0.0f64, f64::max);

    // Stable mode: max density must not exponentially blow up (allow 3x for oscillation)
    assert!(
        rho_max_final < 3.0 * rho_max_init,
        "Jeans stability: density grew too much. init={:.4}, final={:.4}",
        rho_max_init,
        rho_max_final
    );
    println!(
        "Jeans stability: k={:.3}, k_J≈{:.3}, rho_max_init={:.4}, rho_max_final={:.4}",
        k,
        (4.0 * std::f64::consts::PI * g).sqrt() / sigma,
        rho_max_init,
        rho_max_final
    );
}
