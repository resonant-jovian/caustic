//! Landau damping validation: small perturbation in the stable (k > k_J) regime undergoes
//! phase mixing and gravitational Landau damping. Checks no exponential growth occurs.

#[test]
fn landau_damping() {
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::integrator::TimeIntegrator as _;

    // Warm Maxwellian (σ=2): k_J = sqrt(4π*G*ρ0)/σ ≈ sqrt(4π)/2 ≈ 1.77 (with ρ0≈1, G=1)
    // lx=1 → k_fund = π ≈ 3.14 > k_J → stable; gravitational Landau damping regime.
    let lx = 1.0f64;
    let lv = 4.0f64;
    let sigma = 2.0f64;
    let g = 1.0f64;
    let epsilon = 0.05f64;
    let k = std::f64::consts::PI / lx;

    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(8)
        .velocity_resolution(8)
        .t_final(1.0)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let mut grid = UniformGrid6D::new(domain.clone());
    let dx = domain.dx();
    let dv = domain.dv();
    let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

    let mut s_norm = 0.0f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        for iv2 in 0..nv2 {
            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
            for iv3 in 0..nv3 {
                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                s_norm += (-(v1*v1+v2*v2+v3*v3) / (2.0*sigma*sigma)).exp()
                    * dv[0] * dv[1] * dv[2];
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
                            let f = c * (-(v1*v1+v2*v2+v3*v3)/(2.0*sigma*sigma)).exp()
                                * perturb;
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
    let mut integrator = StrangSplitting::new(g);
    let dt = 0.1f64;

    for _ in 0..8 {
        integrator.advance(&mut grid, &poisson, &advector, dt);
    }

    let rho_final = grid.compute_density();
    assert!(!rho_final.data.iter().any(|x| x.is_nan()), "Density contains NaN");

    let rho_max_final = rho_final.data.iter().cloned().fold(0.0f64, f64::max);
    let rho_min_final = rho_final.data.iter().cloned().fold(f64::MAX, f64::min);
    let amp_final = rho_max_final - rho_min_final;

    // k > k_J → stable: perturbation must not grow exponentially (allow 2x for oscillation)
    assert!(
        amp_final < 2.0 * amp_init,
        "Landau damping: amplitude grew (unstable). amp_init={:.4e}, amp_final={:.4e}",
        amp_init, amp_final
    );
    println!(
        "Landau damping: k={:.3}, k_J≈{:.3}, amp_init={:.4e}, amp_final={:.4e}",
        k, (4.0 * std::f64::consts::PI * g).sqrt() / sigma, amp_init, amp_final
    );
}
