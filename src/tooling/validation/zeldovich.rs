//! Zel'dovich pancake validation: 1D plane-wave perturbation collapses to a caustic.
//! IC: f(x,v) ≈ ρ̄ * δ(v − v₀(x)) with v₀(x) = −H*A*sin(k*x) (convergent flow).
//! After t = 1/(k*H*A), shell crossing occurs; max density >> initial.

#[test]
fn zeldovich_pancake() {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::integrator::TimeIntegrator as _;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    // Setup: 1D plane wave in x1. v₀(x) = -A*sin(k*x1) (converging flow toward x=0).
    // t_col = 1/A (with k=1, H=1): at t=1 the caustic forms.
    // We run to t=0.8*t_col and check max density has grown significantly.
    let lx = std::f64::consts::PI; // domain [-π,π]³, width=2π, k_fund=1
    let lv = 2.0f64;
    let rho_bar = 1.0f64;
    let sigma_v = 0.25f64; // narrow Gaussian approximating δ(v)
    let g = 0.1f64; // weak gravity — mainly testing pure streaming collapse
    let a_amp = 1.0f64; // perturbation amplitude
    let k = 1.0f64; // k_fundamental = 2π/(2*lx) = 1.0 for lx=π
    // t_col = 1/(k*a_amp) = 1

    let domain = Domain::builder()
        .spatial_extent(lx)
        .velocity_extent(lv)
        .spatial_resolution(16)
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

    // Velocity normalization for narrow Gaussian
    let mut s_norm = 0.0f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        for iv2 in 0..nv2 {
            let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
            for iv3 in 0..nv3 {
                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                s_norm += (-(v1 * v1 + v2 * v2 + v3 * v3) / (2.0 * sigma_v * sigma_v)).exp()
                    * dv[0]
                    * dv[1]
                    * dv[2];
            }
        }
    }

    // f(x,v) = ρ̄/S * exp(−(v−v₀(x))²/(2σ_v²)) where v₀(x) = −A*sin(k*x1)
    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        let v0 = -a_amp * (k * x1).sin(); // inflow velocity field
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                            // Only the x1 component is perturbed; v2,v3 centred at 0
                            let dv1 = v1 - v0;
                            let v2sq = dv1 * dv1 + v2 * v2 + v3 * v3;
                            let f = rho_bar / s_norm * (-v2sq / (2.0 * sigma_v * sigma_v)).exp();
                            let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                            grid.data[idx] = f;
                        }
                    }
                }
            }
        }
    }

    let rho_init = grid.compute_density();
    let rho_max_init = rho_init.data.iter().cloned().fold(0.0f64, f64::max);
    assert!(rho_max_init > 0.0, "Initial density must be positive");

    let poisson = FftPoisson::new(&domain);
    let advector = SemiLagrangian::new();
    let mut integrator = StrangSplitting::new(g);

    // Run toward the caustic: t_col=1, run to t=0.8
    let dt = 0.05f64;
    let n_steps = 16; // 16 * 0.05 = 0.8 * t_col

    for _ in 0..n_steps {
        integrator.advance(&mut grid, &poisson, &advector, dt).unwrap();
    }

    let rho_final = grid.compute_density();
    assert!(
        !rho_final.data.iter().any(|x| x.is_nan()),
        "Density contains NaN"
    );

    let rho_max_final = rho_final.data.iter().cloned().fold(0.0f64, f64::max);

    // Pancake collapse: streams converge at x=0 → max density grows significantly
    assert!(
        rho_max_final > 1.5 * rho_max_init,
        "Zel'dovich pancake: density should grow toward caustic. init={:.4}, final={:.4}",
        rho_max_init,
        rho_max_final
    );
    println!(
        "Zel'dovich pancake: t_run={:.2}, rho_max_init={:.4}, rho_max_final={:.4}, ratio={:.2}",
        n_steps as f64 * dt,
        rho_max_init,
        rho_max_final,
        rho_max_final / rho_max_init
    );
}
