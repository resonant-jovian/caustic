//! Free-streaming validation test: G=0, Gaussian IC shifts exactly as f(x−vt, v, 0).
//! Validates spatial advection accuracy of the semi-Lagrangian scheme.

#[test]
fn free_streaming() {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::context::SimContext;
    use crate::tooling::core::events::EventEmitter;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::progress::StepProgress;
    use crate::tooling::core::types::DisplacementField;

    // 16×16×16 spatial, 4×4×4 velocity = 262144 cells total
    // Spatial: [−4, 4]³, dx = 0.5.  Velocity: [−2, 2]³, dv = 1.0
    let domain = Domain::builder()
        .spatial_extent(4.0)
        .velocity_extent(2.0)
        .spatial_resolution(16)
        .velocity_resolution(4)
        .t_final(2.0)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let mut grid = UniformGrid6D::new(domain);
    let dx = grid.domain.dx();
    let dv = grid.domain.dv();
    let nx1 = grid.domain.spatial_res.x1 as usize;
    let nx2 = grid.domain.spatial_res.x2 as usize;
    let nx3 = grid.domain.spatial_res.x3 as usize;
    let lx = 4.0f64;
    let lv = 2.0f64;

    // Velocity cell iv1=2, iv2=0, iv3=0 has:
    //   vx = −lv + (2 + 0.5)*dv[0] = −2 + 2.5 = 0.5
    //   vy = −lv + (0 + 0.5)*dv[1] = −2 + 0.5 = −1.5  (uniform in x2 → shift irrelevant)
    //   vz = −lv + (0 + 0.5)*dv[2] = −2 + 0.5 = −1.5  (uniform in x3 → shift irrelevant)
    let iv0 = 2usize;
    let vx = -lv + (iv0 as f64 + 0.5) * dv[0]; // = 0.5

    // Set Gaussian IC: f(x1, x2, x3, iv0, 0, 0) = exp(−x1²/2σ²), σ=1.5
    // Uniform in x2 and x3 (same value for all ix2, ix3)
    let sigma = 1.5f64;
    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        let f = (-x1 * x1 / (2.0 * sigma * sigma)).exp();
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                let idx = grid.index([ix1, ix2, ix3], [iv0, 0, 0]);
                grid.data[idx] = f;
            }
        }
    }

    // dt = 2.0 → shift in x1 = vx * dt = 0.5 * 2.0 = 1.0 physical unit = 2 grid cells
    let dt = 2.0f64;
    let shift = vx * dt; // = 1.0 physical unit

    let dummy = DisplacementField {
        dx: vec![],
        dy: vec![],
        dz: vec![],
        shape: [0, 0, 0],
    };
    let poisson = FftPoisson::new(&grid.domain);
    let advector = SemiLagrangian::new();
    let emitter = EventEmitter::sink();
    let progress = StepProgress::new();
    let ctx = SimContext {
        solver: &poisson,
        advector: &advector,
        emitter: &emitter,
        progress: &progress,
        step: 0,
        time: 0.0,
        dt,
        g: 0.0,
    };
    grid.advect_x(&dummy, &ctx);

    // Check: the Gaussian peak should now be centred at x1 = shift (periodic wrapping)
    // Compare at ix2=0, ix3=0 for all ix1
    let mut max_err = 0.0f64;
    let domain_width = 2.0 * lx;
    for ix1 in 0..nx1 {
        let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
        // Departure point with periodic wrap: the semi-Lagrangian advector uses periodic BC
        let x_dep = x1 - shift;
        let x_dep_wrapped = ((x_dep + lx).rem_euclid(domain_width)) - lx;
        let expected = (-x_dep_wrapped * x_dep_wrapped / (2.0 * sigma * sigma)).exp();
        let idx = grid.index([ix1, 0, 0], [iv0, 0, 0]);
        let actual = grid.data[idx];
        max_err = max_err.max((actual - expected).abs());
    }

    assert!(
        max_err < 0.05,
        "Free streaming L∞ error = {:.4}, expected < 0.05 (shift = {:.2} physical units)",
        max_err,
        shift
    );
}
