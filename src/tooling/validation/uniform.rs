//! Uniform acceleration validation: constant external g, no self-gravity.
//! f shifts linearly in v at constant acceleration.

#[test]
fn uniform_acceleration() {
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::types::AccelerationField;
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;

    // 4×4×4 spatial, 16×16×16 velocity = 262144 cells
    // Spatial: [−2, 2]³, dx=1.0.  Velocity: [−4, 4]³, dv=0.5
    let domain = Domain::builder()
        .spatial_extent(2.0)
        .velocity_extent(4.0)
        .spatial_resolution(4)
        .velocity_resolution(16)
        .t_final(2.0)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Open)
        .build()
        .unwrap();

    let mut grid = UniformGrid6D::new(domain);
    let dv = grid.domain.dv();
    let dx = grid.domain.dx();
    let nv1 = grid.domain.velocity_res.v1 as usize;
    let nv2 = grid.domain.velocity_res.v2 as usize;
    let nv3 = grid.domain.velocity_res.v3 as usize;
    let nx1 = grid.domain.spatial_res.x1 as usize;
    let nx2 = grid.domain.spatial_res.x2 as usize;
    let nx3 = grid.domain.spatial_res.x3 as usize;
    let lv = 4.0f64;

    // Set Gaussian IC in v1 at spatial cell ix1=2, iv2=8 (center), iv3=8 (center)
    // Uniform across all spatial cells so all cells get same Gaussian in v1
    // iv2_0, iv3_0 = center velocity cells
    let iv2_0 = nv2 / 2;
    let iv3_0 = nv3 / 2;
    let sigma = 1.5f64;

    for ix1 in 0..nx1 {
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                for iv1 in 0..nv1 {
                    let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                    let f = (-v1 * v1 / (2.0 * sigma * sigma)).exp();
                    let idx = grid.index([ix1, ix2, ix3], [iv1, iv2_0, iv3_0]);
                    grid.data[idx] = f;
                }
            }
        }
    }

    // Apply constant acceleration a_x = 0.5 to all spatial cells
    // After dt=2.0, velocity shift = a*dt = 0.5*2.0 = 1.0 physical = 2 velocity cells (dv=0.5)
    let ax = 0.5f64;
    let dt = 2.0f64;
    let n_spatial = nx1 * nx2 * nx3;
    let accel = AccelerationField {
        gx: vec![ax; n_spatial],
        gy: vec![0.0; n_spatial],
        gz: vec![0.0; n_spatial],
        shape: [nx1, nx2, nx3],
    };

    grid.advect_v(&accel, dt);

    let shift = ax * dt; // = 1.0 physical velocity unit
    let domain_width = 2.0 * lv;

    // Check Gaussian shifted by shift in v1 at spatial cell (0, 0, 0), velocity (*, iv2_0, iv3_0)
    let mut max_err = 0.0f64;
    for iv1 in 0..nv1 {
        let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
        // Departure point in velocity space — absorbing BC so out-of-bounds → 0
        let v_dep = v1 - shift;
        let expected = if v_dep < -lv || v_dep >= lv {
            0.0
        } else {
            (-v_dep * v_dep / (2.0 * sigma * sigma)).exp()
        };
        let idx = grid.index([0, 0, 0], [iv1, iv2_0, iv3_0]);
        let actual = grid.data[idx];
        max_err = max_err.max((actual - expected).abs());
    }

    assert!(
        max_err < 1e-10,
        "Uniform acceleration L∞ error = {:.2e}, expected < 1e-10 (shift = {:.2})",
        max_err, shift
    );
}
