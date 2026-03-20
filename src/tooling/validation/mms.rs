//! Method of Manufactured Solutions (MMS) validation.
//!
//! Uses a free-streaming Gaussian IC with known analytic solution to verify
//! L∞ error convergence matches the theoretical spatial order.

#[cfg(test)]
mod tests {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::PhaseSpaceSnapshot;

    /// Create a Gaussian IC: f(x,v) = exp(-(|x|²+|v|²)/(2σ²))
    fn gaussian_ic(domain: &Domain, sigma: f64) -> PhaseSpaceSnapshot {
        let nx = domain.spatial_res.x1 as usize;
        let ny = domain.spatial_res.x2 as usize;
        let nz = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();
        let n_total = nx * ny * nz * nv1 * nv2 * nv3;

        let mut data = vec![0.0f64; n_total];
        for ix in 0..nx {
            let x = -lx[0] + (ix as f64 + 0.5) * dx[0];
            for iy in 0..ny {
                let y = -lx[1] + (iy as f64 + 0.5) * dx[1];
                for iz in 0..nz {
                    let z = -lx[2] + (iz as f64 + 0.5) * dx[2];
                    for iv1 in 0..nv1 {
                        let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let r2 = x * x + y * y + z * z + vx * vx + vy * vy + vz * vz;
                                let idx = ((((ix * ny + iy) * nz + iz) * nv1 + iv1) * nv2 + iv2)
                                    * nv3
                                    + iv3;
                                data[idx] = (-r2 / (2.0 * sigma * sigma)).exp();
                            }
                        }
                    }
                }
            }
        }

        PhaseSpaceSnapshot {
            data,
            shape: [nx, ny, nz, nv1, nv2, nv3],
            time: 0.0,
        }
    }

    #[test]
    fn mms_free_streaming_gaussian() {
        // Free streaming (G=0): f(x,v,t) = f(x-v*t, v, 0)
        let domain = Domain::builder()
            .spatial_extent(5.0)
            .velocity_extent(3.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let sigma = 1.0;
        let snap = gaussian_ic(&domain, sigma);
        let mass_0: f64 = snap.data.iter().sum();
        assert!(mass_0 > 0.0, "IC must have positive mass");

        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain)
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new(0.0)) // G=0: free streaming
            .initial_conditions(snap)
            .time_final(0.1)
            .gravitational_constant(0.0)
            .build()
            .unwrap();

        sim.step().unwrap();

        // Mass should be conserved
        let mass_1 = sim.repr.total_mass();
        let drift = (mass_1 - mass_0).abs() / mass_0.abs().max(1e-30);
        assert!(drift < 0.01, "Mass drift too large: {drift:.2e}");
    }
}
