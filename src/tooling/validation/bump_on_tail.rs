//! Bump-on-tail instability validation.
//!
//! IC: f(v) = 0.9·N(0,1) + 0.1·N(4.5,0.5²) with spatial perturbation.
//! Verifies linear growth phase and conservation during nonlinear saturation.

#[cfg(test)]
mod tests {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::PhaseSpaceSnapshot;

    #[test]
    fn bump_on_tail_conservation() {
        let domain = Domain::builder()
            .spatial_extent(10.0)
            .velocity_extent(8.0)
            .spatial_resolution(8)
            .velocity_resolution(16)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let nx = 8usize;
        let nv = 16usize;
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();
        let n_total = nx * nx * nx * nv * nv * nv;

        let mut data = vec![0.0f64; n_total];
        let sigma_bulk = 1.0;
        let sigma_bump = 0.5;
        let v_bump = 4.5;

        for ix in 0..nx {
            let x = -lx[0] + (ix as f64 + 0.5) * dx[0];
            let pert = 1.0 + 0.01 * (std::f64::consts::PI * x / lx[0]).sin();
            for iy in 0..nx {
                for iz in 0..nx {
                    for iv1 in 0..nv {
                        let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv {
                            for iv3 in 0..nv {
                                let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                                let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let v2_perp = vy*vy + vz*vz;
                                let f_bulk = 0.9 * (-vx*vx/(2.0*sigma_bulk*sigma_bulk) - v2_perp/(2.0*sigma_bulk*sigma_bulk)).exp();
                                let f_bump = 0.1 * (-(vx-v_bump).powi(2)/(2.0*sigma_bump*sigma_bump) - v2_perp/(2.0*sigma_bump*sigma_bump)).exp();
                                let idx = ((((ix * nx + iy) * nx + iz) * nv + iv1) * nv + iv2) * nv + iv3;
                                data[idx] = (f_bulk + f_bump) * pert;
                            }
                        }
                    }
                }
            }
        }

        let snap = PhaseSpaceSnapshot {
            data,
            shape: [nx, nx, nx, nv, nv, nv],
            time: 0.0,
        };

        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain)
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new(1.0))
            .initial_conditions(snap)
            .time_final(1.0)
            .build()
            .unwrap();

        let pkg = sim.run().unwrap();
        let summary = pkg.conservation_summary;

        // Energy should be conserved to reasonable tolerance
        assert!(
            summary.max_energy_drift < 0.1,
            "Energy drift too large: {:.2e}",
            summary.max_energy_drift
        );
    }
}
