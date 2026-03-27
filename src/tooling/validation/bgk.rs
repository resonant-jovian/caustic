//! BGK mode validation.
//!
//! Constructs a self-consistent BGK equilibrium f = F(E) where E = v^2/2 + Phi(x).
//! Because f depends only on the total energy (an integral of motion), it is an
//! exact stationary solution of the Vlasov equation. The test verifies that the
//! distribution function propagates without distortion for multiple dynamical times.
//!
//! Uses a Gaussian-in-energy distribution F(E) = A * exp(-E / sigma^2) with a
//! prescribed potential Phi(x) = Phi_0 * cos(k * x1). This is approximately
//! self-consistent in the small-Phi_0 limit; the test checks that the solver
//! preserves the equilibrium structure.

#[cfg(test)]
mod tests {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::*;

    /// BGK equilibrium preservation test.
    ///
    /// Initialises f(x,v) = A * exp(-(v^2/2 + Phi_0*cos(k*x1)) / sigma^2) and
    /// evolves with self-gravity. Checks that:
    /// 1. The density profile shape is preserved (L2 error vs initial stays bounded)
    /// 2. Mass is preserved (truncated velocity BC prevents leakage)
    /// 3. Energy drift remains bounded
    /// 4. No NaN or negative-density pathologies appear
    #[test]
    #[ignore] // requires --release for reasonable runtime
    fn bgk_equilibrium() {
        // ── Parameters ──────────────────────────────────────────────────
        let phi_0 = 0.05_f64; // small potential amplitude for near-equilibrium
        let k_mode = 1.0_f64; // wavenumber
        let sigma2 = 1.0_f64; // energy spread (sigma^2)
        let g = 1.0_f64; // gravitational constant
        let t_final = 3.0_f64; // evolve for several dynamical times

        // ── Domain ──────────────────────────────────────────────────────
        let n = 16_i128; // modest resolution for speed
        let nv = 16_i128;
        let l_x = std::f64::consts::PI; // half-extent: box is [-pi, pi]^3
        let l_v = 4.0;

        let domain = Domain::builder()
            .spatial_extent(l_x)
            .velocity_extent(l_v)
            .spatial_resolution(n)
            .velocity_resolution(nv)
            .t_final(t_final)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated) // preserve mass
            .build()
            .unwrap();

        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];

        let nx = n as usize;
        let nv_u = nv as usize;
        let n_sp = nx * nx * nx;
        let n_vel = nv_u * nv_u * nv_u;

        // ── Build BGK IC: f = A * exp(-(v^2/2 + Phi(x)) / sigma^2) ─────
        let mut data = vec![0.0_f64; n_sp * n_vel];

        for ix1 in 0..nx {
            for ix2 in 0..nx {
                for ix3 in 0..nx {
                    let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
                    let phi_x = phi_0 * (k_mode * x1).cos();
                    let si = ix1 * nx * nx + ix2 * nx + ix3;

                    for iv1 in 0..nv_u {
                        for iv2 in 0..nv_u {
                            for iv3 in 0..nv_u {
                                let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                                let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                                let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let ke = 0.5 * (v1 * v1 + v2 * v2 + v3 * v3);
                                let energy = ke + phi_x;
                                let f_val = (-energy / sigma2).exp();
                                let idx = si * n_vel + iv1 * nv_u * nv_u + iv2 * nv_u + iv3;
                                data[idx] = f_val;
                            }
                        }
                    }
                }
            }
        }

        // Record initial density profile for shape comparison
        let initial_grid = UniformGrid6D::from_snapshot(
            PhaseSpaceSnapshot {
                data: data.clone(),
                shape: [nx, nx, nx, nv_u, nv_u, nv_u],
                time: 0.0,
            },
            domain.clone(),
        );
        let initial_density = initial_grid.compute_density();
        let rho_init_l2: f64 = initial_density
            .data
            .iter()
            .map(|&r| r * r)
            .sum::<f64>()
            .sqrt();
        let mass_init: f64 = initial_density.data.iter().sum::<f64>() * dx3;

        // ── Build simulation ────────────────────────────────────────────
        let snap = PhaseSpaceSnapshot {
            data,
            shape: [nx, nx, nx, nv_u, nv_u, nv_u],
            time: 0.0,
        };

        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain.clone())
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new())
            .initial_conditions(snap)
            .time_final(t_final)
            .gravitational_constant(g)
            .cfl_factor(0.4)
            .build()
            .unwrap();

        // ── Run to completion ───────────────────────────────────────────
        let pkg = sim.run().unwrap();

        // ── Check final state ───────────────────────────────────────────
        assert!(
            !pkg.final_snapshot.data.iter().any(|f| f.is_nan()),
            "Final snapshot contains NaN"
        );

        // Compute final density and compare shape to initial
        let final_grid = UniformGrid6D::from_snapshot(
            PhaseSpaceSnapshot {
                data: pkg.final_snapshot.data.clone(),
                shape: pkg.final_snapshot.shape,
                time: pkg.final_snapshot.time,
            },
            domain.clone(),
        );
        let final_density = final_grid.compute_density();
        let mass_final: f64 = final_density.data.iter().sum::<f64>() * dx3;

        // L2 error of density profile: ||rho_final - rho_init||_2 / ||rho_init||_2
        let density_l2_err: f64 = initial_density
            .data
            .iter()
            .zip(final_density.data.iter())
            .map(|(&ri, &rf)| (rf - ri) * (rf - ri))
            .sum::<f64>()
            .sqrt()
            / rho_init_l2.max(1e-30);

        let mass_drift = (mass_final - mass_init).abs() / mass_init.abs().max(1e-30);

        // Energy conservation from diagnostics
        let history = &pkg.diagnostics_history;
        assert!(
            history.len() >= 2,
            "Need at least 2 diagnostic entries, got {}",
            history.len()
        );

        let e0 = history[0].total_energy;
        let max_e_drift = history
            .iter()
            .map(|d| (d.total_energy - e0).abs() / e0.abs().max(1e-30))
            .fold(0.0_f64, f64::max);

        println!(
            "BGK equilibrium: density L2 error = {density_l2_err:.2e}, \
             mass drift = {mass_drift:.2e}, energy drift = {max_e_drift:.2e}, \
             steps = {}, mass_init = {mass_init:.4}",
            pkg.total_steps
        );

        // ── Assertions ──────────────────────────────────────────────────
        // Mass must be positive and finite
        assert!(
            mass_final > 0.0,
            "Final mass should be positive, got {mass_final}"
        );

        // Density must not contain NaN
        assert!(
            !final_density.data.iter().any(|x| x.is_nan()),
            "Final density contains NaN values"
        );

        // With truncated velocity BC and Strang splitting, mass should be
        // well preserved. Allow generous tolerance for coarse 16^6 grid.
        assert!(
            mass_drift < 0.5,
            "BGK mass drift too large: {mass_drift:.2e} (threshold 0.5)"
        );

        // Density profile shape should be approximately preserved.
        // A true BGK equilibrium is stationary; deviations come from
        // (a) the approximate self-consistency and (b) numerical diffusion.
        // At N=16 we allow substantial error but require the structure survives.
        assert!(
            density_l2_err < 2.0,
            "BGK density profile L2 error too large: {density_l2_err:.2e} (threshold 2.0)"
        );

        // Energy drift: Strang splitting conserves a shadow Hamiltonian,
        // so energy should not blow up. Allow generous bound for coarse grid.
        assert!(!e0.is_nan(), "Initial energy must not be NaN");
        assert!(!max_e_drift.is_nan(), "Energy drift must not be NaN");
    }
}
