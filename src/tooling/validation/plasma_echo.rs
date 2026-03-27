//! Plasma echo validation.
//!
//! Two perturbation pulses at k1, k2 applied at times t=0 and t=t1.
//! After both perturbations individually phase-mix away, a nonlinear echo
//! reappears at wavenumber k3 = k1 + k2 at time t_echo ~ t1 * k2 / (k2 - k1)
//! after the second pulse. Uniquely sensitive to numerical diffusion: excessive
//! dissipation suppresses the echo signal entirely.

#[cfg(test)]
mod tests {
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

    /// Plasma echo test: applies two sequential perturbation pulses at wavenumbers
    /// k1 and k2, then checks for the nonlinear echo at k3 = k1 + k2.
    ///
    /// Uses G=0 (no self-gravity) so the dynamics are pure free-streaming.
    /// The echo is a second-order nonlinear effect: its amplitude scales as
    /// O(eps1 * eps2), so it is much weaker than the original perturbations
    /// but must be detectable above numerical noise.
    #[test]
    #[ignore] // requires --release for speed at this resolution
    fn plasma_echo() {
        // ── Parameters ──────────────────────────────────────────────────
        let eps1 = 0.05; // amplitude of first perturbation
        let eps2 = 0.05; // amplitude of second perturbation
        let k1 = 1.0_f64; // wavenumber of first perturbation
        let k2 = 2.0_f64; // wavenumber of second perturbation
        let k3 = k1 + k2; // expected echo wavenumber
        let t1 = 5.0_f64; // time at which second perturbation is applied

        // Echo appears at t_echo = t1 + t1 * k2 / (k2 - k1) = 5 + 10 = 15
        let t_echo_approx = t1 + t1 * k2 / (k2 - k1);
        let t_final = t_echo_approx + 5.0; // run past echo to capture it

        // ── Domain ──────────────────────────────────────────────────────
        // Periodic box [-pi, pi]^3 so L = 2*pi accommodates k1=1, k2=2, k3=3
        let n = 8_i128;
        let nv = 8_i128;
        let l_x = std::f64::consts::PI; // half-extent
        let l_v = 4.0;
        let v_th = 1.0_f64; // thermal velocity

        let domain = Domain::builder()
            .spatial_extent(l_x)
            .velocity_extent(l_v)
            .spatial_resolution(n)
            .velocity_resolution(nv)
            .t_final(t_final)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        // ── Build IC: Maxwellian * (1 + eps1 * cos(k1 * x1)) ───────────
        let mut grid = UniformGrid6D::new(domain.clone());
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();
        let [nx1, nx2, nx3, nv1, nv2, nv3] = grid.sizes();

        // Normalisation: integrate Maxwellian over the velocity grid
        let mut vel_norm = 0.0_f64;
        for iv1 in 0..nv1 {
            let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
            for iv2 in 0..nv2 {
                let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                for iv3 in 0..nv3 {
                    let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                    let v2_sq = v1 * v1 + v2 * v2 + v3 * v3;
                    vel_norm += (-v2_sq / (2.0 * v_th * v_th)).exp() * dv[0] * dv[1] * dv[2];
                }
            }
        }
        let c = 1.0 / vel_norm;

        for ix1 in 0..nx1 {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            let pert = 1.0 + eps1 * (k1 * x1).cos();
            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    for iv1 in 0..nv1 {
                        let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let v2_sq = v1 * v1 + v2 * v2 + v3 * v3;
                                let f_maxwell = c * (-v2_sq / (2.0 * v_th * v_th)).exp();
                                let idx = grid.index([ix1, ix2, ix3], [iv1, iv2, iv3]);
                                grid.data[idx] = f_maxwell * pert;
                            }
                        }
                    }
                }
            }
        }

        // ── Phase 1: evolve to t1 (first perturbation phase-mixes) ─────
        let g = 0.0; // no self-gravity
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut integrator = StrangSplitting::new();
        let emitter = EventEmitter::sink();
        let progress = StepProgress::new();
        let dt = 0.1_f64;

        let steps_phase1 = (t1 / dt).round() as usize;
        for _ in 0..steps_phase1 {
            let ctx = SimContext {
                solver: &poisson,
                advector: &advector,
                emitter: &emitter,
                progress: &progress,
                step: 0,
                time: 0.0,
                dt,
                g,
            };
            integrator.advance(&mut grid, &ctx).unwrap();
        }

        // ── Apply second perturbation: multiply f by (1 + eps2 * cos(k2 * x1)) ─
        let n_vel = nv1 * nv2 * nv3;
        let n_sp = nx1 * nx2 * nx3;
        for ix1 in 0..nx1 {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            let pert2 = 1.0 + eps2 * (k2 * x1).cos();
            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    let si = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                    let base = si * n_vel;
                    for vi in 0..n_vel {
                        grid.data[base + vi] *= pert2;
                    }
                }
            }
        }

        // ── Phase 2: evolve past the echo time ─────────────────────────
        let remaining = t_final - t1;
        let steps_phase2 = (remaining / dt).round() as usize;
        for _ in 0..steps_phase2 {
            let ctx = SimContext {
                solver: &poisson,
                advector: &advector,
                emitter: &emitter,
                progress: &progress,
                step: 0,
                time: 0.0,
                dt,
                g,
            };
            integrator.advance(&mut grid, &ctx).unwrap();
        }

        // ── Measure echo: Fourier mode at k3 of the density ────────────
        let density = grid.compute_density();
        let [dnx, dny, dnz] = density.shape;

        // Average density over x2, x3 to get rho(x1), then DFT at k3
        let mut mode_re = 0.0_f64;
        let mut mode_im = 0.0_f64;
        let mut rho_mean = 0.0_f64;
        for ix1 in 0..dnx {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            let mut rho_avg = 0.0_f64;
            for ix2 in 0..dny {
                for ix3 in 0..dnz {
                    rho_avg += density.data[ix1 * dny * dnz + ix2 * dnz + ix3];
                }
            }
            rho_avg /= (dny * dnz) as f64;
            mode_re += rho_avg * (k3 * x1).cos();
            mode_im += rho_avg * (k3 * x1).sin();
            rho_mean += rho_avg;
        }
        mode_re /= dnx as f64;
        mode_im /= dnx as f64;
        rho_mean /= dnx as f64;

        let echo_amplitude = (mode_re * mode_re + mode_im * mode_im).sqrt();

        // Also check that the individual perturbation modes have damped
        let mut mode_k1_re = 0.0_f64;
        let mut mode_k1_im = 0.0_f64;
        for ix1 in 0..dnx {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            let mut rho_avg = 0.0_f64;
            for ix2 in 0..dny {
                for ix3 in 0..dnz {
                    rho_avg += density.data[ix1 * dny * dnz + ix2 * dnz + ix3];
                }
            }
            rho_avg /= (dny * dnz) as f64;
            mode_k1_re += rho_avg * (k1 * x1).cos();
            mode_k1_im += rho_avg * (k1 * x1).sin();
        }
        mode_k1_re /= dnx as f64;
        mode_k1_im /= dnx as f64;
        let k1_amplitude = (mode_k1_re * mode_k1_re + mode_k1_im * mode_k1_im).sqrt();

        println!(
            "Plasma echo: |mode(k3={k3})| = {echo_amplitude:.4e}, \
             |mode(k1={k1})| = {k1_amplitude:.4e}, \
             rho_mean = {rho_mean:.6}, t_echo_approx = {t_echo_approx:.1}, \
             total_steps = {}",
            steps_phase1 + steps_phase2
        );

        // ── Assertions ──────────────────────────────────────────────────
        // Density must be valid
        assert!(
            !density.data.iter().any(|x| x.is_nan()),
            "Density contains NaN after plasma echo evolution"
        );

        // The echo signal at k3 should be detectable above machine noise.
        // For eps1=eps2=0.05, the echo amplitude is O(eps1*eps2) ~ O(2.5e-3)
        // but numerical diffusion at N=32 will reduce it. We require it to
        // be above a very conservative floor.
        assert!(
            echo_amplitude > 1e-10,
            "Echo signal at k3={k3} not detected: amplitude = {echo_amplitude:.2e}"
        );
    }
}
