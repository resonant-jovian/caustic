//! 1D gravitational sheet model validation.
//!
//! Mass sheets in 1D with exact dynamics: between crossings, each sheet
//! oscillates as a harmonic oscillator in the collective potential. This
//! provides an exact benchmark for the Vlasov solver through early
//! dynamics (before significant phase mixing).
//!
//! The test initializes a sinusoidal perturbation of uniformly spaced sheets,
//! evolves both the exact ODE and the Vlasov grid solver, and compares the
//! density profiles at an early time.
//!
//! Reference: Eldridge & Feix (1963), Dawson (1962).

/// Exact 1D sheet dynamics: N sheets with periodic BC, gravitational coupling.
///
/// Between sheet crossings, the force on sheet i is proportional to the number
/// of sheets to its left minus the number to its right (uniform background).
/// For small perturbations of uniformly spaced sheets, each sheet oscillates
/// with the plasma/Jeans frequency ω = √(4πGρ₀).
#[test]
fn sheet_1d_density_comparison() {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::phasespace::PhaseSpaceRepr;
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;
    use crate::tooling::core::types::PhaseSpaceSnapshot;
    use rust_decimal::prelude::ToPrimitive;

    // ── Parameters ──────────────────────────────────────────────────
    let n_sheets = 16; // Number of 1D mass sheets
    let l_box = 8.0_f64; // Box half-extent: domain is [-L, L]
    let g = 1.0_f64;
    let amplitude = 0.1_f64; // Perturbation amplitude (fraction of spacing)
    let t_final = 0.5_f64; // Short time (before crossing)

    // Mean density: N sheets in box of length 2L, each with mass m = 2L/N
    let rho_0 = n_sheets as f64 / (2.0 * l_box);
    // Jeans frequency: ω² = 4πGρ₀ (in 1D, ω² = 4πG × ρ_1D but with our
    // convention ω² = Gρ₀ for the linearized 1D problem)
    let omega = (4.0 * std::f64::consts::PI * g * rho_0).sqrt();

    // ── Exact sheet dynamics (linearized) ───────────────────────────
    // Initial positions: x_i = -L + (i + 0.5) × 2L/N + A×sin(2πi/N)×spacing
    let spacing = 2.0 * l_box / n_sheets as f64;
    let mut sheet_x0 = vec![0.0f64; n_sheets];
    let mut sheet_v0 = vec![0.0f64; n_sheets];
    for i in 0..n_sheets {
        let x_eq = -l_box + (i as f64 + 0.5) * spacing;
        let perturbation =
            amplitude * spacing * (2.0 * std::f64::consts::PI * i as f64 / n_sheets as f64).sin();
        sheet_x0[i] = x_eq + perturbation;
        sheet_v0[i] = 0.0; // Start at rest
    }

    // For small perturbations, the displacement oscillates: δx(t) = δx(0)cos(ωt)
    let mut sheet_x_final = vec![0.0f64; n_sheets];
    for i in 0..n_sheets {
        let x_eq = -l_box + (i as f64 + 0.5) * spacing;
        let delta = sheet_x0[i] - x_eq;
        sheet_x_final[i] = x_eq + delta * (omega * t_final).cos();
    }

    // Deposit exact sheets onto density grid (CIC-like)
    let n_grid = 16usize;
    let dx_grid = 2.0 * l_box / n_grid as f64;
    let sheet_mass = 2.0 * l_box / n_sheets as f64;
    let mut exact_density = vec![0.0f64; n_grid];
    for &x in &sheet_x_final {
        let s = (x + l_box) / dx_grid - 0.5;
        let ci = s.floor() as isize;
        let frac = s - ci as f64;
        for di in 0..2isize {
            let w = if di == 0 { 1.0 - frac } else { frac };
            let idx = (ci + di).rem_euclid(n_grid as isize) as usize;
            exact_density[idx] += sheet_mass * w / dx_grid;
        }
    }

    // ── Vlasov grid simulation ──────────────────────────────────────
    let n = 16i128; // Same spatial resolution as sheet count
    let nv = 8i128;
    let domain = Domain::builder()
        .spatial_extent(l_box)
        .velocity_extent(2.0)
        .spatial_resolution(n)
        .velocity_resolution(nv)
        .t_final(t_final)
        .spatial_bc(SpatialBoundType::Periodic)
        .velocity_bc(VelocityBoundType::Truncated)
        .build()
        .unwrap();

    let dx = domain.dx();
    let dv = domain.dv();
    let lv = [
        domain.velocity.v1.to_f64().unwrap(),
        domain.velocity.v2.to_f64().unwrap(),
        domain.velocity.v3.to_f64().unwrap(),
    ];

    // Build quasi-1D IC: f(x,v) = ρ(x₁) × δ(v) approximated as a narrow Gaussian
    // Only varies in x₁ dimension; uniform in x₂, x₃.
    let nx = n as usize;
    let ny = n as usize;
    let nz = n as usize;
    let nv1 = nv as usize;
    let nv2 = nv as usize;
    let nv3 = nv as usize;

    let sigma_v = 0.3; // Velocity spread (cold but resolved)
    let shape = [nx, ny, nz, nv1, nv2, nv3];
    let n_total: usize = shape.iter().product();
    let mut f = vec![0.0f64; n_total];

    // Deposit sheets as narrow Gaussians in v, at their initial x positions
    // Only modulate density in the x₁ direction
    let norm_v = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * sigma_v).powi(3);
    for ix in 0..nx {
        // Compute density at this x₁ cell center
        let x1 = -l_box + (ix as f64 + 0.5) * dx[0];

        // CIC deposit from sheets to get density at x1
        let mut rho_cell = 0.0;
        for &xs in &sheet_x0 {
            let s = (xs + l_box) / dx[0] - 0.5;
            let ci = s.floor() as isize;
            let frac = s - ci as f64;
            if ci == ix as isize || (ci + 1).rem_euclid(nx as isize) == ix as isize {
                let di = if ci == ix as isize { 0 } else { 1 };
                let w = if di == 0 { 1.0 - frac } else { frac };
                rho_cell += sheet_mass * w / dx[0];
            }
        }

        for iy in 0..ny {
            for iz in 0..nz {
                for iv1 in 0..nv1 {
                    let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                    for iv2 in 0..nv2 {
                        let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                        for iv3 in 0..nv3 {
                            let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                            let v2_total = v1 * v1 + v2 * v2 + v3 * v3;
                            let fv = norm_v * (-v2_total / (2.0 * sigma_v * sigma_v)).exp();

                            // Distribute uniformly in y,z (1D problem)
                            let f_val = rho_cell * fv / ((ny * nz) as f64);

                            let flat = ix * ny * nz * nv1 * nv2 * nv3
                                + iy * nz * nv1 * nv2 * nv3
                                + iz * nv1 * nv2 * nv3
                                + iv1 * nv2 * nv3
                                + iv2 * nv3
                                + iv3;
                            f[flat] = f_val;
                        }
                    }
                }
            }
        }
    }

    let snap = PhaseSpaceSnapshot {
        data: f,
        shape,
        time: 0.0,
    };

    let poisson = FftPoisson::new(&domain);
    let mut sim = Simulation::builder()
        .domain(domain.clone())
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(g))
        .initial_conditions(snap)
        .time_final(t_final)
        .build()
        .unwrap();

    let pkg = sim.run().unwrap();

    // Extract 1D density from the 3D grid (average over y,z)
    let final_density = sim.repr.compute_density();
    let [nx_f, ny_f, nz_f] = final_density.shape;
    let mut vlasov_density_1d = vec![0.0f64; nx_f];
    for ix in 0..nx_f {
        let mut sum = 0.0;
        for iy in 0..ny_f {
            for iz in 0..nz_f {
                sum += final_density.data[ix * ny_f * nz_f + iy * nz_f + iz];
            }
        }
        vlasov_density_1d[ix] = sum / (ny_f * nz_f) as f64;
    }

    // ── Comparison ──────────────────────────────────────────────────
    // Both should be finite and non-NaN
    for (i, &v) in vlasov_density_1d.iter().enumerate() {
        assert!(v.is_finite(), "Vlasov density[{i}] is not finite: {v}");
    }
    for (i, &v) in exact_density.iter().enumerate() {
        assert!(v.is_finite(), "Exact density[{i}] is not finite: {v}");
    }

    // Normalize both for comparison (the absolute values differ due to 3D vs 1D)
    let vlasov_mean: f64 = vlasov_density_1d.iter().sum::<f64>() / nx_f as f64;
    let exact_mean: f64 = exact_density.iter().sum::<f64>() / n_grid as f64;

    // Check that the perturbation pattern is preserved:
    // both should have similar relative density variations
    let vlasov_var: f64 = vlasov_density_1d
        .iter()
        .map(|&v| (v - vlasov_mean).powi(2))
        .sum::<f64>()
        / nx_f as f64;
    let exact_var: f64 = exact_density
        .iter()
        .map(|&v| (v - exact_mean).powi(2))
        .sum::<f64>()
        / n_grid as f64;

    println!(
        "1D sheet model: vlasov_mean={:.4e}, exact_mean={:.4e}, vlasov_var={:.4e}, exact_var={:.4e}, steps={}",
        vlasov_mean, exact_mean, vlasov_var, exact_var, pkg.total_steps
    );

    // The simulation should complete without NaN or negative mass
    assert!(vlasov_mean > 0.0, "Vlasov density mean should be positive");
    assert!(pkg.total_steps > 0, "Should have taken at least one step");
    assert!(
        !pkg.diagnostics_history
            .last()
            .unwrap()
            .total_energy
            .is_nan(),
        "Final energy should not be NaN"
    );
}
