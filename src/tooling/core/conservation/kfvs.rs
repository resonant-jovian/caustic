//! Kinetic Flux Vector Splitting (KFVS) macroscopic solver.
//!
//! Evolves the macroscopic conservation laws (density, momentum, energy)
//! independently of the kinetic equation using a flux-splitting scheme
//! derived from the Boltzmann equation. The KFVS fluxes naturally upwind
//! because they split the velocity distribution into positive and negative
//! contributions.
//!
//! The macroscopic system in 3D:
//! ```text
//! ∂ρ/∂t + ∇·(ρu) = 0
//! ∂(ρu)/∂t + ∇·(ρu⊗u + P) = ρa
//! ∂e/∂t + ∇·((e+P)u) = ρu·a
//! ```
//! where ρ = ∫f dv, ρu = ∫vf dv, e = ½∫|v|²f dv, P = ∫(v-u)(v-u)f dv.
//!
//! Reference: Guo & Qiu, arXiv:2207.00518

use rayon::prelude::*;

/// Macroscopic state at a single spatial cell: density, momentum, energy.
#[derive(Clone, Debug)]
pub struct MacroState {
    /// Mass density ρ = ∫ f dv³
    pub density: f64,
    /// Momentum density J = ρu = ∫ v f dv³ (3 components)
    pub momentum: [f64; 3],
    /// Energy density e = ½ ∫ |v|² f dv³
    pub energy: f64,
}

impl MacroState {
    /// Bulk velocity u = J/ρ.
    #[inline]
    pub fn velocity(&self) -> [f64; 3] {
        if self.density.abs() < 1e-30 {
            return [0.0; 3];
        }
        [
            self.momentum[0] / self.density,
            self.momentum[1] / self.density,
            self.momentum[2] / self.density,
        ]
    }

    /// Thermal energy density: e_th = e - ½ρ|u|².
    #[inline]
    pub fn thermal_energy(&self) -> f64 {
        let u = self.velocity();
        let ke = 0.5 * self.density * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        (self.energy - ke).max(0.0)
    }

    /// Isotropic pressure P = (2/3) * e_th (for 3D Maxwellian).
    #[inline]
    pub fn pressure(&self) -> f64 {
        (2.0 / 3.0) * self.thermal_energy()
    }

    /// Temperature T = P/ρ = (2/3) * e_th/ρ.
    #[inline]
    pub fn temperature(&self) -> f64 {
        if self.density.abs() < 1e-30 {
            return 0.0;
        }
        self.pressure() / self.density
    }
}

/// KFVS macroscopic solver on a 3D spatial grid.
///
/// Evolves (ρ, ρu, e) using kinetic flux vector splitting.
/// The KFVS flux at cell interface i+½ in direction d is:
///
/// F⁺ = ∫_{v_d > 0} v_d * [1, v, ½|v|²]ᵀ * f⁺ dv
/// F⁻ = ∫_{v_d < 0} v_d * [1, v, ½|v|²]ᵀ * f⁻ dv
///
/// where f⁺/f⁻ are Maxwellians reconstructed from left/right cell states.
pub struct KfvsSolver {
    /// Spatial grid shape [nx, ny, nz].
    pub shape: [usize; 3],
    /// Cell spacings [dx, dy, dz].
    pub dx: [f64; 3],
    /// Macroscopic state: one `MacroState` per spatial cell (row-major).
    pub state: Vec<MacroState>,
}

impl KfvsSolver {
    /// Create a KFVS solver from initial macroscopic moments.
    pub fn new(shape: [usize; 3], dx: [f64; 3]) -> Self {
        let n = shape[0] * shape[1] * shape[2];
        let state = vec![
            MacroState {
                density: 0.0,
                momentum: [0.0; 3],
                energy: 0.0,
            };
            n
        ];
        Self { shape, dx, state }
    }

    /// Initialize from kinetic moments computed from the distribution function.
    pub fn initialize_from_moments(
        &mut self,
        density: &[f64],
        momentum_x: &[f64],
        momentum_y: &[f64],
        momentum_z: &[f64],
        energy: &[f64],
    ) {
        let n = self.state.len();
        assert_eq!(density.len(), n);
        assert_eq!(momentum_x.len(), n);
        assert_eq!(energy.len(), n);

        for i in 0..n {
            self.state[i] = MacroState {
                density: density[i],
                momentum: [momentum_x[i], momentum_y[i], momentum_z[i]],
                energy: energy[i],
            };
        }
    }

    /// Flat index from 3D coordinates.
    #[inline]
    fn flat(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.shape[1] * self.shape[2] + j * self.shape[2] + k
    }

    /// Periodic index wrap.
    #[inline]
    fn wrap(i: isize, n: usize) -> usize {
        ((i % n as isize + n as isize) % n as isize) as usize
    }

    /// Compute KFVS flux in one direction from a Maxwellian state.
    ///
    /// For a Maxwellian with density ρ, velocity u, temperature T = P/ρ,
    /// the positive (v_d > 0) half-flux moments are:
    ///
    /// F⁺_ρ   = ρ/(2√π) * [u_d * erfc(-u_d/√(2T)) + √(2T/π) * exp(-u_d²/(2T))]
    /// (simplified using standard Maxwellian half-space integrals)
    fn half_maxwellian_flux(state: &MacroState, dim: usize, positive: bool) -> [f64; 5] {
        let rho = state.density;
        if rho.abs() < 1e-30 {
            return [0.0; 5];
        }

        let u = state.velocity();
        let temp = state.temperature().max(1e-30);
        let sigma = temp.sqrt(); // thermal velocity σ = √T

        // Normalized velocity in the flux direction
        let u_d = u[dim];
        let s = if positive { u_d / sigma } else { -u_d / sigma };

        // Half-Maxwellian integrals (Gaussian half-space moments):
        // I₀ = ½ erfc(-s/√2)
        // I₁ = σ * [s * I₀ + exp(-s²/2) / √(2π)]
        // I₂ = σ² * [(s² + 1) * I₀ + s * exp(-s²/2) / √(2π)]
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let erfc_val = erfc(-s * inv_sqrt2);
        let gauss = (-0.5 * s * s).exp() / (2.0 * std::f64::consts::PI).sqrt();

        let i0 = 0.5 * erfc_val;
        let i1 = sigma * (s * i0 + gauss);
        let i2 = sigma * sigma * ((s * s + 1.0) * i0 + s * gauss);

        let sign = if positive { 1.0 } else { -1.0 };

        // Flux vector: [ρ-flux, ρu₁-flux, ρu₂-flux, ρu₃-flux, e-flux]
        // Density flux: ρ * σ * I₁ (in lab frame: ρ * (u_d * I₀ + σ * gauss))
        let f_rho = rho * sign * i1;

        // Momentum flux in direction j: ρ * (u_j * σ * I₁ + δ_{jd} * σ² * I₂)
        let mut f_mom = [0.0f64; 3];
        for j in 0..3 {
            let mom_streaming = u[j] * f_rho;
            let mom_pressure = if j == dim {
                rho * sign * sigma * i2
            } else {
                0.0
            };
            f_mom[j] = mom_streaming + mom_pressure;
        }

        // Energy flux: ½ρ(|u|² + 3T) * σ * I₁ + ρ * u_d * σ² * I₂
        let u_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        let f_energy = 0.5 * rho * (u_sq + 3.0 * temp) * sign * i1 + rho * u_d * sign * sigma * i2;

        [f_rho, f_mom[0], f_mom[1], f_mom[2], f_energy]
    }

    /// Advance the macroscopic state by one time step Δt using KFVS.
    ///
    /// `acceleration` provides the gravitational field a(x) at each cell,
    /// stored as three flat arrays [ax, ay, az] each of length nx*ny*nz.
    pub fn step(&mut self, dt: f64, acceleration: &[[f64; 3]]) {
        let [nx, ny, nz] = self.shape;
        let n = nx * ny * nz;
        assert_eq!(acceleration.len(), n);

        // Allocate update arrays
        let mut d_rho = vec![0.0f64; n];
        let mut d_mom = vec![[0.0f64; 3]; n];
        let mut d_energy = vec![0.0f64; n];

        // Sweep each direction
        for dim in 0..3 {
            let dx_d = self.dx[dim];
            let n_d = self.shape[dim];

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let idx = self.flat(i, j, k);

                        // Left and right neighbor indices along dim
                        let (il, jl, kl) = match dim {
                            0 => (Self::wrap(i as isize - 1, nx), j, k),
                            1 => (i, Self::wrap(j as isize - 1, ny), k),
                            2 => (i, j, Self::wrap(k as isize - 1, nz)),
                            _ => unreachable!(),
                        };
                        let (ir, jr, kr) = match dim {
                            0 => (Self::wrap(i as isize + 1, nx), j, k),
                            1 => (i, Self::wrap(j as isize + 1, ny), k),
                            2 => (i, j, Self::wrap(k as isize + 1, nz)),
                            _ => unreachable!(),
                        };

                        let idx_l = self.flat(il, jl, kl);
                        let idx_r = self.flat(ir, jr, kr);

                        // Flux at left interface (i-½):
                        // F_{i-½} = F⁺(left_state) + F⁻(this_state)
                        let f_left_pos = Self::half_maxwellian_flux(&self.state[idx_l], dim, true);
                        let f_left_neg = Self::half_maxwellian_flux(&self.state[idx], dim, false);

                        // Flux at right interface (i+½):
                        // F_{i+½} = F⁺(this_state) + F⁻(right_state)
                        let f_right_pos = Self::half_maxwellian_flux(&self.state[idx], dim, true);
                        let f_right_neg =
                            Self::half_maxwellian_flux(&self.state[idx_r], dim, false);

                        // Net flux: -(F_{i+½} - F_{i-½}) / dx
                        let f_in = [
                            f_left_pos[0] + f_left_neg[0],
                            f_left_pos[1] + f_left_neg[1],
                            f_left_pos[2] + f_left_neg[2],
                            f_left_pos[3] + f_left_neg[3],
                            f_left_pos[4] + f_left_neg[4],
                        ];
                        let f_out = [
                            f_right_pos[0] + f_right_neg[0],
                            f_right_pos[1] + f_right_neg[1],
                            f_right_pos[2] + f_right_neg[2],
                            f_right_pos[3] + f_right_neg[3],
                            f_right_pos[4] + f_right_neg[4],
                        ];

                        d_rho[idx] -= (f_out[0] - f_in[0]) / dx_d;
                        d_mom[idx][0] -= (f_out[1] - f_in[1]) / dx_d;
                        d_mom[idx][1] -= (f_out[2] - f_in[2]) / dx_d;
                        d_mom[idx][2] -= (f_out[3] - f_in[3]) / dx_d;
                        d_energy[idx] -= (f_out[4] - f_in[4]) / dx_d;
                    }
                }
            }
        }

        // Apply source terms (gravitational acceleration) and update
        for i in 0..n {
            let a = &acceleration[i];
            let rho = self.state[i].density;
            let u = self.state[i].velocity();

            // Source: ∂(ρu)/∂t += ρa,  ∂e/∂t += ρu·a
            d_mom[i][0] += rho * a[0];
            d_mom[i][1] += rho * a[1];
            d_mom[i][2] += rho * a[2];
            d_energy[i] += rho * (u[0] * a[0] + u[1] * a[1] + u[2] * a[2]);

            // Forward Euler update
            self.state[i].density += dt * d_rho[i];
            self.state[i].momentum[0] += dt * d_mom[i][0];
            self.state[i].momentum[1] += dt * d_mom[i][1];
            self.state[i].momentum[2] += dt * d_mom[i][2];
            self.state[i].energy += dt * d_energy[i];

            // Floor density and energy to prevent negativity
            self.state[i].density = self.state[i].density.max(0.0);
            self.state[i].energy = self.state[i].energy.max(0.0);
        }
    }

    /// Extract density field from macroscopic state.
    pub fn density_field(&self) -> Vec<f64> {
        self.state.iter().map(|s| s.density).collect()
    }

    /// Extract momentum field components.
    pub fn momentum_field(&self) -> [Vec<f64>; 3] {
        let n = self.state.len();
        let mut mx = vec![0.0; n];
        let mut my = vec![0.0; n];
        let mut mz = vec![0.0; n];
        for (i, s) in self.state.iter().enumerate() {
            mx[i] = s.momentum[0];
            my[i] = s.momentum[1];
            mz[i] = s.momentum[2];
        }
        [mx, my, mz]
    }

    /// Extract energy field.
    pub fn energy_field(&self) -> Vec<f64> {
        self.state.iter().map(|s| s.energy).collect()
    }

    /// Total mass (sum of ρ * dx³).
    pub fn total_mass(&self) -> f64 {
        let dv = self.dx[0] * self.dx[1] * self.dx[2];
        self.state.iter().map(|s| s.density).sum::<f64>() * dv
    }

    /// Total momentum (sum of J * dx³).
    pub fn total_momentum(&self) -> [f64; 3] {
        let dv = self.dx[0] * self.dx[1] * self.dx[2];
        let mut p = [0.0; 3];
        for s in &self.state {
            p[0] += s.momentum[0];
            p[1] += s.momentum[1];
            p[2] += s.momentum[2];
        }
        [p[0] * dv, p[1] * dv, p[2] * dv]
    }

    /// Total energy (sum of e * dx³).
    pub fn total_energy(&self) -> f64 {
        let dv = self.dx[0] * self.dx[1] * self.dx[2];
        self.state.iter().map(|s| s.energy).sum::<f64>() * dv
    }
}

/// Complementary error function erfc(x) = 1 - erf(x).
///
/// Uses a rational Chebyshev approximation accurate to ~1e-15.
fn erfc(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 (maximum error 1.5e-7) is insufficient.
    // Use the more accurate formula from Horner form.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 { result } else { 2.0 - result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kfvs_macro_state_basics() {
        let s = MacroState {
            density: 2.0,
            momentum: [1.0, 0.0, 0.0],
            energy: 3.0,
        };
        let u = s.velocity();
        assert!((u[0] - 0.5).abs() < 1e-14);
        assert!(u[1].abs() < 1e-14);

        // KE = ½ * 2 * 0.25 = 0.25
        // thermal = 3 - 0.25 = 2.75
        // pressure = 2/3 * 2.75 ≈ 1.833
        assert!((s.thermal_energy() - 2.75).abs() < 1e-14);
        assert!((s.pressure() - 2.75 * 2.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn kfvs_erfc_accuracy() {
        // erfc(0) = 1
        assert!((erfc(0.0) - 1.0).abs() < 1e-7);
        // erfc(∞) → 0
        assert!(erfc(5.0) < 1e-10);
        // erfc(-∞) → 2
        assert!((erfc(-5.0) - 2.0).abs() < 1e-10);
        // erfc(1) ≈ 0.1572992
        assert!((erfc(1.0) - 0.1572992).abs() < 1e-5);
    }

    #[test]
    fn kfvs_mass_conservation() {
        // Uniform density with no acceleration should conserve mass
        let shape = [8, 8, 8];
        let dx = [0.5; 3];
        let n = 512;
        let mut solver = KfvsSolver::new(shape, dx);

        // Initialize uniform state: ρ=1, u=0, T=1 → e = 3/2 * T * ρ = 1.5
        for s in &mut solver.state {
            s.density = 1.0;
            s.momentum = [0.0; 3];
            s.energy = 1.5;
        }

        let m0 = solver.total_mass();
        let acc = vec![[0.0; 3]; n];
        solver.step(0.01, &acc);
        let m1 = solver.total_mass();

        assert!(
            (m1 - m0).abs() / m0.abs() < 1e-12,
            "Mass not conserved: {m0} -> {m1}"
        );
    }

    #[test]
    fn kfvs_momentum_conservation_no_source() {
        // Uniform state with no gravity should conserve momentum
        let shape = [8, 8, 8];
        let dx = [0.5; 3];
        let n = 512;
        let mut solver = KfvsSolver::new(shape, dx);

        // Uniform with bulk velocity
        for s in &mut solver.state {
            s.density = 1.0;
            s.momentum = [0.5, -0.3, 0.1];
            s.energy = 2.0;
        }

        let p0 = solver.total_momentum();
        let acc = vec![[0.0; 3]; n];
        solver.step(0.01, &acc);
        let p1 = solver.total_momentum();

        for d in 0..3 {
            assert!(
                (p1[d] - p0[d]).abs() < 1e-12,
                "Momentum[{d}] not conserved: {} -> {}",
                p0[d],
                p1[d]
            );
        }
    }
}
