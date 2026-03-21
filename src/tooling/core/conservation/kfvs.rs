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

        let dx = self.dx;
        let shape = self.shape;
        let state = &self.state;

        // Phase 1: Compute flux divergence for all 3 dimensions per cell in parallel.
        // Each cell reads only from neighbors (immutable), producing (d_rho, d_mom, d_energy).
        let updates: Vec<(f64, [f64; 3], f64)> = (0..n)
            .into_par_iter()
            .map(|flat| {
                let i = flat / (ny * nz);
                let j = (flat / nz) % ny;
                let k = flat % nz;

                let mut dr = 0.0f64;
                let mut dm = [0.0f64; 3];
                let mut de = 0.0f64;

                for dim in 0..3 {
                    let dx_d = dx[dim];
                    let n_d = shape[dim];

                    let idx_l = match dim {
                        0 => Self::wrap(i as isize - 1, nx) * ny * nz + j * nz + k,
                        1 => i * ny * nz + Self::wrap(j as isize - 1, ny) * nz + k,
                        2 => i * ny * nz + j * nz + Self::wrap(k as isize - 1, nz),
                        _ => unreachable!(),
                    };
                    let idx_r = match dim {
                        0 => Self::wrap(i as isize + 1, nx) * ny * nz + j * nz + k,
                        1 => i * ny * nz + Self::wrap(j as isize + 1, ny) * nz + k,
                        2 => i * ny * nz + j * nz + Self::wrap(k as isize + 1, nz),
                        _ => unreachable!(),
                    };

                    // Flux at left interface (i-½)
                    let fl_pos = Self::half_maxwellian_flux(&state[idx_l], dim, true);
                    let fl_neg = Self::half_maxwellian_flux(&state[flat], dim, false);

                    // Flux at right interface (i+½)
                    let fr_pos = Self::half_maxwellian_flux(&state[flat], dim, true);
                    let fr_neg = Self::half_maxwellian_flux(&state[idx_r], dim, false);

                    // Net flux divergence: -(F_{i+½} - F_{i-½}) / dx
                    for c in 0..5 {
                        let f_in = fl_pos[c] + fl_neg[c];
                        let f_out = fr_pos[c] + fr_neg[c];
                        let div = -(f_out - f_in) / dx_d;
                        match c {
                            0 => dr += div,
                            1 => dm[0] += div,
                            2 => dm[1] += div,
                            3 => dm[2] += div,
                            4 => de += div,
                            _ => unreachable!(),
                        }
                    }
                }

                (dr, dm, de)
            })
            .collect();

        // Phase 2: Apply source terms + Euler update
        self.state
            .par_iter_mut()
            .zip(updates.par_iter())
            .zip(acceleration.par_iter())
            .for_each(|((s, &(dr, dm, de)), a)| {
                let rho = s.density;
                let u = s.velocity();

                // Source: ∂(ρu)/∂t += ρa,  ∂e/∂t += ρu·a
                let dm_src = [rho * a[0], rho * a[1], rho * a[2]];
                let de_src = rho * (u[0] * a[0] + u[1] * a[1] + u[2] * a[2]);

                // Forward Euler update
                s.density = (s.density + dt * dr).max(0.0);
                s.momentum[0] += dt * (dm[0] + dm_src[0]);
                s.momentum[1] += dt * (dm[1] + dm_src[1]);
                s.momentum[2] += dt * (dm[2] + dm_src[2]);
                s.energy = (s.energy + dt * (de + de_src)).max(0.0);
            });
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
        self.state.par_iter().map(|s| s.density).sum::<f64>() * dv
    }

    /// Total momentum (sum of J * dx³).
    pub fn total_momentum(&self) -> [f64; 3] {
        let dv = self.dx[0] * self.dx[1] * self.dx[2];
        let p = self
            .state
            .par_iter()
            .fold(
                || [0.0f64; 3],
                |mut acc, s| {
                    acc[0] += s.momentum[0];
                    acc[1] += s.momentum[1];
                    acc[2] += s.momentum[2];
                    acc
                },
            )
            .reduce(
                || [0.0f64; 3],
                |mut a, b| {
                    a[0] += b[0];
                    a[1] += b[1];
                    a[2] += b[2];
                    a
                },
            );
        [p[0] * dv, p[1] * dv, p[2] * dv]
    }

    /// Total energy (sum of e * dx³).
    pub fn total_energy(&self) -> f64 {
        let dv = self.dx[0] * self.dx[1] * self.dx[2];
        self.state.par_iter().map(|s| s.energy).sum::<f64>() * dv
    }
}

/// Complementary error function erfc(x) = 1 - erf(x).
///
/// Uses rational approximations from Sun/fdlibm (s_erf.c), accurate to ~1e-15.
/// Four regions with minimax rational approximations.
#[allow(clippy::excessive_precision)]
fn erfc(x: f64) -> f64 {
    let ax = x.abs();

    let result = if ax < 0.84375 {
        // Region 1: |x| < 0.84375 — compute erf via rational P/Q, then 1-erf
        if ax < 1.0 / ((1u64 << 28) as f64) {
            return 1.0 - x;
        }
        let z = x * x;
        let r = 1.28379167095512558561e-01
            + z * (-3.25042107247001499370e-01
                + z * (-2.84817495755985104766e-02
                    + z * (-5.77027029648944159157e-03 + z * -2.37630166566501626084e-05)));
        let s = 1.0
            + z * (3.97917223959155352819e-01
                + z * (6.50222499887672944485e-02
                    + z * (5.08130628187576562776e-03
                        + z * (1.32494738004321644526e-04 + z * -3.96022827877536812320e-06))));
        if ax < 0.25 {
            1.0 - (x + x * r / s)
        } else {
            0.5 - (x * r / s + (x - 0.5))
        }
    } else if ax < 1.25 {
        // Region 2: 0.84375 <= |x| < 1.25 — rational approx in (|x|-1)
        let s = ax - 1.0;
        let p = -2.36211856075265944077e-03
            + s * (4.14856118683748331666e-01
                + s * (-3.72207876035701323847e-01
                    + s * (3.18346619901161753674e-01
                        + s * (-1.10894694282396677476e-01
                            + s * (3.54783043195201877747e-02
                                + s * -2.16637559983254089680e-03)))));
        let q = 1.0
            + s * (1.06420880400844228286e-01
                + s * (5.40397917702171048937e-01
                    + s * (7.18286544141962539399e-02
                        + s * (1.26171219808761642112e-01
                            + s * (1.36370839120290507362e-02 + s * 1.19844998467991074170e-02)))));
        // erfc(1) ≈ 0.15493708848953577
        if x >= 0.0 {
            0.15493708848953577 - p / q
        } else {
            1.0 + (0.84506291151046423 + p / q)
        }
    } else if ax < 28.0 {
        // Region 3: 1.25 <= |x| < 28 — erfc via exp(-x²) * P(|x|)/Q(|x|)/|x|
        let s = 1.0 / (ax * ax);
        let (r, ss) = if ax < 2.857142857142857 {
            // 1/0.35 ≈ 2.857
            let r = -9.86494403484714822705e-03
                + s * (-6.93858572707181764372e-01
                    + s * (-1.05586262253232909814e+01
                        + s * (-6.26190508608552143480e+01
                            + s * (-1.62396476920817928905e+02
                                + s * (-1.84605092906711035994e+02
                                    + s * (-8.12874355063065934246e+01
                                        + s * -9.81432934416914548592e+00))))));
            let ss = 1.0
                + s * (1.96512716674392571292e+01
                    + s * (1.37657754143519702237e+02
                        + s * (4.34565877475229228608e+02
                            + s * (6.45387271733267880594e+02
                                + s * (4.29008140027567833386e+02
                                    + s * (1.08635005541779435134e+02
                                        + s * (6.57024977031928170135e+00
                                            + s * -6.04244152148580987438e-02)))))));
            (r, ss)
        } else {
            let r = -9.86494292470009928597e-03
                + s * (-7.99283237680523006574e-01
                    + s * (-1.77579549177547519889e+01
                        + s * (-1.60636384855557935030e+02
                            + s * (-6.37566443368389085394e+02
                                + s * (-1.02509513161107724954e+03
                                    + s * -4.83519191608651397019e+02)))));
            let ss = 1.0
                + s * (3.03380607875625778203e+01
                    + s * (3.25792512996573918826e+02
                        + s * (1.53672958608443695994e+03
                            + s * (3.19985821950859553908e+03
                                + s * (2.55305040643316442583e+03
                                    + s * 4.74528541206955367215e+02)))));
            (r, ss)
        };
        let inv_sqrtpi = 0.5641895835477562869;
        (-ax * ax).exp() * (inv_sqrtpi + r / ss) / ax
    } else {
        // |x| >= 28: erfc is negligibly small
        0.0
    };

    if x < 0.0 { 2.0 - result } else { result }
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
