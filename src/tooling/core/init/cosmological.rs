//! Zel'dovich approximation initial conditions for cosmological structure formation.
//! f is a 3D sheet in 6D phase space (cold dark matter).

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use rust_decimal::prelude::ToPrimitive;

/// Cosmological parameters for Friedmann background.
pub struct CosmologyParams {
    pub h0: f64,
    pub omega_m: f64,
    pub omega_lambda: f64,
    pub a_init: f64,
}

/// 1D matter power spectrum P(k) for seeding perturbations.
pub struct PowerSpectrum {
    /// (k, P(k)) pairs.
    pub values: Vec<(f64, f64)>,
}

/// Zel'dovich pancake IC: cold dark matter sheet.
/// f(x,v,t₀) = ρ̄·δ³(v − v₀(x)) where v₀ is the Zel'dovich velocity field.
pub struct ZeldovichIC {
    pub mean_density: f64,
    pub h0: f64,
    pub omega_m: f64,
    pub omega_lambda: f64,
    pub scale_factor_init: f64,
    pub random_seed: u64,
}

impl ZeldovichIC {
    pub fn new(mean_density: f64, cosmology: CosmologyParams, seed: u64) -> Self {
        Self {
            mean_density,
            h0: cosmology.h0,
            omega_m: cosmology.omega_m,
            omega_lambda: cosmology.omega_lambda,
            scale_factor_init: cosmology.a_init,
            random_seed: seed,
        }
    }

    /// Create a simple single-mode Zel'dovich pancake for testing.
    ///
    /// v₀(x) = -A·sin(k·x₁)
    /// f(x,v) = (ρ̄/S) · exp(-(v - v₀(x))² / (2σ_v²))
    ///
    /// where S normalises the Gaussian to unity over velocity space.
    pub fn new_single_mode(
        mean_density: f64,
        amplitude: f64,
        wavenumber: f64,
        sigma_v: f64,
    ) -> ZeldovichSingleMode {
        ZeldovichSingleMode {
            mean_density,
            amplitude,
            wavenumber,
            sigma_v,
        }
    }

    /// Generate the Zel'dovich displacement field s(q) from P(k) via FFT.
    ///
    /// Requires `rand` crate for Gaussian random field generation.
    /// Use `ZeldovichSingleMode` for single-mode validation tests.
    pub fn displacement_field(&self, _domain: &Domain) -> [Vec<f64>; 3] {
        todo!(
            "requires `rand` crate for Gaussian random field generation; use ZeldovichSingleMode for single-mode tests"
        )
    }

    /// Zel'dovich velocity v₀(q) = H·f·s (growing mode).
    ///
    /// Requires displacement_field which needs `rand` crate.
    pub fn velocity_field(&self, _domain: &Domain) -> [Vec<f64>; 3] {
        todo!(
            "requires displacement_field which needs `rand` crate; use ZeldovichSingleMode for single-mode tests"
        )
    }

    /// Sample onto 6D grid as a thin Gaussian in velocity centred on v₀(x).
    ///
    /// Requires velocity_field which needs `rand` crate.
    pub fn sample_on_grid(&self, _domain: &Domain) -> PhaseSpaceSnapshot {
        todo!(
            "requires velocity_field which needs `rand` crate; use ZeldovichSingleMode for single-mode tests"
        )
    }
}

/// Simple single-mode Zel'dovich pancake IC for validation tests.
pub struct ZeldovichSingleMode {
    pub mean_density: f64,
    pub amplitude: f64,
    pub wavenumber: f64,
    pub sigma_v: f64,
}

impl ZeldovichSingleMode {
    /// Sample f(x,v) = ρ̄/(√(2π)σ_v)³ · exp(-(v - v₀(x))²/(2σ_v²))
    /// where v₀(x) = (-A·sin(k·x₁), 0, 0)
    pub fn sample_on_grid(&self, domain: &Domain) -> PhaseSpaceSnapshot {
        use std::f64::consts::PI;

        let nx1 = domain.spatial_res.x1 as usize;
        let nx2 = domain.spatial_res.x2 as usize;
        let nx3 = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;

        let dx = domain.dx();
        let dv = domain.dv();
        let lx = [
            domain.spatial.x1.to_f64().unwrap(),
            domain.spatial.x2.to_f64().unwrap(),
            domain.spatial.x3.to_f64().unwrap(),
        ];
        let lv = [
            domain.velocity.v1.to_f64().unwrap(),
            domain.velocity.v2.to_f64().unwrap(),
            domain.velocity.v3.to_f64().unwrap(),
        ];

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
        let mut data = vec![0.0f64; total];

        let sigma = self.sigma_v;
        let norm = self.mean_density / ((2.0 * PI).sqrt() * sigma).powi(3);

        for ix1 in 0..nx1 {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            let v0x = -self.amplitude * (self.wavenumber * x1).sin();

            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    let base = ix1 * s_x1 + ix2 * s_x2 + ix3 * s_x3;

                    for iv1 in 0..nv1 {
                        let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let dv1 = v1 - v0x;
                                let dv2 = v2;
                                let dv3 = v3;
                                let v2sq = dv1 * dv1 + dv2 * dv2 + dv3 * dv3;
                                let f = norm * (-v2sq / (2.0 * sigma * sigma)).exp();
                                data[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f;
                            }
                        }
                    }
                }
            }
        }

        PhaseSpaceSnapshot {
            data,
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time: 0.0,
        }
    }
}

/// Growth factor D(a) for flat ΛCDM cosmology.
///
/// Uses the Carroll-Press-Turner (1992) approximation:
///   D(a) ≈ (5/2) Ω_m(a) / [Ω_m(a)^{4/7} - Ω_Λ(a) + (1 + Ω_m(a)/2)(1 + Ω_Λ(a)/70)]
///
/// where Ω_m(a) = Ω_m / (Ω_m + Ω_Λ a³) and Ω_Λ(a) = Ω_Λ a³ / (Ω_m + Ω_Λ a³).
/// Normalized so D(a) → a in the matter-dominated era.
pub fn growth_factor(a: f64, params: &CosmologyParams) -> f64 {
    let om = params.omega_m;
    let ol = params.omega_lambda;

    // E²(a) = Ω_m/a³ + Ω_Λ  (flat universe, Ω_k = 0)
    let a3 = a * a * a;
    let e2 = om / a3 + ol;

    // Ω_m(a) and Ω_Λ(a) at scale factor a
    let om_a = om / (a3 * e2);
    let ol_a = ol / e2;

    // Carroll-Press-Turner approximation
    let d_unnorm =
        2.5 * om_a / (om_a.powf(4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0));

    // Normalize: in matter domination D ∝ a, so normalize to D(1) and scale
    let e2_1 = om + ol;
    let om_1 = om / e2_1;
    let ol_1 = ol / e2_1;
    let d_1 = 2.5 * om_1 / (om_1.powf(4.0 / 7.0) - ol_1 + (1.0 + om_1 / 2.0) * (1.0 + ol_1 / 70.0));

    a * d_unnorm / d_1
}
