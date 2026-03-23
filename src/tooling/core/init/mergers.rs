//! Two-body merger / interaction initial conditions.
//! Superposition of two isolated equilibria displaced and boosted.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use super::isolated::IsolatedEquilibrium;
use rayon::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// f(x,v,t₀) = f₁(x−x₁, v−v₁) + f₂(x−x₂, v−v₂).
/// Exact for collisionless systems before interaction begins.
pub struct MergerIC {
    pub body1: Box<dyn IsolatedEquilibrium>,
    pub mass1: Decimal,
    pub body2: Box<dyn IsolatedEquilibrium>,
    pub mass2: Decimal,
    pub separation: [f64; 3],
    pub relative_velocity: [f64; 3],
    pub impact_parameter: Decimal,
    // Cached f64 values for hot-path computation
    mass1_f64: f64,
    mass2_f64: f64,
    impact_parameter_f64: f64,
}

impl MergerIC {
    /// Create a MergerIC from f64 parameters (backward-compatible).
    pub fn new(
        body1: Box<dyn IsolatedEquilibrium>,
        mass1: f64,
        body2: Box<dyn IsolatedEquilibrium>,
        mass2: f64,
        separation: [f64; 3],
        relative_velocity: [f64; 3],
        impact_parameter: f64,
    ) -> Self {
        Self {
            body1,
            mass1: Decimal::from_f64_retain(mass1).unwrap_or(Decimal::ZERO),
            body2,
            mass2: Decimal::from_f64_retain(mass2).unwrap_or(Decimal::ZERO),
            separation,
            relative_velocity,
            impact_parameter: Decimal::from_f64_retain(impact_parameter).unwrap_or(Decimal::ZERO),
            mass1_f64: mass1,
            mass2_f64: mass2,
            impact_parameter_f64: impact_parameter,
        }
    }

    /// Create a MergerIC from Decimal parameters (exact config).
    pub fn new_decimal(
        body1: Box<dyn IsolatedEquilibrium>,
        mass1: Decimal,
        body2: Box<dyn IsolatedEquilibrium>,
        mass2: Decimal,
        separation: [f64; 3],
        relative_velocity: [f64; 3],
        impact_parameter: Decimal,
    ) -> Self {
        Self {
            body1,
            mass1_f64: mass1.to_f64().unwrap_or(0.0),
            body2,
            mass2_f64: mass2.to_f64().unwrap_or(0.0),
            separation,
            relative_velocity,
            impact_parameter_f64: impact_parameter.to_f64().unwrap_or(0.0),
            mass1,
            mass2,
            impact_parameter,
        }
    }

    /// Sample both components on the grid and sum.
    /// Body 1 is centred at (-sep/2, 0, 0) with velocity (-v_rel/2, 0, 0).
    /// Body 2 is centred at (+sep/2, 0, 0) with velocity (+v_rel/2, 0, 0).
    pub fn sample_on_grid(
        &self,
        domain: &Domain,
        progress: Option<&crate::tooling::core::progress::StepProgress>,
    ) -> PhaseSpaceSnapshot {
        let nx1 = domain.spatial_res.x1 as usize;
        let nx2 = domain.spatial_res.x2 as usize;
        let nx3 = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;

        let dx = domain.dx();
        let dv = domain.dv();
        let lx = domain.lx();
        let lv = domain.lv();

        // Centre-of-mass frame: body 1 at -sep/2, body 2 at +sep/2
        let x1_offset = [
            -self.separation[0] / 2.0,
            -self.separation[1] / 2.0,
            -self.separation[2] / 2.0,
        ];
        let x2_offset = [
            self.separation[0] / 2.0,
            self.separation[1] / 2.0,
            self.separation[2] / 2.0,
        ];
        let v1_offset = [
            -self.relative_velocity[0] / 2.0,
            -self.relative_velocity[1] / 2.0,
            -self.relative_velocity[2] / 2.0,
        ];
        let v2_offset = [
            self.relative_velocity[0] / 2.0,
            self.relative_velocity[1] / 2.0,
            self.relative_velocity[2] / 2.0,
        ];

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
        let mut data = vec![0.0f64; total];

        let counter = std::sync::atomic::AtomicU64::new(0);
        let report_interval = (nx1 / 100).max(1) as u64;

        // Establish 0% baseline so the TUI doesn't jump to a non-zero first value
        if let Some(p) = progress {
            p.set_intra_progress(0, nx1 as u64);
        }

        data.par_chunks_mut(s_x1)
            .enumerate()
            .for_each(|(ix1, chunk)| {
                let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
                for ix2 in 0..nx2 {
                    let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];
                    for ix3 in 0..nx3 {
                        let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                        let base = ix2 * s_x2 + ix3 * s_x3;

                        // Radius from body 1 centre
                        let dx1 = x1 - x1_offset[0];
                        let dy1 = x2 - x1_offset[1];
                        let dz1 = x3 - x1_offset[2];
                        let r1 = (dx1 * dx1 + dy1 * dy1 + dz1 * dz1).sqrt();
                        let phi1 = self.body1.potential(r1);

                        // Radius from body 2 centre
                        let dx2 = x1 - x2_offset[0];
                        let dy2 = x2 - x2_offset[1];
                        let dz2 = x3 - x2_offset[2];
                        let r2 = (dx2 * dx2 + dy2 * dy2 + dz2 * dz2).sqrt();
                        let phi2 = self.body2.potential(r2);

                        for iv1 in 0..nv1 {
                            let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                            for iv2 in 0..nv2 {
                                let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                                for iv3 in 0..nv3 {
                                    let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                    let idx = base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3;

                                    // f₁(x−x₁, v−v₁): energy in body 1's rest frame
                                    let dv1_1 = v1 - v1_offset[0];
                                    let dv1_2 = v2 - v1_offset[1];
                                    let dv1_3 = v3 - v1_offset[2];
                                    let e1 = 0.5 * (dv1_1 * dv1_1 + dv1_2 * dv1_2 + dv1_3 * dv1_3)
                                        + phi1;
                                    let f1 = self.body1.distribution_function(e1, 0.0).max(0.0);

                                    // f₂(x−x₂, v−v₂): energy in body 2's rest frame
                                    let dv2_1 = v1 - v2_offset[0];
                                    let dv2_2 = v2 - v2_offset[1];
                                    let dv2_3 = v3 - v2_offset[2];
                                    let e2 = 0.5 * (dv2_1 * dv2_1 + dv2_2 * dv2_2 + dv2_3 * dv2_3)
                                        + phi2;
                                    let f2 = self.body2.distribution_function(e2, 0.0).max(0.0);

                                    chunk[idx] = f1 + f2;
                                }
                            }
                        }
                    }
                }

                if let Some(p) = progress {
                    let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if c.is_multiple_of(report_interval) {
                        p.set_intra_progress(c, nx1 as u64);
                    }
                }
            });

        PhaseSpaceSnapshot {
            data,
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time: 0.0,
        }
    }

    /// Check that both components fit within the domain.
    pub fn validate(&self, domain: &Domain) -> anyhow::Result<()> {
        let lx = domain.lx()[0];
        let sep_max = self
            .separation
            .iter()
            .map(|s| s.abs())
            .fold(0.0_f64, f64::max);
        if sep_max / 2.0 > lx * 0.9 {
            anyhow::bail!(
                "Merger separation {:.2} exceeds 90% of domain half-extent {:.2}",
                sep_max,
                lx
            );
        }
        Ok(())
    }
}
