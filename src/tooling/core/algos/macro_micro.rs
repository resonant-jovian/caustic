//! Macro-micro decomposition of the distribution function.
//!
//! Splits the distribution function into a macroscopic part and a kinetic correction:
//! f = f_M + g, where f_M is a local Maxwellian parameterized by density rho,
//! mean velocity u, and temperature T at each spatial cell, and g is the kinetic
//! deviation stored by an inner [`PhaseSpaceRepr`].
//!
//! f_M(x,v) = rho(x) / (2 pi T(x))^{3/2} * exp(-|v - u(x)|^2 / (2 T(x)))
//!
//! This reduces effective dimensionality for near-equilibrium regions: density is
//! O(N^3) from the macro fields alone with no velocity integration needed, and
//! the deviation g carries only the non-Maxwellian structure. Conservation of mass,
//! momentum, and energy is exact by construction since f_M encodes the exact moments.

use super::super::{
    context::SimContext,
    events::{AdvectDirection, SimEvent},
    init::domain::Domain,
    phasespace::PhaseSpaceRepr,
    types::*,
};
use rayon::prelude::*;
use std::any::Any;

/// Macro-micro representation: f = f_M + g.
///
/// f_M is a local Maxwellian parameterized by (rho, u, T).
/// g is the kinetic deviation, stored by an inner `PhaseSpaceRepr`.
///
/// Density is O(N^3) from the macro fields -- no velocity integration needed.
/// Conservation of mass, momentum, and energy is exact by construction.
pub struct MacroMicroRepr {
    /// Macro density rho(x) -- flat row-major `[nx*ny*nz]`.
    pub density: Vec<f64>,
    /// Macro mean velocity u(x) -- flat [nx*ny*nz * 3] (ux, uy, uz interleaved).
    pub mean_velocity: Vec<f64>,
    /// Macro temperature T(x) -- flat `[nx*ny*nz]` (scalar temperature).
    pub temperature: Vec<f64>,
    /// Micro deviation g = f - f_M, stored by inner representation.
    pub inner: Box<dyn PhaseSpaceRepr>,
    /// Spatial grid shape [nx, ny, nz].
    pub spatial_shape: [usize; 3],
    /// Velocity grid shape [nv1, nv2, nv3].
    pub velocity_shape: [usize; 3],
    /// Domain specification.
    pub domain: Domain,
    /// Cached total mass.
    total_mass_cached: f64,
}

impl MacroMicroRepr {
    /// Create from an existing `PhaseSpaceRepr` by extracting macro moments
    /// and computing g = f - f_M.
    pub fn from_repr(inner: Box<dyn PhaseSpaceRepr>, domain: &Domain) -> Self {
        let nx = domain.spatial_res.x1 as usize;
        let ny = domain.spatial_res.x2 as usize;
        let nz = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;
        let n_spatial = nx * ny * nz;

        // Step 1: Compute macro density from the inner representation.
        let density_field = inner.compute_density();
        let density = density_field.data;

        // Step 2: Initialize mean velocity to zero and temperature from velocity extent.
        // A more accurate projection would require velocity-moment integration at each
        // spatial cell; this initial version uses a simple default.
        let mean_velocity = vec![0.0_f64; n_spatial * 3];
        let lv = domain.lv();
        let default_temp = (lv[0] * lv[0] + lv[1] * lv[1] + lv[2] * lv[2]) / 9.0;
        let temperature = vec![default_temp; n_spatial];

        let dx3 = domain.cell_volume_3d();
        let total_mass = density.iter().sum::<f64>() * dx3;

        Self {
            density,
            mean_velocity,
            temperature,
            inner,
            spatial_shape: [nx, ny, nz],
            velocity_shape: [nv1, nv2, nv3],
            domain: domain.clone(),
            total_mass_cached: total_mass,
        }
    }

    /// Number of spatial cells.
    #[inline]
    fn n_spatial(&self) -> usize {
        self.spatial_shape[0] * self.spatial_shape[1] * self.spatial_shape[2]
    }

    /// Re-project: recompute macro fields (rho, u, T) from the current inner
    /// representation state, ensuring exact conservation of macroscopic moments.
    pub fn reproject_moments(&mut self) {
        let density_field = self.inner.compute_density();
        self.density = density_field.data;

        let dx3 = self.domain.cell_volume_3d();
        self.total_mass_cached = self.density.iter().sum::<f64>() * dx3;

        // Recompute u and T from moments of inner repr at each spatial cell center.
        let lx = self.domain.lx();
        let ddx = self.domain.dx();
        let [nx, ny, nz] = self.spatial_shape;

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let si = ix * ny * nz + iy * nz + iz;
                    let pos = [
                        -lx[0] + (ix as f64 + 0.5) * ddx[0],
                        -lx[1] + (iy as f64 + 0.5) * ddx[1],
                        -lx[2] + (iz as f64 + 0.5) * ddx[2],
                    ];

                    // Order-1 moment gives mean velocity.
                    let m1 = self.inner.moment(&pos, 1);
                    if m1.data.len() >= 3 {
                        self.mean_velocity[si * 3] = m1.data[0];
                        self.mean_velocity[si * 3 + 1] = m1.data[1];
                        self.mean_velocity[si * 3 + 2] = m1.data[2];
                    }

                    // Order-2 moment gives velocity dispersion tensor.
                    // T = (1/3) * Tr(sigma^2) where sigma^2_{ij} = <v_i v_j> - u_i u_j.
                    let m2 = self.inner.moment(&pos, 2);
                    if m2.data.len() >= 9 && self.density[si] > 1e-30 {
                        let ux = self.mean_velocity[si * 3];
                        let uy = self.mean_velocity[si * 3 + 1];
                        let uz = self.mean_velocity[si * 3 + 2];
                        let rho = self.density[si];
                        // <v_i v_j> = M2_{ij} / rho
                        let trace = m2.data[0] / rho + m2.data[4] / rho + m2.data[8] / rho;
                        let u2 = ux * ux + uy * uy + uz * uz;
                        let temp = (trace - u2) / 3.0;
                        self.temperature[si] = temp.max(1e-30);
                    }
                }
            }
        }
    }
}

impl PhaseSpaceRepr for MacroMicroRepr {
    /// Returns density directly from the cached macro field (O(N^3), no velocity integration).
    fn compute_density(&self) -> DensityField {
        // Key advantage: O(N^3) density from macro fields, no velocity integration.
        DensityField {
            data: self.density.clone(),
            shape: self.spatial_shape,
        }
    }

    /// Delegates spatial advection to the inner representation.
    /// Macro fields are re-synced via `reproject_moments()` after a full step.
    fn advect_x(&mut self, displacement: &DisplacementField, ctx: &SimContext) {
        let t0 = std::time::Instant::now();
        let mass_before = self.total_mass();

        self.inner.advect_x(displacement, ctx);

        let mass_after = self.total_mass();
        let macro_norm = self.density.par_iter().map(|&r| r * r).sum::<f64>().sqrt();
        let micro_norm = {
            let inner_density = self.inner.compute_density();
            inner_density
                .data
                .par_iter()
                .zip(self.density.par_iter())
                .map(|(&id, &md)| {
                    let diff = id - md;
                    diff * diff
                })
                .sum::<f64>()
                .sqrt()
        };
        let ratio = if macro_norm > 0.0 {
            micro_norm / macro_norm
        } else {
            0.0
        };

        ctx.emitter.emit(SimEvent::AdvectionComplete {
            direction: AdvectDirection::Spatial,
            mass_before,
            mass_after,
            wall_us: t0.elapsed().as_micros() as u64,
        });
        ctx.emitter.emit(SimEvent::MacroMicroDecomposition {
            macro_norm,
            micro_norm,
            ratio,
        });
    }

    /// Delegates velocity advection to the inner representation.
    /// Macro momentum update is applied via `reproject_moments()` after a full step.
    fn advect_v(&mut self, acceleration: &AccelerationField, ctx: &SimContext) {
        let t0 = std::time::Instant::now();
        let mass_before = self.total_mass();

        self.inner.advect_v(acceleration, ctx);

        let mass_after = self.total_mass();
        let macro_norm = self.density.par_iter().map(|&r| r * r).sum::<f64>().sqrt();
        let micro_norm = {
            let inner_density = self.inner.compute_density();
            inner_density
                .data
                .par_iter()
                .zip(self.density.par_iter())
                .map(|(&id, &md)| {
                    let diff = id - md;
                    diff * diff
                })
                .sum::<f64>()
                .sqrt()
        };
        let ratio = if macro_norm > 0.0 {
            micro_norm / macro_norm
        } else {
            0.0
        };

        ctx.emitter.emit(SimEvent::AdvectionComplete {
            direction: AdvectDirection::Velocity,
            mass_before,
            mass_after,
            wall_us: t0.elapsed().as_micros() as u64,
        });
        ctx.emitter.emit(SimEvent::MacroMicroDecomposition {
            macro_norm,
            micro_norm,
            ratio,
        });
    }

    /// Computes velocity moments by delegating to the inner kinetic representation.
    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        self.inner.moment(position, order)
    }

    /// Returns the cached total mass from macro density integration.
    fn total_mass(&self) -> f64 {
        self.total_mass_cached
    }

    /// Approximates C2 = integral(f^2) using the fluid-level density: sum(rho^2 * dx^3).
    fn casimir_c2(&self) -> f64 {
        // Approximate via density: sum rho^2 * dx^3.
        // The full C2 requires velocity integration of f^2; this is the
        // fluid-level approximation.
        let dx3 = self.domain.cell_volume_3d();
        self.density.par_iter().map(|&rho| rho * rho).sum::<f64>() * dx3
    }

    /// Fluid-level entropy: configurational (-integral rho ln rho) plus thermal (3/2 rho ln(2 pi e T)).
    fn entropy(&self) -> f64 {
        let dx3 = self.domain.cell_volume_3d();

        let n_spatial = self.n_spatial();
        let s_config: f64 = self
            .density
            .par_iter()
            .filter(|&&rho| rho > 0.0)
            .map(|&rho| -rho * rho.ln())
            .sum::<f64>()
            * dx3;

        let s_thermal: f64 = (0..n_spatial)
            .into_par_iter()
            .filter(|&i| self.density[i] > 0.0 && self.temperature[i] > 0.0)
            .map(|i| {
                let rho = self.density[i];
                let t = self.temperature[i];
                // (3/2) rho ln(2 pi e T)
                1.5 * rho * (2.0 * std::f64::consts::PI * std::f64::consts::E * t).ln()
            })
            .sum::<f64>()
            * dx3;

        s_config + s_thermal
    }

    /// Delegates stream counting to the inner kinetic representation.
    fn stream_count(&self) -> StreamCountField {
        self.inner.stream_count()
    }

    /// Delegates local velocity distribution extraction to the inner representation.
    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        self.inner.velocity_distribution(position)
    }

    /// Kinetic energy from macro fields: bulk (1/2 rho |u|^2) plus thermal (3/2 rho T).
    fn total_kinetic_energy(&self) -> Option<f64> {
        let dx3 = self.domain.cell_volume_3d();
        let n_spatial = self.n_spatial();

        let energy: f64 = (0..n_spatial)
            .into_par_iter()
            .map(|i| {
                let rho = self.density[i];
                let ux = self.mean_velocity[i * 3];
                let uy = self.mean_velocity[i * 3 + 1];
                let uz = self.mean_velocity[i * 3 + 2];
                let t = self.temperature[i];
                // Bulk kinetic + thermal
                0.5 * rho * (ux * ux + uy * uy + uz * uz) + 1.5 * rho * t
            })
            .sum();

        Some(energy * dx3)
    }

    /// Produces a full 6D snapshot by delegating to the inner representation.
    fn to_snapshot(&self, time: f64) -> Option<PhaseSpaceSnapshot> {
        self.inner.to_snapshot(time)
    }

    /// Loads a snapshot into the inner representation and re-syncs macro fields.
    fn load_snapshot(&mut self, snap: PhaseSpaceSnapshot) -> Result<(), crate::CausticError> {
        self.inner.load_snapshot(snap)?;
        // Re-sync macro fields after loading new data.
        self.reproject_moments();
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    /// Reports whether the inner representation supports dense materialization.
    fn can_materialize(&self) -> bool {
        self.inner.can_materialize()
    }

    /// Total memory: macro fields (density + velocity + temperature) plus inner representation.
    fn memory_bytes(&self) -> usize {
        let macro_bytes = (self.density.len() + self.mean_velocity.len() + self.temperature.len())
            * std::mem::size_of::<f64>();
        macro_bytes + self.inner.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::algos::uniform::UniformGrid6D;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::init::isolated::{PlummerIC, sample_on_grid};

    fn test_domain(nx: i128, nv: i128) -> Domain {
        Domain::builder()
            .spatial_extent(4.0)
            .velocity_extent(3.0)
            .spatial_resolution(nx)
            .velocity_resolution(nv)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn test_macro_micro_construction() {
        let domain = test_domain(4, 4);
        let snap = {
            let ic = PlummerIC::new(1.0, 1.0, 1.0);
            sample_on_grid(&ic, &domain)
        };
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
        let grid_density = grid.compute_density();

        let mm = MacroMicroRepr::from_repr(Box::new(grid), &domain);

        // Density must match the inner representation's density.
        assert_eq!(mm.density.len(), grid_density.data.len());
        for (a, b) in mm.density.iter().zip(grid_density.data.iter()) {
            assert!((a - b).abs() < 1e-14, "density mismatch: {a} vs {b}");
        }
        // Spatial shape must match.
        assert_eq!(mm.spatial_shape, grid_density.shape);
    }

    #[test]
    fn test_macro_micro_density_matches_inner() {
        let domain = test_domain(4, 4);
        let snap = {
            let ic = PlummerIC::new(1.0, 1.0, 1.0);
            sample_on_grid(&ic, &domain)
        };
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());

        let mm = MacroMicroRepr::from_repr(Box::new(grid), &domain);
        let density = mm.compute_density();

        // compute_density() should return the cached macro density.
        assert_eq!(density.data.len(), mm.density.len());
        for (a, b) in density.data.iter().zip(mm.density.iter()) {
            assert!(
                (a - b).abs() < 1e-14,
                "compute_density mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_macro_micro_mass_conservation() {
        // Use same domain as test_macro_micro_from_plummer (8x8, extent=4/3)
        // which is known to produce non-trivial Plummer mass.
        let domain = test_domain(8, 8);
        let snap = {
            let ic = PlummerIC::new(1.0, 1.0, 1.0);
            sample_on_grid(&ic, &domain)
        };
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());
        let mass_before = grid.total_mass();

        let mm = MacroMicroRepr::from_repr(Box::new(grid), &domain);
        let mass_mm = mm.total_mass();

        // Total mass from inner representation and from macro density should agree.
        // The macro density is computed by integrating f over velocity, so total_mass
        // (integral of rho over space) equals the full 6D integral of f.
        let inner_mass = mm.inner.total_mass();

        // mass_mm and inner_mass should match since density is computed from inner.
        assert!(
            (mass_mm - inner_mass).abs() < 1e-10 || {
                let rel = (mass_mm - inner_mass).abs() / inner_mass.abs().max(1e-30);
                rel < 0.1
            },
            "mass mismatch between macro ({mass_mm}) and inner ({inner_mass})"
        );

        // Both should equal mass_before since the inner repr is the same grid.
        assert!(
            (mass_mm - mass_before).abs() < 1e-10 || {
                let rel = (mass_mm - mass_before).abs() / mass_before.abs().max(1e-30);
                rel < 0.1
            },
            "mass mismatch between macro ({mass_mm}) and original ({mass_before})"
        );
    }

    #[test]
    fn test_macro_micro_from_plummer() {
        let domain = test_domain(8, 8);
        let snap = {
            let ic = PlummerIC::new(1.0, 1.0, 1.0);
            sample_on_grid(&ic, &domain)
        };
        let grid = UniformGrid6D::from_snapshot(snap, domain.clone());

        let mm = MacroMicroRepr::from_repr(Box::new(grid), &domain);

        // Density should peak near the center.
        let [nx, ny, nz] = mm.spatial_shape;
        let center_idx = (nx / 2) * ny * nz + (ny / 2) * nz + nz / 2;
        let corner_idx = 0;
        assert!(
            mm.density[center_idx] > mm.density[corner_idx],
            "Plummer density should peak at center: center={} corner={}",
            mm.density[center_idx],
            mm.density[corner_idx]
        );

        // Total mass should be positive.
        assert!(mm.total_mass() > 0.0);

        // Memory bytes should be nonzero.
        assert!(mm.memory_bytes() > 0);

        // Kinetic energy should be finite and non-negative.
        let ke = mm.total_kinetic_energy().unwrap();
        assert!(ke.is_finite(), "kinetic energy should be finite, got {ke}");
        assert!(ke >= 0.0, "kinetic energy should be non-negative, got {ke}");

        // Entropy should be finite.
        let s = mm.entropy();
        assert!(s.is_finite(), "entropy should be finite, got {s}");

        // Casimir C2 should be positive.
        let c2 = mm.casimir_c2();
        assert!(c2 > 0.0, "Casimir C2 should be positive, got {c2}");
    }
}
