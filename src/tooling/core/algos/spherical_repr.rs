//! Spherical phase-space representation f(r, v_r, L).
//!
//! Reduces the full 6D Vlasov problem to an effective 3D grid for spherically
//! symmetric systems, enabling ultra-high-resolution halo studies at a fraction
//! of the memory cost. Uses coordinates:
//! - r: radial distance
//! - v_r: radial velocity
//! - L: angular momentum magnitude (conserved exactly along characteristics)
//!
//! Because L is a constant of motion in a spherical potential, the grid only
//! advects along the r and v_r dimensions. The centrifugal pseudo-force
//! L^2 / r^3 is included in the velocity kick sub-step.

use super::super::{init::domain::Domain, phasespace::PhaseSpaceRepr, types::*};
use std::any::Any;

/// Spherically symmetric phase-space representation on a (r, v_r, L) grid.
pub struct SphericalRepr {
    /// Distribution function values: f[ir * nv * nl + iv * nl + il].
    pub data: Vec<f64>,
    /// Grid sizes [nr, nv, nl].
    pub shape: [usize; 3],
    /// Radial grid: r ∈ [r_min, r_max].
    pub r_range: (f64, f64),
    /// Radial velocity grid: v_r ∈ [-v_max, v_max].
    pub v_range: (f64, f64),
    /// Angular momentum grid: L ∈ [0, L_max].
    pub l_range: (f64, f64),
    /// Cell spacings.
    pub dr: f64,
    pub dv: f64,
    pub dl: f64,
    /// Domain (for PhaseSpaceRepr trait compatibility).
    pub domain: Domain,
}

impl SphericalRepr {
    /// Create a new spherical representation with zero-initialized f.
    ///
    /// `r_min` is set to `r_max / nr` to avoid the r=0 coordinate singularity.
    pub fn new(
        domain: Domain,
        nr: usize,
        nv: usize,
        nl: usize,
        r_max: f64,
        v_max: f64,
        l_max: f64,
    ) -> Self {
        let r_min = r_max / nr as f64; // Avoid r=0 singularity
        let dr = (r_max - r_min) / nr as f64;
        let dv = 2.0 * v_max / nv as f64;
        let dl = l_max / nl as f64;

        Self {
            data: vec![0.0; nr * nv * nl],
            shape: [nr, nv, nl],
            r_range: (r_min, r_max),
            v_range: (-v_max, v_max),
            l_range: (0.0, l_max),
            dr,
            dv,
            dl,
            domain,
        }
    }

    /// Get radial coordinate for cell index.
    #[inline]
    pub fn r_at(&self, ir: usize) -> f64 {
        self.r_range.0 + (ir as f64 + 0.5) * self.dr
    }

    /// Get radial velocity for cell index.
    #[inline]
    pub fn vr_at(&self, iv: usize) -> f64 {
        self.v_range.0 + (iv as f64 + 0.5) * self.dv
    }

    /// Get angular momentum for cell index.
    #[inline]
    pub fn l_at(&self, il: usize) -> f64 {
        self.l_range.0 + (il as f64 + 0.5) * self.dl
    }

    /// Flat index into the data array for grid point (ir, iv, il).
    #[inline]
    pub fn index(&self, ir: usize, iv: usize, il: usize) -> usize {
        ir * self.shape[1] * self.shape[2] + iv * self.shape[2] + il
    }
}

impl PhaseSpaceRepr for SphericalRepr {
    fn set_progress(&mut self, _p: std::sync::Arc<super::super::progress::StepProgress>) {}

    /// Compute spherically averaged density rho(r) by integrating f over v_r and L.
    fn compute_density(&self) -> DensityField {
        // Compute spherically averaged density rho(r) = integral f dv_r dL * 4*pi*r^2
        let [nr, nv, nl] = self.shape;
        // Map to a 3D density field with shape [nr, 1, 1] for compatibility
        let mut rho = vec![0.0f64; nr];
        for (ir, rho_val) in rho.iter_mut().enumerate() {
            let r = self.r_at(ir);
            let r2 = r * r;
            let mut sum = 0.0;
            for iv in 0..nv {
                for il in 0..nl {
                    sum += self.data[self.index(ir, iv, il)];
                }
            }
            *rho_val = sum * self.dv * self.dl * 4.0 * std::f64::consts::PI * r2;
        }

        DensityField {
            data: rho,
            shape: [nr, 1, 1],
        }
    }

    /// Radial drift sub-step: semi-Lagrangian shift along r with dr/dt = v_r.
    fn advect_x(&mut self, _displacement: &DisplacementField, dt: f64) {
        // Radial advection: dr/dt = v_r
        let [nr, nv, nl] = self.shape;
        let src = self.data.clone();

        for iv in 0..nv {
            let vr = self.vr_at(iv);
            let shift = vr * dt / self.dr; // Shift in cell units

            for il in 0..nl {
                // Semi-Lagrangian shift along r for this (v_r, L) slice
                let line: Vec<f64> = (0..nr).map(|ir| src[self.index(ir, iv, il)]).collect();

                let mut shifted = vec![0.0f64; nr];
                // Simple linear interpolation for radial shift
                for (ir, shifted_val) in shifted.iter_mut().enumerate() {
                    let dep = ir as f64 - shift;
                    let i0 = dep.floor() as isize;
                    let t = dep - dep.floor();
                    let clamp = |j: isize| j.clamp(0, nr as isize - 1) as usize;
                    *shifted_val = (1.0 - t) * line[clamp(i0)] + t * line[clamp(i0 + 1)];
                }

                for (ir, &shifted_val) in shifted.iter().enumerate() {
                    let idx = ir * nv * nl + iv * nl + il;
                    self.data[idx] = shifted_val;
                }
            }
        }
    }

    /// Velocity kick sub-step: semi-Lagrangian shift along v_r with
    /// dv_r/dt = -dPhi/dr + L^2/r^3 (gravitational + centrifugal acceleration).
    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        // Velocity kick: dv_r/dt = -dPhi/dr + L^2/(r^3)
        let [nr, nv, nl] = self.shape;
        let src = self.data.clone();

        for ir in 0..nr {
            let r = self.r_at(ir);
            let gr = if ir < acceleration.gx.len() {
                acceleration.gx[ir]
            } else {
                0.0
            };

            for il in 0..nl {
                let l_ang = self.l_at(il);
                // Centrifugal acceleration
                let a_cent = if r > 1e-30 {
                    l_ang * l_ang / (r * r * r)
                } else {
                    0.0
                };
                let total_accel = gr + a_cent;
                let shift = total_accel * dt / self.dv;

                // Semi-Lagrangian shift along v_r
                let line: Vec<f64> = (0..nv).map(|iv| src[self.index(ir, iv, il)]).collect();

                for iv in 0..nv {
                    let dep = iv as f64 - shift;
                    let i0 = dep.floor() as isize;
                    let t = dep - dep.floor();
                    let clamp = |j: isize| j.clamp(0, nv as isize - 1) as usize;
                    let idx = ir * nv * nl + iv * nl + il;
                    self.data[idx] = (1.0 - t) * line[clamp(i0)] + t * line[clamp(i0 + 1)];
                }
            }
        }
    }

    /// Velocity moment (stub -- returns an empty tensor).
    fn moment(&self, _position: &[f64; 3], _order: usize) -> Tensor {
        Tensor {
            data: vec![],
            rank: 0,
            shape: vec![],
        }
    }

    /// Total mass by integrating f over the (r, v_r, L) grid with 4*pi*r^2 Jacobian.
    fn total_mass(&self) -> f64 {
        let [nr, nv, nl] = self.shape;
        let mut mass = 0.0;
        for ir in 0..nr {
            let r = self.r_at(ir);
            for iv in 0..nv {
                for il in 0..nl {
                    mass += self.data[self.index(ir, iv, il)]
                        * 4.0
                        * std::f64::consts::PI
                        * r
                        * r
                        * self.dr
                        * self.dv
                        * self.dl;
                }
            }
        }
        mass
    }

    /// Casimir C2 = integral of f^2 over the reduced phase space.
    fn casimir_c2(&self) -> f64 {
        let [nr, nv, nl] = self.shape;
        let dphase = self.dr * self.dv * self.dl;
        let mut c2 = 0.0;
        for ir in 0..nr {
            let r = self.r_at(ir);
            let r2 = r * r;
            for iv in 0..nv {
                for il in 0..nl {
                    let f = self.data[self.index(ir, iv, il)];
                    c2 += f * f * 4.0 * std::f64::consts::PI * r2 * dphase;
                }
            }
        }
        c2
    }

    /// Entropy S = -integral of f ln f over the reduced phase space.
    fn entropy(&self) -> f64 {
        let [nr, nv, nl] = self.shape;
        let dphase = self.dr * self.dv * self.dl;
        let mut s = 0.0;
        for ir in 0..nr {
            let r = self.r_at(ir);
            let r2 = r * r;
            for iv in 0..nv {
                for il in 0..nl {
                    let f = self.data[self.index(ir, iv, il)];
                    if f > 0.0 {
                        s += -f * f.ln() * 4.0 * std::f64::consts::PI * r2 * dphase;
                    }
                }
            }
        }
        s
    }

    /// Stream count (stub -- returns zeros; not applicable in spherical coords).
    fn stream_count(&self) -> StreamCountField {
        StreamCountField {
            data: vec![0; self.shape[0]],
            shape: [self.shape[0], 1, 1],
        }
    }

    /// Local velocity distribution (stub -- returns empty; use radial profiles instead).
    fn velocity_distribution(&self, _position: &[f64; 3]) -> Vec<f64> {
        vec![]
    }

    /// Total kinetic energy T = 0.5 * integral of f*(v_r^2 + L^2/r^2) over phase space.
    fn total_kinetic_energy(&self) -> Option<f64> {
        let [nr, nv, nl] = self.shape;
        let dphase = self.dr * self.dv * self.dl;
        let mut t = 0.0;
        for ir in 0..nr {
            let r = self.r_at(ir);
            let r2 = r * r;
            for iv in 0..nv {
                let vr = self.vr_at(iv);
                for il in 0..nl {
                    let l_ang = self.l_at(il);
                    let f = self.data[self.index(ir, iv, il)];
                    let v2 = vr * vr
                        + if r > 1e-30 {
                            l_ang * l_ang / (r * r)
                        } else {
                            0.0
                        };
                    t += 0.5 * f * v2 * 4.0 * std::f64::consts::PI * r2 * dphase;
                }
            }
        }
        Some(t)
    }

    /// Serialize the 3D grid into a `PhaseSpaceSnapshot` for checkpointing.
    fn to_snapshot(&self, time: f64) -> Option<PhaseSpaceSnapshot> {
        Some(PhaseSpaceSnapshot {
            data: self.data.clone(),
            shape: [self.shape[0], self.shape[1], self.shape[2], 1, 1, 1],
            time,
        })
    }

    /// Restore the grid data from a previously saved snapshot.
    fn load_snapshot(&mut self, snap: PhaseSpaceSnapshot) -> Result<(), crate::CausticError> {
        self.data = snap.data;
        Ok(())
    }

    /// Downcast to `&dyn Any` for runtime type queries.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Downcast to `&mut dyn Any` for runtime type queries.
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    /// Heap memory used by the distribution function data array.
    fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
    }
}
