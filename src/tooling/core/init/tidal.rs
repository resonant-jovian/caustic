//! Tidal stream initial conditions: progenitor cluster orbiting in a fixed host potential.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use super::isolated::IsolatedEquilibrium;

/// Tidal stream IC: progenitor cluster on an orbit in an external host potential.
pub struct TidalIC {
    /// Fixed external host potential Φ_host(x). Does not evolve self-consistently.
    pub host_potential: Box<dyn Fn([f64; 3]) -> f64 + Send + Sync>,
    pub progenitor: Box<dyn IsolatedEquilibrium>,
    pub progenitor_position: [f64; 3],
    pub progenitor_velocity: [f64; 3],
}

impl TidalIC {
    pub fn new(
        host_potential: Box<dyn Fn([f64; 3]) -> f64 + Send + Sync>,
        progenitor: Box<dyn IsolatedEquilibrium>,
        progenitor_position: [f64; 3],
        progenitor_velocity: [f64; 3],
    ) -> Self {
        Self {
            host_potential,
            progenitor,
            progenitor_position,
            progenitor_velocity,
        }
    }

    /// Sample progenitor f centred on (progenitor_position, progenitor_velocity).
    ///
    /// At each (x, v) grid point: compute position and velocity relative to the
    /// progenitor centre, evaluate energy in progenitor's rest frame, then use the
    /// progenitor's distribution function.
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

        let pos = self.progenitor_position;
        let vel = self.progenitor_velocity;

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

        for ix1 in 0..nx1 {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            for ix2 in 0..nx2 {
                let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];
                for ix3 in 0..nx3 {
                    let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                    let base = ix1 * s_x1 + ix2 * s_x2 + ix3 * s_x3;

                    // Position relative to progenitor centre
                    let x_rel = [x1 - pos[0], x2 - pos[1], x3 - pos[2]];
                    let r =
                        (x_rel[0] * x_rel[0] + x_rel[1] * x_rel[1] + x_rel[2] * x_rel[2]).sqrt();
                    let phi = self.progenitor.potential(r);

                    for iv1 in 0..nv1 {
                        let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];

                                // Velocity relative to progenitor
                                let dv1 = v1 - vel[0];
                                let dv2 = v2 - vel[1];
                                let dv3 = v3 - vel[2];
                                let v2sq = dv1 * dv1 + dv2 * dv2 + dv3 * dv3;
                                let energy = 0.5 * v2sq + phi;
                                let f = self.progenitor.distribution_function(energy, 0.0).max(0.0);
                                data[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f;
                            }
                        }
                    }
                }
            }

            if let Some(p) = progress {
                let c = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c % report_interval == 0 {
                    p.set_intra_progress(c, nx1 as u64);
                }
            }
        }

        PhaseSpaceSnapshot {
            data,
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time: 0.0,
        }
    }

    /// Compute escape velocity from host potential at given galactocentric radius r.
    /// Particles with velocity > v_esc at radius r are tidal debris.
    pub fn escape_velocity(&self, r: f64) -> f64 {
        let phi = (self.host_potential)([r, 0.0, 0.0]);
        (2.0 * phi.abs()).sqrt()
    }
}
