//! Method of lines (unsplit) integrator. Treats Vlasov as a 6D PDE without operator
//! splitting. Evaluates the RHS of the Vlasov equation directly using finite differences
//! and advances with a classic Runge-Kutta scheme (RK2, RK3, or RK4).
//!
//! The Vlasov equation:
//!   df/dt = -v . grad_x f  +  g(x) . grad_v f
//!
//! where g(x) = -grad Phi is the gravitational acceleration.

use std::sync::Arc;

use rayon::prelude::*;

use super::super::{
    advecator::Advector,
    init::domain::Domain,
    integrator::{StepProducts, TimeIntegrator},
    phasespace::PhaseSpaceRepr,
    progress::{StepPhase, StepProgress},
    solver::PoissonSolver,
    types::*,
};

/// Method-of-lines Runge-Kutta integrator for the full 6D Vlasov PDE.
///
/// Unlike split integrators (Strang, Yoshida), this evaluates the complete
/// RHS = -v . grad_x f + g . grad_v f at each stage, re-solving Poisson
/// for the intermediate state. This avoids splitting error but is more
/// expensive per timestep and subject to CFL constraints.
pub struct UnsplitIntegrator {
    /// Number of Runge-Kutta stages (2, 3, or 4).
    pub rk_stages: usize,
    /// Gravitational constant G.
    pub g: f64,
    /// Computational domain (grid spacings, extents, BCs).
    pub domain: Domain,
    progress: Option<Arc<StepProgress>>,
}

impl UnsplitIntegrator {
    /// Create a new unsplit integrator.
    ///
    /// # Panics
    /// Panics if `rk_stages` is not 2, 3, or 4.
    pub fn new(rk_stages: usize, g: f64, domain: Domain) -> Self {
        assert!(
            rk_stages == 2 || rk_stages == 3 || rk_stages == 4,
            "rk_stages must be 2, 3, or 4; got {}",
            rk_stages
        );
        Self {
            rk_stages,
            g,
            domain,
            progress: None,
        }
    }

    /// Grid dimensions as `[nx1, nx2, nx3, nv1, nv2, nv3]`.
    fn sizes(&self) -> [usize; 6] {
        [
            self.domain.spatial_res.x1 as usize,
            self.domain.spatial_res.x2 as usize,
            self.domain.spatial_res.x3 as usize,
            self.domain.velocity_res.v1 as usize,
            self.domain.velocity_res.v2 as usize,
            self.domain.velocity_res.v3 as usize,
        ]
    }

    /// Stride array for row-major 6D indexing: ix1*s[0] + ix2*s[1] + ... + iv3*s[5].
    fn strides(&self) -> [usize; 6] {
        let [_nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;
        [s_x1, s_x2, s_x3, s_v1, s_v2, s_v3]
    }

    /// Velocity extents [-Lv, Lv] for each dimension.
    fn lv(&self) -> [f64; 3] {
        self.domain.lv()
    }

    /// Evaluate the RHS of the Vlasov equation:
    ///   RHS = -(v1 df/dx1 + v2 df/dx2 + v3 df/dx3) + (gx df/dv1 + gy df/dv2 + gz df/dv3)
    ///
    /// Uses 1st-order upwind finite differences:
    /// - If advection velocity > 0: backward difference (f[i] - f[i-1]) / h
    /// - If advection velocity < 0: forward difference  (f[i+1] - f[i]) / h
    ///
    /// Spatial BCs: periodic (wrap with rem_euclid).
    /// Velocity BCs: zero outside domain (clamp to 0 for out-of-bounds neighbors).
    fn evaluate_rhs(&self, data: &[f64], accel: &AccelerationField) -> Vec<f64> {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let strides = self.strides();
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lv = self.lv();
        let n_total = data.len();

        let use_periodic_spatial = matches!(
            self.domain.spatial_bc,
            super::super::init::domain::SpatialBoundType::Periodic
        );

        let mut rhs = vec![0.0f64; n_total];

        // Parallelize over ix1 slabs. Each slab is independent: reads from
        // the shared `data` array, writes to its own disjoint portion of `rhs`.
        let slab_stride = strides[0];
        rhs.par_chunks_mut(slab_stride)
            .enumerate()
            .for_each(|(ix1, rhs_slab)| {
                for ix2 in 0..nx2 {
                    for ix3 in 0..nx3 {
                        let si = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                        let gx = accel.gx[si];
                        let gy = accel.gy[si];
                        let gz = accel.gz[si];

                        for iv1 in 0..nv1 {
                            for iv2 in 0..nv2 {
                                for iv3 in 0..nv3 {
                                    // Global index for reading from `data`
                                    let idx = ix1 * strides[0]
                                        + ix2 * strides[1]
                                        + ix3 * strides[2]
                                        + iv1 * strides[3]
                                        + iv2 * strides[4]
                                        + iv3 * strides[5];
                                    // Local index within this ix1-slab for writing
                                    let local = idx - ix1 * slab_stride;

                                    let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                                    let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                                    let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];

                                    let df_dx1 = if use_periodic_spatial {
                                        upwind_periodic(data, idx, strides[0], ix1, nx1, dx[0], v1)
                                    } else {
                                        upwind_open(data, idx, strides[0], ix1, nx1, dx[0], v1)
                                    };
                                    let df_dx2 = if use_periodic_spatial {
                                        upwind_periodic(data, idx, strides[1], ix2, nx2, dx[1], v2)
                                    } else {
                                        upwind_open(data, idx, strides[1], ix2, nx2, dx[1], v2)
                                    };
                                    let df_dx3 = if use_periodic_spatial {
                                        upwind_periodic(data, idx, strides[2], ix3, nx3, dx[2], v3)
                                    } else {
                                        upwind_open(data, idx, strides[2], ix3, nx3, dx[2], v3)
                                    };
                                    let df_dv1 =
                                        upwind_zero_bc(data, idx, strides[3], iv1, nv1, dv[0], gx);
                                    let df_dv2 =
                                        upwind_zero_bc(data, idx, strides[4], iv2, nv2, dv[1], gy);
                                    let df_dv3 =
                                        upwind_zero_bc(data, idx, strides[5], iv3, nv3, dv[2], gz);

                                    rhs_slab[local] = -(v1 * df_dx1 + v2 * df_dx2 + v3 * df_dx3)
                                        + (gx * df_dv1 + gy * df_dv2 + gz * df_dv3);
                                }
                            }
                        }
                    }
                }
            });

        rhs
    }
}

/// 1st-order upwind finite difference with periodic boundary conditions.
///
/// `idx` is the flat index of the current point. `stride` is the stride along the
/// dimension of interest. `pos` is the grid index along that dimension, `n` is the
/// grid size. `h` is the cell spacing. `vel` is the advection velocity that determines
/// the upwind direction.
#[inline]
fn upwind_periodic(
    data: &[f64],
    idx: usize,
    stride: usize,
    pos: usize,
    n: usize,
    h: f64,
    vel: f64,
) -> f64 {
    let f_here = data[idx];
    if vel >= 0.0 {
        // Backward difference: (f[i] - f[i-1]) / h
        let prev_pos = ((pos as isize - 1).rem_euclid(n as isize)) as usize;
        // Compute neighbor index by subtracting current position contribution and adding prev
        let neighbor_idx = idx - pos * stride + prev_pos * stride;
        (f_here - data[neighbor_idx]) / h
    } else {
        // Forward difference: (f[i+1] - f[i]) / h
        let next_pos = (pos + 1) % n;
        let neighbor_idx = idx - pos * stride + next_pos * stride;
        (data[neighbor_idx] - f_here) / h
    }
}

/// 1st-order upwind finite difference with open/absorbing boundary conditions.
///
/// At boundaries, uses one-sided differences. Out-of-domain values are treated as 0.
#[inline]
fn upwind_open(
    data: &[f64],
    idx: usize,
    stride: usize,
    pos: usize,
    n: usize,
    h: f64,
    vel: f64,
) -> f64 {
    let f_here = data[idx];
    if vel >= 0.0 {
        // Backward difference
        if pos == 0 {
            // No left neighbor: assume f[-1] = 0
            f_here / h
        } else {
            let neighbor_idx = idx - stride;
            (f_here - data[neighbor_idx]) / h
        }
    } else {
        // Forward difference
        if pos == n - 1 {
            // No right neighbor: assume f[n] = 0
            -f_here / h
        } else {
            let neighbor_idx = idx + stride;
            (data[neighbor_idx] - f_here) / h
        }
    }
}

/// 1st-order upwind finite difference with zero (absorbing) velocity boundary conditions.
///
/// At velocity boundaries, the out-of-domain neighbor is taken as 0.
#[inline]
fn upwind_zero_bc(
    data: &[f64],
    idx: usize,
    stride: usize,
    pos: usize,
    n: usize,
    h: f64,
    vel: f64,
) -> f64 {
    let f_here = data[idx];
    if vel >= 0.0 {
        // Backward difference
        if pos == 0 {
            f_here / h
        } else {
            let neighbor_idx = idx - stride;
            (f_here - data[neighbor_idx]) / h
        }
    } else {
        // Forward difference
        if pos == n - 1 {
            -f_here / h
        } else {
            let neighbor_idx = idx + stride;
            (data[neighbor_idx] - f_here) / h
        }
    }
}

/// Integrate f over velocity dimensions to produce a density field:
///   rho(ix1, ix2, ix3) = sum_{iv1, iv2, iv3} f[...] * dv1 * dv2 * dv3
fn compute_density_from_data(data: &[f64], shape: [usize; 6], dv: [f64; 3]) -> DensityField {
    let [nx1, nx2, nx3, nv1, nv2, nv3] = shape;
    let dv3 = dv[0] * dv[1] * dv[2];

    let s_v3 = 1usize;
    let s_v2 = nv3;
    let s_v1 = nv2 * nv3;
    let s_x3 = nv1 * s_v1;
    let s_x2 = nx3 * s_x3;
    let s_x1 = nx2 * s_x2;

    let n_spatial = nx1 * nx2 * nx3;
    let mut rho = vec![0.0f64; n_spatial];

    for ix1 in 0..nx1 {
        for ix2 in 0..nx2 {
            for ix3 in 0..nx3 {
                let si = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                let base = ix1 * s_x1 + ix2 * s_x2 + ix3 * s_x3;
                let mut sum = 0.0f64;
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            sum += data[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3];
                        }
                    }
                }
                rho[si] = sum * dv3;
            }
        }
    }

    DensityField {
        data: rho,
        shape: [nx1, nx2, nx3],
    }
}

impl TimeIntegrator for UnsplitIntegrator {
    fn advance(
        &mut self,
        repr: &mut dyn PhaseSpaceRepr,
        solver: &dyn PoissonSolver,
        advector: &dyn Advector,
        dt: f64,
    ) -> StepProducts {
        let _span = tracing::info_span!("unsplit_advance").entered();

        if let Some(ref p) = self.progress {
            p.start_step();
            p.set_phase(StepPhase::UnsplitStage1);
            p.set_sub_step(0, self.rk_stages as u8);
        }

        let shape = self.sizes();
        let dv = self.domain.dv();
        let g = self.g;

        // Extract current state as flat data
        let snap0 = repr.to_snapshot(0.0);
        let y0 = snap0.data;
        let n = y0.len();

        // Helper: compute acceleration from raw 6D data
        let compute_accel = |data: &[f64]| -> AccelerationField {
            let density = compute_density_from_data(data, shape, dv);
            let potential = solver.solve(&density, g);
            solver.compute_acceleration(&potential)
        };

        match self.rk_stages {
            2 => {
                // RK2 (Heun's method):
                //   k1 = RHS(y_n)
                //   k2 = RHS(y_n + dt*k1)
                //   y_{n+1} = y_n + dt/2 * (k1 + k2)

                let accel0 = compute_accel(&y0);
                let k1 = self.evaluate_rhs(&y0, &accel0);

                // Reusable stage buffer (allocated once, reused for all stages)
                let mut y_stage = vec![0.0f64; n];
                for i in 0..n {
                    y_stage[i] = y0[i] + dt * k1[i];
                }

                if let Some(ref p) = self.progress {
                    p.set_phase(StepPhase::UnsplitStage2);
                    p.set_sub_step(1, 2);
                }
                let accel1 = compute_accel(&y_stage);
                let k2 = self.evaluate_rhs(&y_stage, &accel1);

                // y_{n+1} = y_n + dt/2 * (k1 + k2) — reuse y_stage for final output
                for i in 0..n {
                    y_stage[i] = y0[i] + 0.5 * dt * (k1[i] + k2[i]);
                }

                repr.load_snapshot(PhaseSpaceSnapshot {
                    data: y_stage,
                    shape: snap0.shape,
                    time: snap0.time + dt,
                });
            }

            3 => {
                // RK3 (Kutta's third-order method):
                //   k1 = RHS(y_n)
                //   k2 = RHS(y_n + dt/2 * k1)
                //   k3 = RHS(y_n - dt*k1 + 2*dt*k2)
                //   y_{n+1} = y_n + dt/6 * (k1 + 4*k2 + k3)

                let accel0 = compute_accel(&y0);
                let k1 = self.evaluate_rhs(&y0, &accel0);

                // Single reusable stage buffer for all intermediate states
                let mut y_stage = vec![0.0f64; n];

                // Stage 2: y0 + dt/2 * k1
                if let Some(ref p) = self.progress {
                    p.set_phase(StepPhase::UnsplitStage2);
                    p.set_sub_step(1, 3);
                }
                for i in 0..n {
                    y_stage[i] = y0[i] + 0.5 * dt * k1[i];
                }
                let accel2 = compute_accel(&y_stage);
                let k2 = self.evaluate_rhs(&y_stage, &accel2);

                // Stage 3: y0 - dt*k1 + 2*dt*k2 (reuse y_stage)
                if let Some(ref p) = self.progress {
                    p.set_phase(StepPhase::UnsplitStage3);
                    p.set_sub_step(2, 3);
                }
                for i in 0..n {
                    y_stage[i] = y0[i] - dt * k1[i] + 2.0 * dt * k2[i];
                }
                let accel3 = compute_accel(&y_stage);
                let k3 = self.evaluate_rhs(&y_stage, &accel3);

                // y_{n+1} = y_n + dt/6 * (k1 + 4*k2 + k3) — reuse y_stage
                for i in 0..n {
                    y_stage[i] = y0[i] + dt / 6.0 * (k1[i] + 4.0 * k2[i] + k3[i]);
                }

                repr.load_snapshot(PhaseSpaceSnapshot {
                    data: y_stage,
                    shape: snap0.shape,
                    time: snap0.time + dt,
                });
            }

            4 => {
                // Classic RK4:
                //   k1 = RHS(y_n)
                //   k2 = RHS(y_n + dt/2 * k1)
                //   k3 = RHS(y_n + dt/2 * k2)
                //   k4 = RHS(y_n + dt * k3)
                //   y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

                let accel0 = compute_accel(&y0);
                let k1 = self.evaluate_rhs(&y0, &accel0);

                // Single reusable stage buffer for all intermediate states
                let mut y_stage = vec![0.0f64; n];

                // Stage 2: y0 + dt/2 * k1
                if let Some(ref p) = self.progress {
                    p.set_phase(StepPhase::UnsplitStage2);
                    p.set_sub_step(1, 4);
                }
                for i in 0..n {
                    y_stage[i] = y0[i] + 0.5 * dt * k1[i];
                }
                let accel2 = compute_accel(&y_stage);
                let k2 = self.evaluate_rhs(&y_stage, &accel2);

                // Stage 3: y0 + dt/2 * k2 (reuse y_stage)
                if let Some(ref p) = self.progress {
                    p.set_phase(StepPhase::UnsplitStage3);
                    p.set_sub_step(2, 4);
                }
                for i in 0..n {
                    y_stage[i] = y0[i] + 0.5 * dt * k2[i];
                }
                let accel3 = compute_accel(&y_stage);
                let k3 = self.evaluate_rhs(&y_stage, &accel3);

                // Stage 4: y0 + dt * k3 (reuse y_stage)
                if let Some(ref p) = self.progress {
                    p.set_phase(StepPhase::UnsplitStage4);
                    p.set_sub_step(3, 4);
                }
                for i in 0..n {
                    y_stage[i] = y0[i] + dt * k3[i];
                }
                let accel4 = compute_accel(&y_stage);
                let k4 = self.evaluate_rhs(&y_stage, &accel4);

                // y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4) — reuse y_stage
                for i in 0..n {
                    y_stage[i] =
                        y0[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                }

                repr.load_snapshot(PhaseSpaceSnapshot {
                    data: y_stage,
                    shape: snap0.shape,
                    time: snap0.time + dt,
                });
            }

            _ => unreachable!("rk_stages validated in constructor"),
        }

        let density = repr.compute_density();
        let potential = solver.solve(&density, g);
        let acceleration = solver.compute_acceleration(&potential);
        StepProducts { density, potential, acceleration }
    }

    /// CFL condition for the unsplit integrator.
    ///
    /// dt <= cfl_factor * min(min_dx / v_max, min_dv / g_max)
    ///
    /// - v_max is the maximum velocity from the domain velocity extent.
    /// - g_max is estimated from the dynamical time: g_max ~ sqrt(G * rho_max) * L_box,
    ///   or equivalently the inverse dynamical time 1/sqrt(G * rho_max).
    fn max_dt(&self, repr: &dyn PhaseSpaceRepr, cfl_factor: f64) -> f64 {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lv = self.lv();

        // v_max: maximum velocity magnitude from domain extent
        let v_max = lv[0].max(lv[1]).max(lv[2]);

        // Minimum spatial and velocity cell sizes
        let dx_min = dx[0].min(dx[1]).min(dx[2]);
        let dv_min = dv[0].min(dv[1]).min(dv[2]);

        // Spatial CFL: dt <= dx_min / v_max
        let dt_spatial = if v_max > 0.0 { dx_min / v_max } else { 1e10 };

        // Velocity CFL: dt <= dv_min / g_max
        // Estimate g_max from density
        let density = repr.compute_density();
        let rho_max = density.data.iter().cloned().fold(0.0_f64, f64::max);

        let dt_velocity = if rho_max > 0.0 && self.g > 0.0 {
            // g_max ~ sqrt(G * rho_max) * L where L is the box size
            let lx = self.domain.lx();
            let l_box = 2.0 * lx[0].max(lx[1]).max(lx[2]);
            let g_max = (self.g * rho_max).sqrt() * l_box;
            if g_max > 0.0 { dv_min / g_max } else { 1e10 }
        } else {
            1e10
        };

        cfl_factor * dt_spatial.min(dt_velocity)
    }

    fn set_progress(&mut self, progress: Arc<StepProgress>) {
        self.progress = Some(progress);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsplit_free_streaming() {
        // G=0, so no acceleration. Vlasov reduces to df/dt + v . grad f = 0.
        // A Gaussian blob should shift by v*dt in position.
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::poisson::fft::FftPoisson;

        let domain = Domain::builder()
            .spatial_extent(4.0)
            .velocity_extent(2.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        // Build a Gaussian blob IC manually: f(x,v) = exp(-|x|^2/2) * exp(-|v|^2/2)
        let [nx1, nx2, nx3] = [8usize; 3];
        let [nv1, nv2, nv3] = [8usize; 3];
        let dx = domain.dx();
        let dv = domain.dv();
        let lx = 4.0_f64;
        let lv = 2.0_f64;
        let n_total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
        let mut data = vec![0.0f64; n_total];

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        for ix1 in 0..nx1 {
            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    let x1 = -lx + (ix1 as f64 + 0.5) * dx[0];
                    let x2 = -lx + (ix2 as f64 + 0.5) * dx[1];
                    let x3 = -lx + (ix3 as f64 + 0.5) * dx[2];
                    let r2_x = x1 * x1 + x2 * x2 + x3 * x3;
                    for iv1 in 0..nv1 {
                        for iv2 in 0..nv2 {
                            for iv3 in 0..nv3 {
                                let v1 = -lv + (iv1 as f64 + 0.5) * dv[0];
                                let v2 = -lv + (iv2 as f64 + 0.5) * dv[1];
                                let v3 = -lv + (iv3 as f64 + 0.5) * dv[2];
                                let r2_v = v1 * v1 + v2 * v2 + v3 * v3;
                                let idx = ix1 * s_x1
                                    + ix2 * s_x2
                                    + ix3 * s_x3
                                    + iv1 * s_v1
                                    + iv2 * s_v2
                                    + iv3 * s_v3;
                                data[idx] = (-0.5 * r2_x).exp() * (-0.5 * r2_v).exp();
                            }
                        }
                    }
                }
            }
        }

        let snap = PhaseSpaceSnapshot {
            data,
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time: 0.0,
        };

        // Run with unsplit RK4, G=0
        let mut unsplit = UnsplitIntegrator::new(4, 0.0, domain.clone());
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();

        let mut repr: Box<dyn PhaseSpaceRepr> =
            Box::new(UniformGrid6D::from_snapshot(snap, domain));

        let dt = 0.01;
        unsplit.advance(&mut *repr, &poisson, &advector, dt);

        let result = repr.to_snapshot(dt);
        // Basic sanity: result should be finite and have positive mass
        assert!(
            result.data.iter().all(|v| v.is_finite()),
            "result has non-finite values"
        );
        let mass: f64 = result.data.iter().sum::<f64>();
        assert!(mass > 0.0, "total mass should be positive");
    }

    #[test]
    #[should_panic(expected = "rk_stages must be 2, 3, or 4")]
    fn unsplit_invalid_stages() {
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        let domain = Domain::builder()
            .spatial_extent(1.0)
            .velocity_extent(1.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();
        let _ = UnsplitIntegrator::new(5, 1.0, domain);
    }

    #[test]
    fn unsplit_rk2_runs() {
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::init::isolated::{PlummerIC, sample_on_grid};
        use crate::tooling::core::poisson::fft::FftPoisson;

        let domain = Domain::builder()
            .spatial_extent(4.0)
            .velocity_extent(2.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 0.0);
        let snap = sample_on_grid(&ic, &domain);

        let mut unsplit = UnsplitIntegrator::new(2, 0.0, domain.clone());
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut repr: Box<dyn PhaseSpaceRepr> =
            Box::new(UniformGrid6D::from_snapshot(snap, domain));

        unsplit.advance(&mut *repr, &poisson, &advector, 0.01);
        let result = repr.to_snapshot(0.01);
        assert!(result.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn unsplit_rk3_runs() {
        use crate::tooling::core::algos::lagrangian::SemiLagrangian;
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::init::isolated::{PlummerIC, sample_on_grid};
        use crate::tooling::core::poisson::fft::FftPoisson;

        let domain = Domain::builder()
            .spatial_extent(4.0)
            .velocity_extent(2.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 0.0);
        let snap = sample_on_grid(&ic, &domain);

        let mut unsplit = UnsplitIntegrator::new(3, 0.0, domain.clone());
        let poisson = FftPoisson::new(&domain);
        let advector = SemiLagrangian::new();
        let mut repr: Box<dyn PhaseSpaceRepr> =
            Box::new(UniformGrid6D::from_snapshot(snap, domain));

        unsplit.advance(&mut *repr, &poisson, &advector, 0.01);
        let result = repr.to_snapshot(0.01);
        assert!(result.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn unsplit_max_dt_no_gravity() {
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::init::isolated::{PlummerIC, sample_on_grid};

        let domain = Domain::builder()
            .spatial_extent(4.0)
            .velocity_extent(2.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 0.0);
        let snap = sample_on_grid(&ic, &domain);
        let repr = UniformGrid6D::from_snapshot(snap, domain.clone());

        let unsplit = UnsplitIntegrator::new(4, 0.0, domain.clone());
        let dt = unsplit.max_dt(&repr, 0.5);
        // With G=0, velocity CFL is infinite, so dt is limited by spatial CFL only
        let dx_min = domain.dx()[0].min(domain.dx()[1]).min(domain.dx()[2]);
        let v_max = 2.0; // velocity extent
        let expected = 0.5 * dx_min / v_max;
        assert!(
            (dt - expected).abs() < 1e-12,
            "max_dt should be cfl * dx_min / v_max = {}, got {}",
            expected,
            dt
        );
    }

    #[test]
    fn compute_density_from_data_matches_repr() {
        use crate::tooling::core::algos::uniform::UniformGrid6D;
        use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
        use crate::tooling::core::init::isolated::{PlummerIC, sample_on_grid};

        let domain = Domain::builder()
            .spatial_extent(4.0)
            .velocity_extent(2.0)
            .spatial_resolution(4)
            .velocity_resolution(4)
            .t_final(0.1)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Truncated)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 1.0);

        // Generate two identical snapshots (sample_on_grid is deterministic)
        let snap_for_repr = sample_on_grid(&ic, &domain);
        let snap_for_helper = sample_on_grid(&ic, &domain);
        let repr = UniformGrid6D::from_snapshot(snap_for_repr, domain.clone());

        let rho_repr = repr.compute_density();
        let rho_helper =
            compute_density_from_data(&snap_for_helper.data, [4, 4, 4, 4, 4, 4], domain.dv());

        assert_eq!(rho_repr.data.len(), rho_helper.data.len());
        for (a, b) in rho_repr.data.iter().zip(rho_helper.data.iter()) {
            assert!(
                (a - b).abs() < 1e-14,
                "density mismatch: repr={}, helper={}",
                a,
                b
            );
        }
    }
}
