//! Range-separated Poisson solver.
//!
//! Splits the gravitational potential into short-range (direct) and
//! long-range (FFT-based) components for improved accuracy near the singularity.
//!
//! The Green's function G(r) = -1/(4 pi r) is decomposed as:
//!   G_short(r) = G(r) * eta(r / r_s)          -- compactly supported near-field
//!   G_long(r)  = G(r) * (1 - eta(r / r_s))    -- smooth far-field
//!
//! where eta(x) = exp(-x^2) is a Gaussian cutoff.
//!
//! The long-range part is delegated to an inner `PoissonSolver` (typically
//! `FftPoisson`). The short-range correction is applied via direct summation
//! within a stencil of radius `r_s`.

use super::super::context::SimContext;
use super::super::solver::PoissonSolver;
use super::super::types::{AccelerationField, DensityField, PotentialField};
use super::utils::finite_difference_acceleration;
use rayon::prelude::*;

/// Range-separated Poisson solver.
///
/// Decomposes the Green's function G(r) = -1/(4 pi r) into a short-range
/// component (direct summation within radius `r_s`) and a long-range
/// component (solved by an inner [`PoissonSolver`], typically FFT-based).
pub struct RangeSeparatedPoisson {
    /// Split radius in physical (not grid) units.
    split_radius: f64,
    /// Grid spacing [dx, dy, dz].
    dx: [f64; 3],
    /// Grid shape [nx, ny, nz].
    shape: [usize; 3],
    /// Inner solver for the long-range component.
    inner: Box<dyn PoissonSolver + Send + Sync>,
}

impl RangeSeparatedPoisson {
    /// Create a new range-separated Poisson solver.
    ///
    /// # Arguments
    /// * `domain` -- simulation domain (provides grid spacing and resolution)
    /// * `split_radius` -- the splitting radius r_s in physical units.
    ///   Cells within distance r_s receive a direct short-range correction.
    /// * `inner` -- the solver used for the smooth long-range component
    ///   (e.g. `FftPoisson`).
    pub fn new(
        domain: &super::super::init::domain::Domain,
        split_radius: f64,
        inner: Box<dyn PoissonSolver + Send + Sync>,
    ) -> Self {
        let shape = [
            domain.spatial_res.x1 as usize,
            domain.spatial_res.x2 as usize,
            domain.spatial_res.x3 as usize,
        ];
        Self {
            split_radius,
            dx: domain.dx(),
            shape,
            inner,
        }
    }
}

impl PoissonSolver for RangeSeparatedPoisson {
    /// Solve for the gravitational potential by combining long-range (inner solver)
    /// and short-range (direct summation within `r_s`) contributions.
    fn solve(&self, density: &DensityField, ctx: &SimContext) -> PotentialField {
        let g = ctx.g;
        let [nx, ny, nz] = self.shape;

        // 1. Long-range: solve with inner solver (gets the smooth far-field part).
        let long_range = self.inner.solve(density, ctx);

        // 2. Short-range: direct summation within split_radius.
        //
        // The inner solver already computes the full convolution with the FFT
        // Green's function.  The short-range *correction* accounts for the
        // difference between the exact singular kernel and the smooth kernel
        // that the FFT sees at sub-grid scales:
        //
        //   Phi_correction(x) = G * sum_{|y-x| < r_s, y != x}
        //       rho(y) * eta(|y-x| / r_s) / (4 pi |y-x|) * dV
        //
        // where eta(u) = exp(-u^2).

        let r_s = self.split_radius;
        // Stencil half-width in cells along each axis.
        let stencil = [
            (r_s / self.dx[0]).ceil() as i32,
            (r_s / self.dx[1]).ceil() as i32,
            (r_s / self.dx[2]).ceil() as i32,
        ];
        let dx3 = self.dx[0] * self.dx[1] * self.dx[2];

        let mut phi = long_range.data;

        // Parallel over (ix, iy) slabs.  Each thread processes one row of nz
        // cells, adding the short-range correction in-place.
        phi.par_chunks_mut(nz).enumerate().for_each(|(ij, chunk)| {
            let ix = ij / ny;
            let iy = ij % ny;
            for iz in 0..nz {
                let mut correction = 0.0_f64;
                for di in -stencil[0]..=stencil[0] {
                    for dj in -stencil[1]..=stencil[1] {
                        for dk in -stencil[2]..=stencil[2] {
                            if di == 0 && dj == 0 && dk == 0 {
                                continue;
                            }
                            // Periodic wrapping.
                            let jx = ((ix as i32 + di).rem_euclid(nx as i32)) as usize;
                            let jy = ((iy as i32 + dj).rem_euclid(ny as i32)) as usize;
                            let jz = ((iz as i32 + dk).rem_euclid(nz as i32)) as usize;

                            let rx = di as f64 * self.dx[0];
                            let ry = dj as f64 * self.dx[1];
                            let rz = dk as f64 * self.dx[2];
                            let r = (rx * rx + ry * ry + rz * rz).sqrt();

                            // Short-range kernel: eta(r/r_s) / (4 pi r).
                            let eta = (-r * r / (r_s * r_s)).exp();
                            let kernel = eta / (4.0 * std::f64::consts::PI * r);

                            let rho_j = density.data[jx * ny * nz + jy * nz + jz];
                            correction += g * rho_j * kernel * dx3;
                        }
                    }
                }
                chunk[iz] += correction;
            }
        });

        PotentialField {
            data: phi,
            shape: self.shape,
        }
    }

    /// Compute the gravitational acceleration via second-order finite differences on the potential.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        finite_difference_acceleration(potential, &self.dx)
    }
}

#[cfg(test)]
mod tests {
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::context::SimContext;
    use crate::tooling::core::events::EventEmitter;
    use crate::tooling::core::progress::StepProgress;
    use crate::tooling::core::solver::PoissonSolver as _;

    use super::*;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::poisson::fft::FftPoisson;

    /// Helper: build a small cubic periodic domain.
    fn test_domain(n: i128, l: f64) -> Domain {
        Domain::builder()
            .spatial_extent(l)
            .velocity_extent(1.0)
            .spatial_resolution(n)
            .velocity_resolution(n)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn test_range_separated_construction() {
        let domain = test_domain(8, 4.0);
        let inner = Box::new(FftPoisson::new(&domain));
        let _solver = RangeSeparatedPoisson::new(&domain, 1.0, inner);
        // Construction must not panic.
    }

    #[test]
    fn test_range_separated_uniform_density() {
        let n = 8_usize;
        let domain = test_domain(n as i128, 4.0);
        let inner = Box::new(FftPoisson::new(&domain));
        let solver = RangeSeparatedPoisson::new(&domain, 1.5, inner);

        // Uniform density should produce a nearly flat potential (zero gradient).
        let rho = vec![1.0; n * n * n];
        let density = DensityField {
            data: rho,
            shape: [n, n, n],
        };

        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {
            solver: &solver as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,
        };

        let pot = solver.solve(&density, &_ctx);
        let acc = solver.compute_acceleration(&pot);

        // All acceleration components should be near zero for uniform density.
        let max_gx = acc.gx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_gy = acc.gy.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_gz = acc.gz.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_gx < 1e-10,
            "gx should be ~0 for uniform density, got {max_gx}"
        );
        assert!(
            max_gy < 1e-10,
            "gy should be ~0 for uniform density, got {max_gy}"
        );
        assert!(
            max_gz < 1e-10,
            "gz should be ~0 for uniform density, got {max_gz}"
        );
    }

    #[test]
    fn test_range_separated_vs_fft() {
        // A smooth Gaussian density: the range-separated solver should give
        // results close to plain FFT since both handle smooth fields well.
        let n = 16_usize;
        let l = 4.0;
        let domain = test_domain(n as i128, l);
        let dx = domain.dx();

        let fft_only = FftPoisson::new(&domain);
        let inner = Box::new(FftPoisson::new(&domain));
        let range_sep = RangeSeparatedPoisson::new(&domain, 1.0, inner);

        // Gaussian density centred on the box.
        let sigma = 1.0_f64;
        let mut rho = vec![0.0; n * n * n];
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let x = -l + (ix as f64 + 0.5) * dx[0];
                    let y = -l + (iy as f64 + 0.5) * dx[1];
                    let z = -l + (iz as f64 + 0.5) * dx[2];
                    let r2 = x * x + y * y + z * z;
                    rho[ix * n * n + iy * n + iz] = (-r2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        let density = DensityField {
            data: rho,
            shape: [n, n, n],
        };

        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {
            solver: &fft_only as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,
        };

        let pot_fft = fft_only.solve(&density, &_ctx);
        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {
            solver: &range_sep as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,
        };

        let pot_rs = range_sep.solve(&density, &_ctx);

        // The potentials should be correlated.  Compute relative L2 difference.
        let diff_sq: f64 = pot_fft
            .data
            .iter()
            .zip(pot_rs.data.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let norm_sq: f64 = pot_fft.data.iter().map(|v| v * v).sum();

        let rel_err = (diff_sq / norm_sq).sqrt();
        // The correction adds near-field detail, so there will be some
        // difference, but they should be in the same ballpark for smooth data.
        assert!(
            rel_err < 0.5,
            "Relative L2 error between FFT and range-separated should be moderate, got {rel_err}"
        );
    }

    #[test]
    fn test_range_separated_point_source() {
        let n = 16_usize;
        let domain = test_domain(n as i128, 4.0);
        let inner = Box::new(FftPoisson::new(&domain));
        let solver = RangeSeparatedPoisson::new(&domain, 1.5, inner);

        // Point-like source: all mass in one cell.
        let mut rho = vec![0.0; n * n * n];
        let centre = n / 2;
        rho[centre * n * n + centre * n + centre] = 1.0;
        let density = DensityField {
            data: rho,
            shape: [n, n, n],
        };

        let _advector = SemiLagrangian::new();

        let _emitter = EventEmitter::sink();

        let _progress = StepProgress::new();

        let _ctx = SimContext {
            solver: &solver as &dyn crate::tooling::core::solver::PoissonSolver,

            advector: &_advector,

            emitter: &_emitter,

            progress: &_progress,

            step: 0,

            time: 0.0,

            dt: 0.0,

            g: 1.0,
        };

        let pot = solver.solve(&density, &_ctx);

        // Potential at the source cell should be finite and negative.
        let phi_centre = pot.data[centre * n * n + centre * n + centre];
        assert!(phi_centre.is_finite(), "Potential must be finite");
        assert!(
            phi_centre < 0.0,
            "Potential at source should be negative, got {phi_centre}"
        );

        // Potential should decay away from the source: a neighbouring cell
        // should have a less-negative (higher) value.
        let phi_neighbour = pot.data[centre * n * n + centre * n + centre + 1];
        assert!(
            phi_neighbour > phi_centre,
            "Potential should decay: phi_neighbour ({phi_neighbour}) > phi_centre ({phi_centre})"
        );
    }
}
