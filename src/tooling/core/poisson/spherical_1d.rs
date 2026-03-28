//! 1D radial Poisson solver for spherically symmetric density.
//!
//! Solves the Poisson equation in a single radial coordinate by
//! substituting u = r*Phi to reduce the radial ODE
//! d²(r·Phi)/dr² = 4*pi*G*r*rho(r) to a standard second-order form,
//! then solves the resulting tridiagonal system via the Thomas algorithm.
//!
//! Designed for use with `SphericalRepr` where the density is purely
//! radial, avoiding the cost of a full 3D solve.

use super::super::{context::SimContext, events::{SimEvent, SolverKind}, solver::PoissonSolver, types::*};

/// 1D Poisson solver for spherically symmetric density on a uniform radial grid.
pub struct Spherical1DPoisson {
    /// Number of radial grid cells.
    pub nr: usize,
    /// Radial cell spacing.
    pub dr: f64,
    /// Inner edge of the radial grid.
    pub r_min: f64,
}

impl Spherical1DPoisson {
    /// Create a solver for `nr` radial cells of width `dr` starting at `r_min`.
    pub fn new(nr: usize, dr: f64, r_min: f64) -> Self {
        Self { nr, dr, r_min }
    }

    /// Radial coordinate at cell center.
    #[inline]
    fn r_at(&self, ir: usize) -> f64 {
        self.r_min + (ir as f64 + 0.5) * self.dr
    }
}

impl PoissonSolver for Spherical1DPoisson {
    /// Solve for the radial potential via the Thomas algorithm on the tridiagonal system.
    fn solve(&self, density: &DensityField, ctx: &SimContext) -> PotentialField {
        let t0 = std::time::Instant::now();
        let g = ctx.g;
        let nr = density.data.len();
        let four_pi_g = 4.0 * std::f64::consts::PI * g;

        // Solve d²(r·Φ)/dr² = 4πG·r·ρ(r)
        // Let u = r·Φ, then u'' = 4πG·r·ρ
        // Tridiagonal: (u[i-1] - 2*u[i] + u[i+1]) / dr² = 4πG·r_i·ρ_i
        // BC: u(0) = 0 (regularity), u(r_max) = -G*M/1 (if we know total mass, or u' = Phi at boundary)

        let dr2 = self.dr * self.dr;
        let mut rhs = vec![0.0f64; nr];
        for (i, rhs_val) in rhs.iter_mut().enumerate() {
            let r = self.r_at(i);
            *rhs_val = four_pi_g * r * density.data[i] * dr2;
        }

        // Tridiagonal solve: a[i]*u[i-1] + b[i]*u[i] + c[i]*u[i+1] = rhs[i]
        // a = 1, b = -2, c = 1
        let mut u = vec![0.0f64; nr];

        // Forward sweep (Thomas algorithm)
        let mut c_prime = vec![0.0f64; nr];
        let mut d_prime = vec![0.0f64; nr];

        c_prime[0] = -0.5; // c[0]/b[0] = 1/(-2)
        d_prime[0] = rhs[0] / -2.0;

        for i in 1..nr {
            let m = 1.0 / (-2.0 - c_prime[i - 1]);
            c_prime[i] = if i < nr - 1 { m } else { 0.0 };
            d_prime[i] = (rhs[i] - d_prime[i - 1]) * m;
        }

        // Back substitution
        u[nr - 1] = d_prime[nr - 1];
        for i in (0..nr - 1).rev() {
            u[i] = d_prime[i] - c_prime[i] * u[i + 1];
        }

        // Convert u = r·Φ back to Φ
        let mut phi = vec![0.0f64; nr];
        for i in 0..nr {
            let r = self.r_at(i);
            phi[i] = if r > 1e-30 {
                u[i] / r
            } else {
                u[0] / self.r_at(0)
            };
        }

        let result = PotentialField {
            data: phi,
            shape: [nr, 1, 1],
        };
        ctx.emitter.emit(SimEvent::PoissonSolveComplete {
            solver: SolverKind::Spherical1D,
            wall_us: t0.elapsed().as_micros() as u64,
        });
        result
    }

    /// Compute radial gravitational acceleration via centered finite differences.
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        let nr = potential.data.len();
        let mut gr = vec![0.0f64; nr];

        for (gr_val, window) in gr[1..nr - 1].iter_mut().zip(potential.data.windows(3)) {
            *gr_val = -(window[2] - window[0]) / (2.0 * self.dr);
        }
        if nr > 1 {
            gr[0] = -(potential.data[1] - potential.data[0]) / self.dr;
            gr[nr - 1] = -(potential.data[nr - 1] - potential.data[nr - 2]) / self.dr;
        }

        AccelerationField {
            gx: gr,
            gy: vec![0.0; nr],
            gz: vec![0.0; nr],
            shape: [nr, 1, 1],
        }
    }
}
