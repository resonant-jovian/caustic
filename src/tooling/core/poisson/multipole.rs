//! Multipole expansion of the gravitational potential using real spherical harmonics.
//!
//! Computes multipole moments q_lm from the density field and evaluates
//! the far-field potential via:
//!   Phi(x) = -G * sum_{l,m} [4pi/(2l+1)] * q_lm * Y_lm(theta,phi) / r^{l+1}
//!
//! Cost is O(l_max^2 * N) for moment computation and O(l_max^2) per
//! evaluation point, making it efficient for providing Dirichlet boundary
//! values on box faces. Uses real spherical harmonics up to a configurable
//! l_max (currently l <= 2 is implemented; typically 4-8 is sufficient for
//! gravitational applications).

use super::super::types::DensityField;

/// Boundary values on the 6 faces of a 3D box.
pub struct BoundaryValues {
    /// Potential values on each face: [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi].
    /// Each face is a 2D array stored row-major.
    pub faces: [Vec<f64>; 6],
    /// Grid shape [nx, ny, nz] used to index the face arrays.
    pub shape: [usize; 3],
}

/// Multipole expansion calculator for computing far-field boundary conditions.
pub struct MultipoleExpansion {
    /// Maximum angular momentum quantum number.
    pub l_max: usize,
    /// Grid spacing.
    pub dx: [f64; 3],
    /// Grid shape.
    pub shape: [usize; 3],
}

impl MultipoleExpansion {
    /// Create a multipole expansion calculator for the given grid geometry.
    pub fn new(shape: [usize; 3], dx: [f64; 3], l_max: usize) -> Self {
        Self { l_max, dx, shape }
    }

    /// Compute multipole moments q_lm from the density field.
    ///
    /// q_lm = integral rho(x) * r^l * Y_lm(theta, phi) * dx^3
    ///
    /// Returns moments indexed as [(l, m)] for l=0..l_max, m=-l..l.
    pub fn compute_moments(&self, density: &DensityField, g: f64) -> Vec<(usize, i32, f64)> {
        let [nx, ny, nz] = self.shape;
        let cx = nx as f64 / 2.0 * self.dx[0];
        let cy = ny as f64 / 2.0 * self.dx[1];
        let cz = nz as f64 / 2.0 * self.dx[2];
        let dv = self.dx[0] * self.dx[1] * self.dx[2];

        let mut moments = Vec::new();

        for l in 0..=self.l_max {
            for m in -(l as i32)..=(l as i32) {
                let mut q_lm = 0.0f64;
                for ix in 0..nx {
                    let x = (ix as f64 + 0.5) * self.dx[0] - cx;
                    for iy in 0..ny {
                        let y = (iy as f64 + 0.5) * self.dx[1] - cy;
                        for iz in 0..nz {
                            let z = (iz as f64 + 0.5) * self.dx[2] - cz;
                            let rho = density.data[ix * ny * nz + iy * nz + iz];
                            let r = (x * x + y * y + z * z).sqrt();
                            if r > 1e-30 {
                                let ylm = real_spherical_harmonic(l, m, x, y, z, r);
                                q_lm += rho * r.powi(l as i32) * ylm * dv;
                            }
                        }
                    }
                }
                moments.push((l, m, q_lm));
            }
        }

        moments
    }

    /// Evaluate the multipole potential at a point (x, y, z).
    pub fn evaluate_potential(
        &self,
        moments: &[(usize, i32, f64)],
        x: f64,
        y: f64,
        z: f64,
        g: f64,
    ) -> f64 {
        let r = (x * x + y * y + z * z).sqrt();
        if r < 1e-30 {
            return 0.0;
        }

        let mut phi = 0.0f64;
        for &(l, m, q_lm) in moments {
            let ylm = real_spherical_harmonic(l, m, x, y, z, r);
            let factor = 4.0 * std::f64::consts::PI / (2 * l + 1) as f64;
            phi += -g * factor * q_lm * ylm / r.powi(l as i32 + 1);
        }
        phi
    }

    /// Compute boundary values from multipole expansion.
    pub fn compute_boundary_values(&self, density: &DensityField, g: f64) -> BoundaryValues {
        let [nx, ny, nz] = self.shape;
        let cx = nx as f64 / 2.0 * self.dx[0];
        let cy = ny as f64 / 2.0 * self.dx[1];
        let cz = nz as f64 / 2.0 * self.dx[2];

        let moments = self.compute_moments(density, g);

        let mut faces = [
            vec![0.0; ny * nz], // x_lo
            vec![0.0; ny * nz], // x_hi
            vec![0.0; nx * nz], // y_lo
            vec![0.0; nx * nz], // y_hi
            vec![0.0; nx * ny], // z_lo
            vec![0.0; nx * ny], // z_hi
        ];

        // x_lo face (ix = 0)
        let x = 0.5 * self.dx[0] - cx;
        for iy in 0..ny {
            let y = (iy as f64 + 0.5) * self.dx[1] - cy;
            for iz in 0..nz {
                let z = (iz as f64 + 0.5) * self.dx[2] - cz;
                faces[0][iy * nz + iz] = self.evaluate_potential(&moments, x, y, z, g);
            }
        }

        // x_hi face (ix = nx-1)
        let x = (nx as f64 - 0.5) * self.dx[0] - cx;
        for iy in 0..ny {
            let y = (iy as f64 + 0.5) * self.dx[1] - cy;
            for iz in 0..nz {
                let z = (iz as f64 + 0.5) * self.dx[2] - cz;
                faces[1][iy * nz + iz] = self.evaluate_potential(&moments, x, y, z, g);
            }
        }

        // y_lo face
        let y = 0.5 * self.dx[1] - cy;
        for ix in 0..nx {
            let x = (ix as f64 + 0.5) * self.dx[0] - cx;
            for iz in 0..nz {
                let z = (iz as f64 + 0.5) * self.dx[2] - cz;
                faces[2][ix * nz + iz] = self.evaluate_potential(&moments, x, y, z, g);
            }
        }

        // y_hi face
        let y = (ny as f64 - 0.5) * self.dx[1] - cy;
        for ix in 0..nx {
            let x = (ix as f64 + 0.5) * self.dx[0] - cx;
            for iz in 0..nz {
                let z = (iz as f64 + 0.5) * self.dx[2] - cz;
                faces[3][ix * nz + iz] = self.evaluate_potential(&moments, x, y, z, g);
            }
        }

        // z_lo face
        let z = 0.5 * self.dx[2] - cz;
        for ix in 0..nx {
            let x = (ix as f64 + 0.5) * self.dx[0] - cx;
            for iy in 0..ny {
                let y = (iy as f64 + 0.5) * self.dx[1] - cy;
                faces[4][ix * ny + iy] = self.evaluate_potential(&moments, x, y, z, g);
            }
        }

        // z_hi face
        let z = (nz as f64 - 0.5) * self.dx[2] - cz;
        for ix in 0..nx {
            let x = (ix as f64 + 0.5) * self.dx[0] - cx;
            for iy in 0..ny {
                let y = (iy as f64 + 0.5) * self.dx[1] - cy;
                faces[5][ix * ny + iy] = self.evaluate_potential(&moments, x, y, z, g);
            }
        }

        BoundaryValues {
            faces,
            shape: self.shape,
        }
    }
}

/// Real spherical harmonic Y_lm evaluated at Cartesian coordinates.
///
/// Only implements l=0,1,2 for now (sufficient for most gravitational applications).
fn real_spherical_harmonic(l: usize, m: i32, x: f64, y: f64, z: f64, r: f64) -> f64 {
    use std::f64::consts::PI;
    let r_inv = 1.0 / r.max(1e-30);
    match (l, m) {
        (0, 0) => 0.5 * (1.0 / PI).sqrt(),
        (1, -1) => (3.0 / (4.0 * PI)).sqrt() * y * r_inv,
        (1, 0) => (3.0 / (4.0 * PI)).sqrt() * z * r_inv,
        (1, 1) => (3.0 / (4.0 * PI)).sqrt() * x * r_inv,
        (2, -2) => 0.5 * (15.0 / PI).sqrt() * x * y * r_inv * r_inv,
        (2, -1) => 0.5 * (15.0 / PI).sqrt() * y * z * r_inv * r_inv,
        (2, 0) => 0.25 * (5.0 / PI).sqrt() * (3.0 * z * z * r_inv * r_inv - 1.0),
        (2, 1) => 0.5 * (15.0 / PI).sqrt() * x * z * r_inv * r_inv,
        (2, 2) => 0.25 * (15.0 / PI).sqrt() * (x * x - y * y) * r_inv * r_inv,
        _ => 0.0, // Higher l not implemented
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monopole_point_mass() {
        // A uniform density in a small region acts like a point mass at large r
        let n = 8;
        let dx = [1.0; 3];
        let shape = [n, n, n];
        let mut rho = vec![0.0; n * n * n];

        // Place mass at center
        let mid = n / 2;
        rho[mid * n * n + mid * n + mid] = 1.0;

        let density = DensityField { data: rho, shape };
        let me = MultipoleExpansion::new(shape, dx, 0);
        let moments = me.compute_moments(&density, 1.0);

        // Monopole moment should be non-zero
        assert!(
            moments[0].2.abs() > 0.0,
            "Monopole moment should be non-zero"
        );
    }
}
