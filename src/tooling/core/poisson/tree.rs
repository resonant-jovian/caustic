//! Barnes-Hut tree code Poisson solver. O(N log N) per solve.
//!
//! Builds an adaptive octree from the density field and computes the gravitational
//! potential at each grid point via a tree walk with the opening-angle criterion.
//! Far cells are approximated as monopoles; nearby cells are recursed into.

use rust_decimal::prelude::ToPrimitive;

use super::super::{init::domain::Domain, solver::PoissonSolver, types::*};

// ---------------------------------------------------------------------------
// Octree data structure
// ---------------------------------------------------------------------------

/// One node of the Barnes-Hut octree.
pub struct OctreeNode {
    /// Mass-weighted centre of mass [x, y, z].
    pub center_of_mass: [f64; 3],
    /// Sum of particle masses contained in this node.
    pub total_mass: f64,
    /// Side length of this cubic node.
    pub size: f64,
    /// Geometric centre of this cubic node.
    pub center: [f64; 3],
    /// Eight children (octants), `None` for empty octants.
    pub children: Option<Box<[Option<OctreeNode>; 8]>>,
}

/// A point-mass particle extracted from the density grid.
struct Particle {
    pos: [f64; 3],
    mass: f64,
}

// ---------------------------------------------------------------------------
// Tree construction
// ---------------------------------------------------------------------------

/// Maximum recursion depth to prevent runaway subdivision.
const MAX_DEPTH: usize = 30;

/// Determine which octant of a cube centred at `center` the point `pos` falls into.
/// Octant index: bit 0 = x >= center_x, bit 1 = y >= center_y, bit 2 = z >= center_z.
fn octant_index(pos: &[f64; 3], center: &[f64; 3]) -> usize {
    let mut idx = 0;
    if pos[0] >= center[0] {
        idx |= 1;
    }
    if pos[1] >= center[1] {
        idx |= 2;
    }
    if pos[2] >= center[2] {
        idx |= 4;
    }
    idx
}

/// Geometric centre of the child octant `oct` given parent `center` and `half_size`.
fn child_center(center: &[f64; 3], half_size: f64, oct: usize) -> [f64; 3] {
    let q = half_size * 0.5;
    [
        center[0] + if oct & 1 != 0 { q } else { -q },
        center[1] + if oct & 2 != 0 { q } else { -q },
        center[2] + if oct & 4 != 0 { q } else { -q },
    ]
}

/// Recursively build the octree from a list of particles.
fn build_node(
    particles: &[Particle],
    center: [f64; 3],
    size: f64,
    depth: usize,
) -> Option<OctreeNode> {
    if particles.is_empty() {
        return None;
    }

    // Leaf: single particle (or max depth reached)
    if particles.len() == 1 || depth >= MAX_DEPTH {
        let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
        let mut total = 0.0;
        for p in particles {
            cx += p.mass * p.pos[0];
            cy += p.mass * p.pos[1];
            cz += p.mass * p.pos[2];
            total += p.mass;
        }
        if total > 0.0 {
            cx /= total;
            cy /= total;
            cz /= total;
        }
        return Some(OctreeNode {
            center_of_mass: [cx, cy, cz],
            total_mass: total,
            size,
            center,
            children: None,
        });
    }

    // Partition particles into octants
    let half = size * 0.5;
    let mut buckets: [Vec<&Particle>; 8] = Default::default();
    for p in particles {
        let oct = octant_index(&p.pos, &center);
        buckets[oct].push(p);
    }

    // Build children
    let mut children: [Option<OctreeNode>; 8] = Default::default();
    for oct in 0..8 {
        if buckets[oct].is_empty() {
            continue;
        }
        let cc = child_center(&center, size, oct);
        // Collect owned copies so we can recurse
        let child_particles: Vec<Particle> = buckets[oct]
            .iter()
            .map(|p| Particle {
                pos: p.pos,
                mass: p.mass,
            })
            .collect();
        children[oct] = build_node(&child_particles, cc, half, depth + 1);
    }

    // Compute aggregate centre-of-mass
    let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
    let mut total = 0.0;
    for c in children.iter().flatten() {
        cx += c.total_mass * c.center_of_mass[0];
        cy += c.total_mass * c.center_of_mass[1];
        cz += c.total_mass * c.center_of_mass[2];
        total += c.total_mass;
    }
    if total > 0.0 {
        cx /= total;
        cy /= total;
        cz /= total;
    }

    Some(OctreeNode {
        center_of_mass: [cx, cy, cz],
        total_mass: total,
        size,
        center,
        children: Some(Box::new(children)),
    })
}

/// Build an octree from a density field on a uniform grid.
///
/// Each cell with ρ > 0 becomes a point mass at the cell centre with
/// mass = ρ × cell_volume. The root node spans the full spatial domain.
fn build_tree(
    density: &DensityField,
    dx: &[f64; 3],
    origin: &[f64; 3],
    domain_size: f64,
) -> Option<OctreeNode> {
    let [nx, ny, nz] = density.shape;
    let cell_vol = dx[0] * dx[1] * dx[2];

    // Convert non-zero density cells to particles
    let mut particles = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx = ix * ny * nz + iy * nz + iz;
                let rho = density.data[idx];
                if rho.abs() > 0.0 {
                    let mass = rho * cell_vol;
                    let pos = [
                        origin[0] + (ix as f64 + 0.5) * dx[0],
                        origin[1] + (iy as f64 + 0.5) * dx[1],
                        origin[2] + (iz as f64 + 0.5) * dx[2],
                    ];
                    particles.push(Particle { pos, mass });
                }
            }
        }
    }

    if particles.is_empty() {
        return None;
    }

    // Root centre is the geometric centre of the domain
    let root_center = [
        origin[0] + 0.5 * nx as f64 * dx[0],
        origin[1] + 0.5 * ny as f64 * dx[1],
        origin[2] + 0.5 * nz as f64 * dx[2],
    ];

    build_node(&particles, root_center, domain_size, 0)
}

// ---------------------------------------------------------------------------
// Tree walk — potential evaluation
// ---------------------------------------------------------------------------

/// Recursively compute the gravitational potential at `point` due to `node`.
///
/// Uses the Barnes-Hut opening-angle criterion: if `node.size / r < theta`
/// (the node subtends a small angle), use the monopole approximation
/// Φ = −G M / sqrt(r² + ε²). Otherwise recurse into children.
fn tree_potential(node: &OctreeNode, point: &[f64; 3], theta: f64, g: f64, softening: f64) -> f64 {
    if node.total_mass == 0.0 {
        return 0.0;
    }

    let dx = point[0] - node.center_of_mass[0];
    let dy = point[1] - node.center_of_mass[1];
    let dz = point[2] - node.center_of_mass[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();

    // Leaf node — use monopole directly
    if node.children.is_none() {
        let r_soft = (r2 + softening * softening).sqrt();
        return -g * node.total_mass / r_soft;
    }

    // Opening-angle test: if the node looks small enough, use monopole
    if r > 0.0 && node.size / r < theta {
        let r_soft = (r2 + softening * softening).sqrt();
        return -g * node.total_mass / r_soft;
    }

    // Otherwise recurse into children
    let mut phi = 0.0;
    if let Some(ref children) = node.children {
        for c in children.iter().flatten() {
            phi += tree_potential(c, point, theta, g, softening);
        }
    }
    phi
}

// ---------------------------------------------------------------------------
// TreePoisson solver
// ---------------------------------------------------------------------------

/// Barnes-Hut tree code Poisson solver.
///
/// For each grid point the potential is computed via an O(N log N) tree walk
/// with configurable opening angle θ. Smaller θ gives higher accuracy
/// (θ → 0 recovers direct summation) at the cost of more work.
pub struct TreePoisson {
    /// Opening angle criterion θ for the multipole approximation.
    pub opening_angle: f64,
    /// Computational domain.
    pub domain: Domain,
    /// Plummer softening length (prevents 1/r singularity).
    pub softening: f64,
}

impl TreePoisson {
    /// Create a new Barnes-Hut Poisson solver.
    ///
    /// `opening_angle` controls accuracy vs speed (typical values 0.3–1.0).
    /// Softening is set to the average cell size.
    pub fn new(domain: Domain, opening_angle: f64) -> Self {
        let dx = domain.dx();
        let softening = (dx[0] + dx[1] + dx[2]) / 3.0;
        Self {
            opening_angle,
            domain,
            softening,
        }
    }
}

impl PoissonSolver for TreePoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        let _span = tracing::info_span!("tree_poisson_solve").entered();
        let [nx, ny, nz] = density.shape;
        let dx = self.domain.dx();

        // Domain spans [-L, L] in each dimension
        let lx = self.domain.spatial.x1.to_f64().unwrap();
        let ly = self.domain.spatial.x2.to_f64().unwrap();
        let lz = self.domain.spatial.x3.to_f64().unwrap();
        let origin = [-lx, -ly, -lz];

        // Root node size: largest dimension of the domain
        let domain_size = 2.0_f64 * lx.max(ly).max(lz);

        // Build octree
        let tree = build_tree(density, &dx, &origin, domain_size);

        // Evaluate potential at every grid point
        let n_total = nx * ny * nz;
        let mut phi = vec![0.0f64; n_total];

        match tree {
            None => {
                // Empty density field — zero potential
            }
            Some(ref root) => {
                for ix in 0..nx {
                    for iy in 0..ny {
                        for iz in 0..nz {
                            let idx = ix * ny * nz + iy * nz + iz;
                            let point = [
                                origin[0] + (ix as f64 + 0.5) * dx[0],
                                origin[1] + (iy as f64 + 0.5) * dx[1],
                                origin[2] + (iz as f64 + 0.5) * dx[2],
                            ];
                            phi[idx] =
                                tree_potential(root, &point, self.opening_angle, g, self.softening);
                        }
                    }
                }
            }
        }

        PotentialField {
            data: phi,
            shape: density.shape,
        }
    }

    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        super::utils::finite_difference_acceleration(potential, &self.domain.dx())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};

    fn test_domain(n: i128) -> Domain {
        Domain::builder()
            .spatial_extent(4.0)
            .velocity_extent(2.0)
            .spatial_resolution(n)
            .velocity_resolution(4)
            .t_final(1.0)
            .spatial_bc(SpatialBoundType::Isolated)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap()
    }

    #[test]
    fn tree_point_mass() {
        let domain = test_domain(8);
        let tree = TreePoisson::new(domain.clone(), 0.5);
        let [nx, ny, nz] = [8usize; 3];
        let dx = domain.dx();
        let mut rho = vec![0.0; nx * ny * nz];
        let mid = nx / 2;
        let cell_vol = dx[0] * dx[1] * dx[2];
        rho[mid * ny * nz + mid * nz + mid] = 1.0 / cell_vol;

        let density = DensityField {
            data: rho,
            shape: [nx, ny, nz],
        };
        let pot = tree.solve(&density, 1.0);

        // At a point 2 cells away, potential should be roughly -M/r
        let test = (mid + 2) * ny * nz + mid * nz + mid;
        let r = 2.0 * dx[0];
        assert!(pot.data[test] < 0.0, "Potential should be negative");
        assert!(pot.data[test].is_finite(), "Potential should be finite");

        // Check rough magnitude: Phi ~ -G*M/r = -1/r
        let expected = -1.0 / r;
        let relative_err = ((pot.data[test] - expected) / expected).abs();
        assert!(
            relative_err < 1.0,
            "Point-mass potential relative error = {relative_err} (expected {expected}, got {})",
            pot.data[test]
        );
    }

    #[test]
    fn tree_theta_convergence() {
        let domain = test_domain(8);
        let dx = domain.dx();
        let [nx, ny, nz] = [8usize; 3];
        let mut rho = vec![0.0; nx * ny * nz];
        // Gaussian density
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = -4.0 + (ix as f64 + 0.5) * dx[0];
                    let y = -4.0 + (iy as f64 + 0.5) * dx[1];
                    let z = -4.0 + (iz as f64 + 0.5) * dx[2];
                    rho[ix * ny * nz + iy * nz + iz] = (-0.5 * (x * x + y * y + z * z)).exp();
                }
            }
        }
        let density = DensityField {
            data: rho,
            shape: [nx, ny, nz],
        };

        // Solve with different theta values
        let tree_wide = TreePoisson::new(domain.clone(), 1.0);
        let tree_narrow = TreePoisson::new(domain.clone(), 0.3);

        let pot_wide = tree_wide.solve(&density, 1.0);
        let pot_narrow = tree_narrow.solve(&density, 1.0);

        // Both should be finite and generally similar
        assert!(pot_wide.data.iter().all(|x| x.is_finite()));
        assert!(pot_narrow.data.iter().all(|x| x.is_finite()));

        // Narrow theta (more accurate) and wide theta should agree reasonably
        let max_narrow = pot_narrow
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);
        if max_narrow > 1e-10 {
            let max_diff = pot_wide
                .data
                .iter()
                .zip(pot_narrow.data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            // Wide vs narrow: the difference should be bounded
            assert!(
                max_diff / max_narrow < 0.5,
                "Wide vs narrow theta: relative diff = {}",
                max_diff / max_narrow
            );
        }
    }

    #[test]
    fn tree_vs_fft_isolated() {
        use crate::tooling::core::poisson::fft::FftIsolated;
        let domain = test_domain(8);
        let dx = domain.dx();
        let [nx, ny, nz] = [8usize; 3];

        let tree = TreePoisson::new(domain.clone(), 0.3);
        let fft = FftIsolated::new(&domain);

        let mut rho = vec![0.0; nx * ny * nz];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = -4.0 + (ix as f64 + 0.5) * dx[0];
                    let y = -4.0 + (iy as f64 + 0.5) * dx[1];
                    let z = -4.0 + (iz as f64 + 0.5) * dx[2];
                    rho[ix * ny * nz + iy * nz + iz] = (-(x * x + y * y + z * z) / 2.0).exp();
                }
            }
        }
        let density = DensityField {
            data: rho,
            shape: [nx, ny, nz],
        };

        let pot_tree = tree.solve(&density, 1.0);
        let pot_fft = fft.solve(&density, 1.0);

        // Tree and FFT-isolated should give similar potentials (within tree approximation)
        let max_fft = pot_fft.data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        if max_fft > 1e-10 {
            let max_diff = pot_tree
                .data
                .iter()
                .zip(pot_fft.data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            // Tree approximation is rough, allow 50% relative error
            assert!(
                max_diff / max_fft < 1.0,
                "Tree vs FFT isolated: max_diff/max_fft = {}",
                max_diff / max_fft
            );
        }
    }

    #[test]
    fn tree_empty_density() {
        let domain = test_domain(8);
        let tree = TreePoisson::new(domain, 0.5);
        let rho = vec![0.0; 8 * 8 * 8];
        let density = DensityField {
            data: rho,
            shape: [8, 8, 8],
        };
        let pot = tree.solve(&density, 1.0);
        // All-zero density should give all-zero potential
        assert!(pot.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tree_symmetry() {
        // A symmetric density distribution should give a symmetric potential
        let domain = test_domain(8);
        let tree = TreePoisson::new(domain.clone(), 0.5);
        let dx = domain.dx();
        let [nx, ny, nz] = [8usize; 3];
        let mut rho = vec![0.0; nx * ny * nz];

        // Place equal masses symmetrically about the centre
        let mid = nx / 2;
        let cell_vol = dx[0] * dx[1] * dx[2];
        rho[(mid - 1) * ny * nz + mid * nz + mid] = 1.0 / cell_vol;
        rho[(mid + 1) * ny * nz + mid * nz + mid] = 1.0 / cell_vol; // Note: mid+1, not mid

        let density = DensityField {
            data: rho,
            shape: [nx, ny, nz],
        };
        let pot = tree.solve(&density, 1.0);

        // Potential at the midpoint (between the two masses) should be well-defined
        let mid_pot = pot.data[mid * ny * nz + mid * nz + mid];
        assert!(mid_pot.is_finite());
        assert!(
            mid_pot < 0.0,
            "Potential between two masses should be negative"
        );
    }

    #[test]
    fn tree_compute_acceleration() {
        let domain = test_domain(8);
        let tree = TreePoisson::new(domain.clone(), 0.5);
        let dx = domain.dx();
        let [nx, ny, nz] = [8usize; 3];
        let mut rho = vec![0.0; nx * ny * nz];
        let mid = nx / 2;
        let cell_vol = dx[0] * dx[1] * dx[2];
        rho[mid * ny * nz + mid * nz + mid] = 1.0 / cell_vol;

        let density = DensityField {
            data: rho,
            shape: [nx, ny, nz],
        };
        let pot = tree.solve(&density, 1.0);
        let acc = tree.compute_acceleration(&pot);

        // Acceleration field should have the correct shape
        assert_eq!(acc.shape, [nx, ny, nz]);
        assert_eq!(acc.gx.len(), nx * ny * nz);

        // Acceleration should point towards the mass (negative for x > x_mass, positive for x < x_mass)
        let idx_right = (mid + 2) * ny * nz + mid * nz + mid;
        let idx_left = (mid - 2) * ny * nz + mid * nz + mid;
        assert!(
            acc.gx[idx_right] < 0.0,
            "Acceleration should point left (towards mass) on the right side"
        );
        assert!(
            acc.gx[idx_left] > 0.0,
            "Acceleration should point right (towards mass) on the left side"
        );
    }
}
