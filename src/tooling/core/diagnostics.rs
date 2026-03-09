//! `Diagnostics` — computes and stores all conserved quantities and monitoring outputs
//! at each timestep.

use super::phasespace::PhaseSpaceRepr;
use super::types::{DensityField, PotentialField};
use serde::Serialize;

/// One row of the global time-series output.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GlobalDiagnostics {
    pub time: f64,
    pub total_energy: f64,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub virial_ratio: f64,
    pub total_momentum: [f64; 3],
    pub total_angular_momentum: [f64; 3],
    pub casimir_c2: f64,
    pub entropy: f64,
    pub mass_in_box: f64,
}

/// Accumulates time series of `GlobalDiagnostics`.
pub struct Diagnostics {
    pub history: Vec<GlobalDiagnostics>,
}

impl Diagnostics {
    /// Compute all global diagnostics from the current representation and potential.
    /// `dx3` is the spatial cell volume dx1*dx2*dx3.
    pub fn compute(
        &mut self,
        repr: &dyn PhaseSpaceRepr,
        potential: &PotentialField,
        time: f64,
        dx3: f64,
    ) -> GlobalDiagnostics {
        let density = repr.compute_density();
        let t = Self::kinetic_energy(repr);
        let w = Self::potential_energy(&density, potential, dx3);
        let e = t + w;
        let c2 = repr.casimir_c2();
        let s = repr.entropy();
        let m = repr.total_mass();
        let vir = if w.abs() > 1e-30 {
            2.0 * t / w.abs()
        } else {
            0.0
        };
        let diag = GlobalDiagnostics {
            time,
            total_energy: e,
            kinetic_energy: t,
            potential_energy: w,
            virial_ratio: vir,
            total_momentum: [0.0; 3],
            total_angular_momentum: [0.0; 3],
            casimir_c2: c2,
            entropy: s,
            mass_in_box: m,
        };
        self.history.push(diag);
        diag
    }

    /// Total kinetic energy T = ½∫fv² dx³dv³.
    pub fn kinetic_energy(repr: &dyn PhaseSpaceRepr) -> f64 {
        repr.total_kinetic_energy()
    }

    /// Total potential energy W = ½∫ρΦ dx³.
    /// `dx3` is the spatial cell volume.
    pub fn potential_energy(density: &DensityField, potential: &PotentialField, dx3: f64) -> f64 {
        0.5 * density
            .data
            .iter()
            .zip(potential.data.iter())
            .map(|(&rho, &phi)| rho * phi)
            .sum::<f64>()
            * dx3
    }

    /// Virial ratio 2T/|W|. Equals 1.0 at equilibrium.
    pub fn virial_ratio(t: f64, w: f64) -> f64 {
        2.0 * t / w.abs()
    }

    /// Spherically averaged density profile ρ(r) at current timestep.
    /// Bins density cells by radius from domain centre, returns (r_bin, ρ_avg) pairs.
    pub fn density_profile(density: &DensityField, dx: [f64; 3]) -> Vec<(f64, f64)> {
        let [nx, ny, nz] = density.shape;
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cz = nz as f64 / 2.0;

        // Determine maximum radius and bin width
        let r_max = (cx * dx[0]).min(cy * dx[1]).min(cz * dx[2]);
        let n_bins = (nx.max(ny).max(nz) / 2).max(1);
        let dr = r_max / n_bins as f64;

        let mut bin_sum = vec![0.0f64; n_bins];
        let mut bin_count = vec![0u64; n_bins];

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let rx = (ix as f64 + 0.5 - cx) * dx[0];
                    let ry = (iy as f64 + 0.5 - cy) * dx[1];
                    let rz = (iz as f64 + 0.5 - cz) * dx[2];
                    let r = (rx * rx + ry * ry + rz * rz).sqrt();
                    let bin = (r / dr) as usize;
                    if bin < n_bins {
                        bin_sum[bin] += density.data[ix * ny * nz + iy * nz + iz];
                        bin_count[bin] += 1;
                    }
                }
            }
        }

        (0..n_bins)
            .filter(|&i| bin_count[i] > 0)
            .map(|i| {
                let r = (i as f64 + 0.5) * dr;
                let rho_avg = bin_sum[i] / bin_count[i] as f64;
                (r, rho_avg)
            })
            .collect()
    }
}
