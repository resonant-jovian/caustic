//! Phase-space structure diagnostics: stream count, caustic surfaces, power spectrum,
//! and growth rates.
//!
//! [`PhaseSpaceDiagnostics`] aggregates per-cell stream-count data from the
//! representation, identifies multi-stream (caustic) cells, and provides static
//! helpers for Fourier-space analysis. [`PhaseSpaceDiagnostics::power_spectrum`]
//! computes the density power spectrum P(k) = |rho_hat(k)|^2 binned by
//! wavenumber magnitude, while [`field_energy_spectrum`] gives the gravitational
//! field energy E(k) = |k|^2 |Phi_hat(k)|^2. Growth-rate fitting via
//! [`PhaseSpaceDiagnostics::growth_rates`] enables validation against analytic
//! dispersion relations (Jeans, Landau, two-stream).

use super::super::{phasespace::PhaseSpaceRepr, poisson::utils::fft_3d_c2c_inplace, types::*};
use rayon::prelude::*;
use rustfft::num_complex::Complex;

/// Collected phase-space structure diagnostics for a single time step.
pub struct PhaseSpaceDiagnostics {
    /// Number of overlapping phase-space streams in each spatial cell.
    pub stream_count: StreamCountField,
    /// Per-cell Boltzmann entropy (currently unpopulated; requires grid-level access).
    pub local_entropy: Vec<f64>,
    /// Grid indices `[ix1, ix2, ix3]` of cells with stream count > 1 (caustic locations).
    pub caustic_cells: Vec<[usize; 3]>,
}

impl PhaseSpaceDiagnostics {
    /// Build diagnostics from the current phase-space representation.
    ///
    /// Queries the stream-count field and marks every cell with count > 1 as a
    /// caustic location.
    pub fn compute(repr: &dyn PhaseSpaceRepr) -> Self {
        let stream_count = repr.stream_count();
        let [nx1, nx2, nx3] = stream_count.shape;

        // Find cells where stream count > 1 (caustic locations)
        let mut caustic_cells = Vec::new();
        for ix1 in 0..nx1 {
            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    let idx = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                    if stream_count.data[idx] > 1 {
                        caustic_cells.push([ix1, ix2, ix3]);
                    }
                }
            }
        }

        Self {
            stream_count,
            local_entropy: Vec::new(), // per-cell entropy needs grid access not in trait
            caustic_cells,
        }
    }

    /// Extract f(v|x) at a given physical position for dark matter detection predictions.
    pub fn velocity_distribution_at(repr: &dyn PhaseSpaceRepr, x: [f64; 3]) -> Vec<f64> {
        repr.velocity_distribution(&x)
    }

    /// Power spectrum P(k) = |ρ̂(k)|². FFT of density field.
    ///
    /// Returns (k, P(k)) pairs binned by integer |k| magnitude.
    pub fn power_spectrum(density: &DensityField) -> Vec<(f64, f64)> {
        let [nx, ny, nz] = density.shape;

        // 3D C2C FFT
        let mut data: Vec<Complex<f64>> =
            density.data.iter().map(|&r| Complex::new(r, 0.0)).collect();
        fft_3d_c2c_inplace(&mut data, density.shape);

        // Bin |ρ̂(k)|² by integer |k| magnitude
        let k_max = ((nx * nx + ny * ny + nz * nz) as f64).sqrt().ceil() as usize + 1;
        let mut power_sum = vec![0.0; k_max];
        let mut count = vec![0u64; k_max];

        for ix in 0..nx {
            let kx = if ix < nx / 2 {
                ix as i64
            } else {
                ix as i64 - nx as i64
            };
            for iy in 0..ny {
                let ky = if iy < ny / 2 {
                    iy as i64
                } else {
                    iy as i64 - ny as i64
                };
                for iz in 0..nz {
                    let kz = if iz < nz / 2 {
                        iz as i64
                    } else {
                        iz as i64 - nz as i64
                    };
                    let k_mag = ((kx * kx + ky * ky + kz * kz) as f64).sqrt();
                    let bin = k_mag.round() as usize;
                    if bin < k_max {
                        let c = data[ix * ny * nz + iy * nz + iz];
                        power_sum[bin] += c.norm_sqr();
                        count[bin] += 1;
                    }
                }
            }
        }

        // Average per bin, skip k=0 (DC)
        let mut result = Vec::new();
        for bin in 1..k_max {
            if count[bin] > 0 {
                result.push((bin as f64, power_sum[bin] / count[bin] as f64));
            }
        }
        result
    }

    /// Stability analysis: fit exponential growth rate to each k-mode from density history.
    ///
    /// For each k-bin, collects amplitude time series from per-step power spectra,
    /// then fits exponential via linear regression on ln(amplitude) vs t.
    /// Returns (k, growth_rate) pairs.
    pub fn growth_rates(density_history: &[DensityField], dt: f64) -> Vec<(f64, f64)> {
        if density_history.is_empty() {
            return Vec::new();
        }

        // Compute power spectrum at each timestep (independent per snapshot)
        let spectra: Vec<Vec<(f64, f64)>> = density_history
            .par_iter()
            .map(Self::power_spectrum)
            .collect();

        // Collect all k-bins from the first spectrum
        if spectra[0].is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        for (i, &(k, _)) in spectra[0].iter().enumerate() {
            // Collect ln(amplitude) time series for this k-bin
            let mut sum_t = 0.0;
            let mut sum_ln = 0.0;
            let mut sum_t2 = 0.0;
            let mut sum_tln = 0.0;
            let mut n = 0.0;

            for (step, spectrum) in spectra.iter().enumerate() {
                if i < spectrum.len() {
                    let amp = spectrum[i].1.sqrt();
                    if amp > 0.0 {
                        let t = step as f64 * dt;
                        let ln_amp = amp.ln();
                        sum_t += t;
                        sum_ln += ln_amp;
                        sum_t2 += t * t;
                        sum_tln += t * ln_amp;
                        n += 1.0;
                    }
                }
            }

            // Linear regression: ln(A) = gamma*t + c  =>  gamma = slope
            if n > 1.0 {
                let denom = n * sum_t2 - sum_t * sum_t;
                if denom.abs() > 1e-30 {
                    let gamma = (n * sum_tln - sum_t * sum_ln) / denom;
                    result.push((k, gamma));
                }
            }
        }
        result
    }
}

/// Gravitational field energy spectrum E(k) = |k|² |Φ̂(k)|² binned by |k|.
///
/// Returns (k, E(k)) pairs. This measures the gravitational energy per mode,
/// useful for tracking energy cascade and scale-dependent dynamics.
pub fn field_energy_spectrum(potential: &PotentialField, dx: [f64; 3]) -> Vec<(f64, f64)> {
    let [nx, ny, nz] = potential.shape;
    let lx = nx as f64 * dx[0];
    let ly = ny as f64 * dx[1];
    let lz = nz as f64 * dx[2];

    // 3D FFT of potential
    let mut data: Vec<Complex<f64>> = potential
        .data
        .iter()
        .map(|&r| Complex::new(r, 0.0))
        .collect();
    fft_3d_c2c_inplace(&mut data, potential.shape);

    // Bin |k|² |Φ̂(k)|² by |k|
    let k_max = ((nx * nx + ny * ny + nz * nz) as f64).sqrt().ceil() as usize + 1;
    let mut energy_sum = vec![0.0; k_max];
    let mut count = vec![0u64; k_max];

    use std::f64::consts::PI;
    for ix in 0..nx {
        let kx_idx = if ix < nx / 2 {
            ix as i64
        } else {
            ix as i64 - nx as i64
        };
        let kx = 2.0 * PI * kx_idx as f64 / lx;
        for iy in 0..ny {
            let ky_idx = if iy < ny / 2 {
                iy as i64
            } else {
                iy as i64 - ny as i64
            };
            let ky = 2.0 * PI * ky_idx as f64 / ly;
            for iz in 0..nz {
                let kz_idx = if iz < nz / 2 {
                    iz as i64
                } else {
                    iz as i64 - nz as i64
                };
                let kz = 2.0 * PI * kz_idx as f64 / lz;
                let k2 = kx * kx + ky * ky + kz * kz;
                let k_mag_int = ((kx_idx * kx_idx + ky_idx * ky_idx + kz_idx * kz_idx) as f64)
                    .sqrt()
                    .round() as usize;
                if k_mag_int > 0 && k_mag_int < k_max {
                    let c = data[ix * ny * nz + iy * nz + iz];
                    energy_sum[k_mag_int] += k2 * c.norm_sqr();
                    count[k_mag_int] += 1;
                }
            }
        }
    }

    let mut result = Vec::new();
    for bin in 1..k_max {
        if count[bin] > 0 {
            result.push((bin as f64, energy_sum[bin] / count[bin] as f64));
        }
    }
    result
}

/// Spherically-averaged potential power spectrum P(k) = <|Φ̂(k)|²>.
///
/// 3D FFT of the potential field, binned by wavenumber magnitude |k|.
/// Unlike [`field_energy_spectrum`], this does NOT weight by |k|².
/// Returns `(k, P(k))` pairs for non-empty bins.
pub fn potential_power_spectrum(potential: &PotentialField, dx: [f64; 3]) -> Vec<(f64, f64)> {
    let [nx, ny, nz] = potential.shape;
    let n = nx * ny * nz;
    if n == 0 {
        return vec![];
    }

    let mut buffer: Vec<Complex<f64>> = potential
        .data
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .collect();
    fft_3d_c2c_inplace(&mut buffer, potential.shape);

    // Bin |Φ̂(k)|² by wavenumber magnitude
    let dk = [
        2.0 * std::f64::consts::PI / (nx as f64 * dx[0]),
        2.0 * std::f64::consts::PI / (ny as f64 * dx[1]),
        2.0 * std::f64::consts::PI / (nz as f64 * dx[2]),
    ];
    let k_nyquist = [
        (nx / 2) as f64 * dk[0],
        (ny / 2) as f64 * dk[1],
        (nz / 2) as f64 * dk[2],
    ];
    let k_max = (k_nyquist[0].powi(2) + k_nyquist[1].powi(2) + k_nyquist[2].powi(2)).sqrt();
    let dk_bin = dk.iter().copied().fold(f64::INFINITY, f64::min);
    let n_bins = ((k_max / dk_bin) as usize + 1).max(1);

    let mut power_bins = vec![0.0f64; n_bins];
    let mut count_bins = vec![0usize; n_bins];
    let norm = 1.0 / (n as f64 * n as f64);

    for ikx in 0..nx {
        let kx = if ikx <= nx / 2 { ikx as f64 } else { ikx as f64 - nx as f64 } * dk[0];
        for iky in 0..ny {
            let ky = if iky <= ny / 2 { iky as f64 } else { iky as f64 - ny as f64 } * dk[1];
            for ikz in 0..nz {
                let kz = if ikz <= nz / 2 { ikz as f64 } else { ikz as f64 - nz as f64 } * dk[2];
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                if k_mag < 1e-14 {
                    continue;
                }
                let c = buffer[ikx * ny * nz + iky * nz + ikz];
                let power = (c.re * c.re + c.im * c.im) * norm;
                let bin = (k_mag / dk_bin).round() as usize;
                if bin < n_bins {
                    power_bins[bin] += power;
                    count_bins[bin] += 1;
                }
            }
        }
    }

    power_bins
        .iter()
        .zip(count_bins.iter())
        .enumerate()
        .filter(|&(_, (_, &c))| c > 0)
        .map(|(i, (&p, &c))| {
            let k = (i as f64 + 0.5) * dk_bin;
            (k, p / c as f64)
        })
        .collect()
}

/// L2 norm of the Poisson residual: ||∇²Φ − 4πGρ||₂.
///
/// Uses a 7-point finite-difference stencil on interior cells. Returns the
/// RMS residual, useful for verifying Poisson solver accuracy.
pub fn poisson_residual_l2(
    density: &DensityField,
    potential: &PotentialField,
    g: f64,
    dx: [f64; 3],
) -> f64 {
    let [nx, ny, nz] = density.shape;
    let four_pi_g = 4.0 * std::f64::consts::PI * g;
    let inv_dx2 = [
        1.0 / (dx[0] * dx[0]),
        1.0 / (dx[1] * dx[1]),
        1.0 / (dx[2] * dx[2]),
    ];
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for ix in 1..nx.saturating_sub(1) {
        for iy in 1..ny.saturating_sub(1) {
            for iz in 1..nz.saturating_sub(1) {
                let idx = ix * ny * nz + iy * nz + iz;
                let phi = potential.data[idx];

                let lap_x = (potential.data[(ix + 1) * ny * nz + iy * nz + iz]
                    + potential.data[(ix - 1) * ny * nz + iy * nz + iz]
                    - 2.0 * phi)
                    * inv_dx2[0];
                let lap_y = (potential.data[ix * ny * nz + (iy + 1) * nz + iz]
                    + potential.data[ix * ny * nz + (iy - 1) * nz + iz]
                    - 2.0 * phi)
                    * inv_dx2[1];
                let lap_z = (potential.data[ix * ny * nz + iy * nz + (iz + 1)]
                    + potential.data[ix * ny * nz + iy * nz + (iz - 1)]
                    - 2.0 * phi)
                    * inv_dx2[2];
                let laplacian = lap_x + lap_y + lap_z;

                let rhs = four_pi_g * density.data[idx];
                let residual = laplacian - rhs;
                sum_sq += residual * residual;
                count += 1;
            }
        }
    }

    if count > 0 {
        (sum_sq / count as f64).sqrt()
    } else {
        0.0
    }
}
