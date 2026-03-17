//! Phase-space structure diagnostics: stream count, caustic surfaces, power spectrum,
//! growth rates.

use super::super::{phasespace::PhaseSpaceRepr, types::*};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

/// Phase-space structure diagnostic outputs.
pub struct PhaseSpaceDiagnostics {
    pub stream_count: StreamCountField,
    pub local_entropy: Vec<f64>,
    pub caustic_cells: Vec<[usize; 3]>,
}

impl PhaseSpaceDiagnostics {
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

        // 3D C2C FFT via sequential 1D FFTs along each axis
        let mut planner = FftPlanner::new();

        // Convert to complex
        let mut data: Vec<Complex<f64>> =
            density.data.iter().map(|&r| Complex::new(r, 0.0)).collect();

        // FFT along z (axis 2, contiguous)
        let fft_z = planner.plan_fft_forward(nz);
        for ix in 0..nx {
            for iy in 0..ny {
                let offset = ix * ny * nz + iy * nz;
                fft_z.process(&mut data[offset..offset + nz]);
            }
        }

        // FFT along y (axis 1)
        let fft_y = planner.plan_fft_forward(ny);
        let mut buf = vec![Complex::new(0.0, 0.0); ny];
        for ix in 0..nx {
            for iz in 0..nz {
                for iy in 0..ny {
                    buf[iy] = data[ix * ny * nz + iy * nz + iz];
                }
                fft_y.process(&mut buf);
                for iy in 0..ny {
                    data[ix * ny * nz + iy * nz + iz] = buf[iy];
                }
            }
        }

        // FFT along x (axis 0)
        let fft_x = planner.plan_fft_forward(nx);
        let mut buf = vec![Complex::new(0.0, 0.0); nx];
        for iy in 0..ny {
            for iz in 0..nz {
                for ix in 0..nx {
                    buf[ix] = data[ix * ny * nz + iy * nz + iz];
                }
                fft_x.process(&mut buf);
                for ix in 0..nx {
                    data[ix * ny * nz + iy * nz + iz] = buf[ix];
                }
            }
        }

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
        let spectra: Vec<Vec<(f64, f64)>> =
            density_history.par_iter().map(Self::power_spectrum).collect();

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
