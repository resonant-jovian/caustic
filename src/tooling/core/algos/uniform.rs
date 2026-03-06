//! Brute-force uniform 6D grid. Memory O(N⁶). Simple and correct; primary reference
//! implementation for validation.

use rust_decimal::prelude::ToPrimitive;
use super::super::{
    types::*,
    phasespace::PhaseSpaceRepr,
    init::domain::{Domain, SpatialBoundType, VelocityBoundType},
};
use super::lagrangian::sl_shift_1d;

/// Stores f on a uniform (Nx1×Nx2×Nx3×Nv1×Nv2×Nv3) grid as a flat `Vec<f64>`.
/// Index order: x1 fastest-changing outer, v3 fastest-changing inner (row-major).
pub struct UniformGrid6D {
    pub data: Vec<f64>,
    pub domain: Domain,
}

impl UniformGrid6D {
    /// Allocate Nx³ × Nv³ floats, zero-initialised.
    pub fn new(domain: Domain) -> Self {
        let n = domain.total_cells();
        Self { data: vec![0.0; n], domain }
    }

    pub fn from_snapshot(snap: PhaseSpaceSnapshot, domain: Domain) -> Self {
        assert_eq!(snap.data.len(), domain.total_cells(),
            "snapshot size mismatch: {} vs {}", snap.data.len(), domain.total_cells());
        Self { data: snap.data, domain }
    }

    /// Linear index into flat Vec from (ix1, ix2, ix3, iv1, iv2, iv3) — row-major 6D.
    pub fn index(&self, ix: [usize; 3], iv: [usize; 3]) -> usize {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let s_v3 = 1;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;
        ix[0] * s_x1 + ix[1] * s_x2 + ix[2] * s_x3
            + iv[0] * s_v1 + iv[1] * s_v2 + iv[2] * s_v3
    }

    pub(crate) fn sizes(&self) -> [usize; 6] {
        let d = &self.domain;
        [
            d.spatial_res.x1 as usize, d.spatial_res.x2 as usize, d.spatial_res.x3 as usize,
            d.velocity_res.v1 as usize, d.velocity_res.v2 as usize, d.velocity_res.v3 as usize,
        ]
    }

    fn lx(&self) -> [f64; 3] {
        [
            self.domain.spatial.x1.to_f64().unwrap(),
            self.domain.spatial.x2.to_f64().unwrap(),
            self.domain.spatial.x3.to_f64().unwrap(),
        ]
    }

    fn lv(&self) -> [f64; 3] {
        [
            self.domain.velocity.v1.to_f64().unwrap(),
            self.domain.velocity.v2.to_f64().unwrap(),
            self.domain.velocity.v3.to_f64().unwrap(),
        ]
    }
}

impl PhaseSpaceRepr for UniformGrid6D {
    fn compute_density(&self) -> DensityField {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.domain.dv();
        let dv3 = dv[0] * dv[1] * dv[2];

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let data: Vec<f64> = (0..nx1 * nx2 * nx3).map(|si| {
            let ix3 = si % nx3;
            let ix2 = (si / nx3) % nx2;
            let ix1 = si / (nx2 * nx3);
            let base = ix1 * s_x1 + ix2 * s_x2 + ix3 * s_x3;
            let sum: f64 = (0..nv1 * nv2 * nv3).map(|vi| {
                let iv3 = vi % nv3;
                let iv2 = (vi / nv3) % nv2;
                let iv1 = vi / (nv2 * nv3);
                self.data[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3]
            }).sum();
            sum * dv3
        }).collect();

        DensityField { data, shape: [nx1, nx2, nx3] }
    }

    fn advect_x(&mut self, _displacement: &DisplacementField, dt: f64) {
        // Note: displacement field unused for 6D uniform grid;
        // velocity-cell displacements are computed internally from the velocity grid.
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.lx();
        let lv = self.lv();
        let periodic = matches!(self.domain.spatial_bc, SpatialBoundType::Periodic);

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let flat_idx = |ix: [usize; 3], iv: [usize; 3]| -> usize {
            ix[0] * s_x1 + ix[1] * s_x2 + ix[2] * s_x3
                + iv[0] * s_v1 + iv[1] * s_v2 + iv[2] * s_v3
        };

        let mut buf = self.data.clone();

        for iv1 in 0..nv1 {
            for iv2 in 0..nv2 {
                for iv3 in 0..nv3 {
                    let iv = [iv1, iv2, iv3];
                    let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                    let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                    let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                    let disp = [vx * dt, vy * dt, vz * dt];

                    // Shift along x1
                    for ix2 in 0..nx2 {
                        for ix3 in 0..nx3 {
                            let line: Vec<f64> = (0..nx1)
                                .map(|ix1| buf[flat_idx([ix1, ix2, ix3], iv)])
                                .collect();
                            let shifted = sl_shift_1d(&line, disp[0], dx[0], nx1, lx[0], periodic);
                            for ix1 in 0..nx1 {
                                buf[flat_idx([ix1, ix2, ix3], iv)] = shifted[ix1];
                            }
                        }
                    }

                    // Shift along x2
                    for ix1 in 0..nx1 {
                        for ix3 in 0..nx3 {
                            let line: Vec<f64> = (0..nx2)
                                .map(|ix2| buf[flat_idx([ix1, ix2, ix3], iv)])
                                .collect();
                            let shifted = sl_shift_1d(&line, disp[1], dx[1], nx2, lx[1], periodic);
                            for ix2 in 0..nx2 {
                                buf[flat_idx([ix1, ix2, ix3], iv)] = shifted[ix2];
                            }
                        }
                    }

                    // Shift along x3
                    for ix1 in 0..nx1 {
                        for ix2 in 0..nx2 {
                            let line: Vec<f64> = (0..nx3)
                                .map(|ix3| buf[flat_idx([ix1, ix2, ix3], iv)])
                                .collect();
                            let shifted = sl_shift_1d(&line, disp[2], dx[2], nx3, lx[2], periodic);
                            for ix3 in 0..nx3 {
                                buf[flat_idx([ix1, ix2, ix3], iv)] = shifted[ix3];
                            }
                        }
                    }
                }
            }
        }

        self.data = buf;
    }

    fn advect_v(&mut self, acceleration: &AccelerationField, dt: f64) {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.domain.dv();
        let lv = self.lv();
        // Open BC absorbs particles leaving the velocity box; Truncated reflects
        let periodic_v = matches!(self.domain.velocity_bc, VelocityBoundType::Truncated);

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let flat_idx = |ix: [usize; 3], iv: [usize; 3]| -> usize {
            ix[0] * s_x1 + ix[1] * s_x2 + ix[2] * s_x3
                + iv[0] * s_v1 + iv[1] * s_v2 + iv[2] * s_v3
        };

        let mut buf = self.data.clone();

        for ix1 in 0..nx1 {
            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    let ix = [ix1, ix2, ix3];
                    let flat_ix = ix1 * nx2 * nx3 + ix2 * nx3 + ix3;
                    let ax = acceleration.gx[flat_ix];
                    let ay = acceleration.gy[flat_ix];
                    let az = acceleration.gz[flat_ix];
                    let disp = [ax * dt, ay * dt, az * dt];

                    // Shift along v1
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            let line: Vec<f64> = (0..nv1)
                                .map(|iv1| buf[flat_idx(ix, [iv1, iv2, iv3])])
                                .collect();
                            let shifted = sl_shift_1d(&line, disp[0], dv[0], nv1, lv[0], periodic_v);
                            for iv1 in 0..nv1 {
                                buf[flat_idx(ix, [iv1, iv2, iv3])] = shifted[iv1];
                            }
                        }
                    }

                    // Shift along v2
                    for iv1 in 0..nv1 {
                        for iv3 in 0..nv3 {
                            let line: Vec<f64> = (0..nv2)
                                .map(|iv2| buf[flat_idx(ix, [iv1, iv2, iv3])])
                                .collect();
                            let shifted = sl_shift_1d(&line, disp[1], dv[1], nv2, lv[1], periodic_v);
                            for iv2 in 0..nv2 {
                                buf[flat_idx(ix, [iv1, iv2, iv3])] = shifted[iv2];
                            }
                        }
                    }

                    // Shift along v3
                    for iv1 in 0..nv1 {
                        for iv2 in 0..nv2 {
                            let line: Vec<f64> = (0..nv3)
                                .map(|iv3| buf[flat_idx(ix, [iv1, iv2, iv3])])
                                .collect();
                            let shifted = sl_shift_1d(&line, disp[2], dv[2], nv3, lv[2], periodic_v);
                            for iv3 in 0..nv3 {
                                buf[flat_idx(ix, [iv1, iv2, iv3])] = shifted[iv3];
                            }
                        }
                    }
                }
            }
        }

        self.data = buf;
    }

    fn moment(&self, position: &[f64; 3], order: usize) -> Tensor {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let lx = self.lx();
        let lv = self.lv();
        let dv3 = dv[0] * dv[1] * dv[2];

        let ix1 = ((position[0] + lx[0]) / dx[0]).floor().clamp(0.0, (nx1 - 1) as f64) as usize;
        let ix2 = ((position[1] + lx[1]) / dx[1]).floor().clamp(0.0, (nx2 - 1) as f64) as usize;
        let ix3 = ((position[2] + lx[2]) / dx[2]).floor().clamp(0.0, (nx3 - 1) as f64) as usize;

        match order {
            0 => {
                let sum: f64 = (0..nv1 * nv2 * nv3).map(|vi| {
                    let iv3 = vi % nv3;
                    let iv2 = (vi / nv3) % nv2;
                    let iv1 = vi / (nv2 * nv3);
                    self.data[self.index([ix1, ix2, ix3], [iv1, iv2, iv3])]
                }).sum::<f64>() * dv3;
                Tensor { data: vec![sum], rank: 0, shape: vec![] }
            }
            1 => {
                let mut vbar = [0.0f64; 3];
                let mut rho = 0.0f64;
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            let f = self.data[self.index([ix1, ix2, ix3], [iv1, iv2, iv3])];
                            let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                            let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                            vbar[0] += f * vx;
                            vbar[1] += f * vy;
                            vbar[2] += f * vz;
                            rho += f;
                        }
                    }
                }
                rho *= dv3;
                let scale = if rho > 1e-30 { dv3 / rho } else { 0.0 };
                Tensor {
                    data: vec![vbar[0] * scale, vbar[1] * scale, vbar[2] * scale],
                    rank: 1,
                    shape: vec![3],
                }
            }
            2 => {
                let mut m2 = [0.0f64; 9];
                for iv1 in 0..nv1 {
                    for iv2 in 0..nv2 {
                        for iv3 in 0..nv3 {
                            let f = self.data[self.index([ix1, ix2, ix3], [iv1, iv2, iv3])];
                            let v = [
                                -lv[0] + (iv1 as f64 + 0.5) * dv[0],
                                -lv[1] + (iv2 as f64 + 0.5) * dv[1],
                                -lv[2] + (iv3 as f64 + 0.5) * dv[2],
                            ];
                            for i in 0..3 {
                                for j in 0..3 {
                                    m2[i * 3 + j] += f * v[i] * v[j];
                                }
                            }
                        }
                    }
                }
                Tensor {
                    data: m2.iter().map(|&x| x * dv3).collect(),
                    rank: 2,
                    shape: vec![3, 3],
                }
            }
            _ => Tensor { data: vec![], rank: order, shape: vec![] },
        }
    }

    fn total_mass(&self) -> f64 {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        self.data.iter().sum::<f64>() * dx3 * dv3
    }

    fn casimir_c2(&self) -> f64 {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        self.data.iter().map(|&f| f * f).sum::<f64>() * dx3 * dv3
    }

    fn entropy(&self) -> f64 {
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        self.data.iter()
            .filter(|&&f| f > 0.0)
            .map(|&f| -f * f.ln())
            .sum::<f64>() * dx3 * dv3
    }

    fn stream_count(&self) -> StreamCountField {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dv = self.domain.dv();
        let dv23 = dv[1] * dv[2];

        let mut out = vec![0u32; nx1 * nx2 * nx3];

        for ix1 in 0..nx1 {
            for ix2 in 0..nx2 {
                for ix3 in 0..nx3 {
                    // Marginal f_1(v1|x) = sum_{v2,v3} f(v|x) * dv2 * dv3
                    let marginal: Vec<f64> = (0..nv1).map(|iv1| {
                        (0..nv2 * nv3).map(|vi23| {
                            let iv3 = vi23 % nv3;
                            let iv2 = vi23 / nv3;
                            self.data[self.index([ix1, ix2, ix3], [iv1, iv2, iv3])]
                        }).sum::<f64>() * dv23
                    }).collect();

                    // Count peaks: cells where f[i-1] < f[i] > f[i+1]
                    let mut peaks = 0u32;
                    for i in 1..nv1.saturating_sub(1) {
                        if marginal[i] > marginal[i - 1] && marginal[i] > marginal[i + 1] {
                            peaks += 1;
                        }
                    }
                    out[ix1 * nx2 * nx3 + ix2 * nx3 + ix3] = peaks;
                }
            }
        }

        StreamCountField { data: out, shape: [nx1, nx2, nx3] }
    }

    fn velocity_distribution(&self, position: &[f64; 3]) -> Vec<f64> {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.domain.dx();
        let lx = self.lx();

        let ix1 = ((position[0] + lx[0]) / dx[0]).floor().clamp(0.0, (nx1 - 1) as f64) as usize;
        let ix2 = ((position[1] + lx[1]) / dx[1]).floor().clamp(0.0, (nx2 - 1) as f64) as usize;
        let ix3 = ((position[2] + lx[2]) / dx[2]).floor().clamp(0.0, (nx3 - 1) as f64) as usize;

        (0..nv1 * nv2 * nv3).map(|vi| {
            let iv3 = vi % nv3;
            let iv2 = (vi / nv3) % nv2;
            let iv1 = vi / (nv2 * nv3);
            self.data[self.index([ix1, ix2, ix3], [iv1, iv2, iv3])]
        }).collect()
    }

    fn total_kinetic_energy(&self) -> f64 {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        let dx = self.domain.dx();
        let dv = self.domain.dv();
        let dx3 = dx[0] * dx[1] * dx[2];
        let dv3 = dv[0] * dv[1] * dv[2];
        let lv = self.lv();

        let mut t = 0.0f64;
        for iv1 in 0..nv1 {
            let vx = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
            for iv2 in 0..nv2 {
                let vy = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                for iv3 in 0..nv3 {
                    let vz = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                    let v2 = vx * vx + vy * vy + vz * vz;
                    for ix1 in 0..nx1 {
                        for ix2 in 0..nx2 {
                            for ix3 in 0..nx3 {
                                t += self.data[self.index([ix1, ix2, ix3], [iv1, iv2, iv3])] * v2;
                            }
                        }
                    }
                }
            }
        }
        0.5 * t * dx3 * dv3
    }

    fn to_snapshot(&self, time: f64) -> PhaseSpaceSnapshot {
        let [nx1, nx2, nx3, nv1, nv2, nv3] = self.sizes();
        PhaseSpaceSnapshot {
            data: self.data.clone(),
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time,
        }
    }
}
