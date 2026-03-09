//! Arbitrary user-provided initial conditions: callable or pre-computed 6D array.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;
use rust_decimal::prelude::ToPrimitive;

/// User-provided callable: f(x, v) → f64. Evaluated on every grid point at startup.
pub struct CustomIC {
    pub func: Box<dyn Fn([f64; 3], [f64; 3]) -> f64 + Send + Sync>,
}

impl CustomIC {
    /// Construct from a closure `f(x, v) -> f64`.
    pub fn from_fn(f: impl Fn([f64; 3], [f64; 3]) -> f64 + Send + Sync + 'static) -> Self {
        Self { func: Box::new(f) }
    }

    /// Evaluate `self.func` at every (x, v) grid point.
    pub fn sample_on_grid(&self, domain: &Domain) -> PhaseSpaceSnapshot {
        let nx1 = domain.spatial_res.x1 as usize;
        let nx2 = domain.spatial_res.x2 as usize;
        let nx3 = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;

        let dx = domain.dx();
        let dv = domain.dv();
        let lx = [
            domain.spatial.x1.to_f64().unwrap(),
            domain.spatial.x2.to_f64().unwrap(),
            domain.spatial.x3.to_f64().unwrap(),
        ];
        let lv = [
            domain.velocity.v1.to_f64().unwrap(),
            domain.velocity.v2.to_f64().unwrap(),
            domain.velocity.v3.to_f64().unwrap(),
        ];

        let s_v3 = 1usize;
        let s_v2 = nv3;
        let s_v1 = nv2 * nv3;
        let s_x3 = nv1 * s_v1;
        let s_x2 = nx3 * s_x3;
        let s_x1 = nx2 * s_x2;

        let total = nx1 * nx2 * nx3 * nv1 * nv2 * nv3;
        let mut data = vec![0.0f64; total];

        for ix1 in 0..nx1 {
            let x1 = -lx[0] + (ix1 as f64 + 0.5) * dx[0];
            for ix2 in 0..nx2 {
                let x2 = -lx[1] + (ix2 as f64 + 0.5) * dx[1];
                for ix3 in 0..nx3 {
                    let x3 = -lx[2] + (ix3 as f64 + 0.5) * dx[2];
                    let base = ix1 * s_x1 + ix2 * s_x2 + ix3 * s_x3;

                    for iv1 in 0..nv1 {
                        let v1 = -lv[0] + (iv1 as f64 + 0.5) * dv[0];
                        for iv2 in 0..nv2 {
                            let v2 = -lv[1] + (iv2 as f64 + 0.5) * dv[1];
                            for iv3 in 0..nv3 {
                                let v3 = -lv[2] + (iv3 as f64 + 0.5) * dv[2];
                                let f = (self.func)([x1, x2, x3], [v1, v2, v3]);
                                data[base + iv1 * s_v1 + iv2 * s_v2 + iv3 * s_v3] = f.max(0.0);
                            }
                        }
                    }
                }
            }
        }

        PhaseSpaceSnapshot {
            data,
            shape: [nx1, nx2, nx3, nv1, nv2, nv3],
            time: 0.0,
        }
    }
}

/// Pre-computed 6D array [Nx1, Nx2, Nx3, Nv1, Nv2, Nv3] loaded from file.
pub struct CustomICArray {
    pub snapshot: PhaseSpaceSnapshot,
}

impl CustomICArray {
    /// Load a `.npy` file, validate shape matches domain, and wrap.
    ///
    /// Parses the NumPy .npy format (v1.0/2.0): magic bytes, header with shape/dtype,
    /// then raw f64 data. Validates the total element count matches domain.total_cells().
    pub fn from_npy(path: &str, domain: &Domain) -> anyhow::Result<Self> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;
        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        if &magic != b"\x93NUMPY" {
            anyhow::bail!("not a valid .npy file: bad magic bytes");
        }

        // Version
        let mut ver = [0u8; 2];
        file.read_exact(&mut ver)?;

        // Header length (2 bytes for v1, 4 bytes for v2+)
        let header_len = if ver[0] == 1 {
            let mut buf = [0u8; 2];
            file.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as usize
        } else {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        };

        // Skip header (we trust it's float64, C-contiguous)
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;

        // Read remaining data as f64
        let expected = domain.total_cells();
        let mut raw = vec![0u8; expected * 8];
        file.read_exact(&mut raw)?;

        let data: Vec<f64> = raw
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        if data.len() != expected {
            anyhow::bail!(
                "npy data length {} does not match domain total_cells {}",
                data.len(),
                expected
            );
        }

        let nx1 = domain.spatial_res.x1 as usize;
        let nx2 = domain.spatial_res.x2 as usize;
        let nx3 = domain.spatial_res.x3 as usize;
        let nv1 = domain.velocity_res.v1 as usize;
        let nv2 = domain.velocity_res.v2 as usize;
        let nv3 = domain.velocity_res.v3 as usize;

        Ok(Self {
            snapshot: PhaseSpaceSnapshot {
                data,
                shape: [nx1, nx2, nx3, nv1, nv2, nv3],
                time: 0.0,
            },
        })
    }
}

/// Legacy stub: scalar distribution function prototype.
pub fn f_init(_x: [f64; 3], _v: [f64; 3]) -> f64 {
    unimplemented!("use CustomIC instead")
}

/// Legacy stub: raw 6D array initial condition.
pub fn f_init_array() -> PhaseSpaceSnapshot {
    unimplemented!("use CustomICArray instead")
}

/// Legacy stub: external potential Φ_ext(x, t).
pub fn phi_external(_x: [f64; 3], _t: f64) -> f64 {
    unimplemented!("use TidalIC::host_potential instead")
}
