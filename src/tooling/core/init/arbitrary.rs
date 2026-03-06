//! Arbitrary user-provided initial conditions: callable or pre-computed 6D array.

use super::super::types::PhaseSpaceSnapshot;
use super::domain::Domain;

/// User-provided callable: f(x, v) → f64. Evaluated on every grid point at startup.
pub struct CustomIC {
    pub func: Box<dyn Fn([f64; 3], [f64; 3]) -> f64 + Send + Sync>,
}

impl CustomIC {
    /// Construct from a closure `f(x, v) -> f64`.
    pub fn from_fn(f: impl Fn([f64; 3], [f64; 3]) -> f64 + Send + Sync + 'static) -> Self {
        todo!()
    }

    /// Evaluate `self.func` at every (x, v) grid point in parallel.
    pub fn sample_on_grid(&self, domain: &Domain) -> PhaseSpaceSnapshot {
        todo!("evaluate self.func at every (x,v) grid point in parallel")
    }
}

/// Pre-computed 6D array [Nx1, Nx2, Nx3, Nv1, Nv2, Nv3] loaded from file.
pub struct CustomICArray {
    pub snapshot: PhaseSpaceSnapshot,
}

impl CustomICArray {
    /// Load a `.npy` file, validate shape matches domain, and wrap.
    pub fn from_npy(path: &str, domain: &Domain) -> anyhow::Result<Self> {
        todo!("load .npy file, validate shape matches domain")
    }
}

/// Legacy stub: scalar distribution function prototype.
pub fn f_init(x: [f64; 3], v: [f64; 3]) -> f64 {
    todo!()
}

/// Legacy stub: raw 6D array initial condition.
pub fn f_init_array() -> PhaseSpaceSnapshot {
    todo!()
}

/// Legacy stub: external potential Φ_ext(x, t).
pub fn phi_external(x: [f64; 3], t: f64) -> f64 {
    todo!()
}
