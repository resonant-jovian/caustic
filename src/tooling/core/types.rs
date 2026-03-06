//! Shared physics field types used across all traits and implementations throughout caustic.

/// 3D scalar field ρ(x); flat `Vec<f64>` with `(nx1, nx2, nx3)` shape.
pub struct DensityField {
    pub data: Vec<f64>,
    pub shape: [usize; 3],
}

/// 3D scalar potential field Φ(x); same layout as `DensityField`.
pub struct PotentialField {
    pub data: Vec<f64>,
    pub shape: [usize; 3],
}

/// 3D vector field g = −∇Φ; three flat `Vec<f64>` for gx, gy, gz components.
pub struct AccelerationField {
    pub gx: Vec<f64>,
    pub gy: Vec<f64>,
    pub gz: Vec<f64>,
    pub shape: [usize; 3],
}

/// Displacement Δx = v·Δt for the spatial drift sub-step.
pub struct DisplacementField {
    pub dx: Vec<f64>,
    pub dy: Vec<f64>,
    pub dz: Vec<f64>,
    pub shape: [usize; 3],
}

/// General velocity moment tensor of arbitrary rank.
pub struct Tensor {
    pub data: Vec<f64>,
    pub rank: usize,
    pub shape: Vec<usize>,
}

/// Full 6D snapshot f(x,v) at one instant; flat `Vec<f64>` with shape
/// `(nx1, nx2, nx3, nv1, nv2, nv3)`. Only stored at checkpoint intervals.
pub struct PhaseSpaceSnapshot {
    pub data: Vec<f64>,
    pub shape: [usize; 6],
    pub time: f64,
}

/// Number of distinct velocity streams n_s(x) at each spatial point.
/// Nonzero where the sheet has folded (caustic formation).
pub struct StreamCountField {
    pub data: Vec<u32>,
    pub shape: [usize; 3],
}
