//! Shared physics field types used across all traits and implementations throughout caustic.

/// 3D scalar field ρ(x); flat `Vec<f64>` with `(nx1, nx2, nx3)` shape.
#[derive(Clone)]
pub struct DensityField {
    /// Flat row-major density values. Index = ix*ny*nz + iy*nz + iz.
    pub data: Vec<f64>,
    /// Grid dimensions `[nx, ny, nz]` (number of spatial cells per axis).
    pub shape: [usize; 3],
}

/// 3D scalar potential field Φ(x); same layout as `DensityField`.
#[derive(Clone)]
pub struct PotentialField {
    /// Flat row-major potential values. Index = ix*ny*nz + iy*nz + iz.
    pub data: Vec<f64>,
    /// Grid dimensions `[nx, ny, nz]` (number of spatial cells per axis).
    pub shape: [usize; 3],
}

/// 3D vector field g = −∇Φ; three flat `Vec<f64>` for gx, gy, gz components.
pub struct AccelerationField {
    /// x-component of acceleration. Flat row-major, index = ix*ny*nz + iy*nz + iz.
    pub gx: Vec<f64>,
    /// y-component of acceleration. Same layout as `gx`.
    pub gy: Vec<f64>,
    /// z-component of acceleration. Same layout as `gx`.
    pub gz: Vec<f64>,
    /// Grid dimensions `[nx, ny, nz]` (number of spatial cells per axis).
    pub shape: [usize; 3],
}

/// Displacement Δx = v·Δt for the spatial drift sub-step.
pub struct DisplacementField {
    /// x-displacement. Flat row-major, index = ix*ny*nz + iy*nz + iz.
    pub dx: Vec<f64>,
    /// y-displacement. Same layout as `dx`.
    pub dy: Vec<f64>,
    /// z-displacement. Same layout as `dx`.
    pub dz: Vec<f64>,
    /// Grid dimensions `[nx, ny, nz]` (number of spatial cells per axis).
    pub shape: [usize; 3],
}

/// General velocity moment tensor of arbitrary rank.
pub struct Tensor {
    /// Flat row-major tensor components. For a rank-k tensor in 3D, length is 3^k.
    pub data: Vec<f64>,
    /// Tensor rank (0 = scalar, 1 = vector, 2 = matrix, ...).
    pub rank: usize,
    /// Size of each tensor dimension (e.g. `[3, 3]` for a rank-2 tensor in 3D).
    pub shape: Vec<usize>,
}

/// Full 6D snapshot f(x,v) at one instant; flat `Vec<f64>` with shape
/// `(nx1, nx2, nx3, nv1, nv2, nv3)`. Only stored at checkpoint intervals.
pub struct PhaseSpaceSnapshot {
    /// Flat row-major 6D distribution values. Index =
    /// ix*(ny*nz*nvx*nvy*nvz) + iy*(nz*nvx*nvy*nvz) + iz*(nvx*nvy*nvz)
    /// + ivx*(nvy*nvz) + ivy*nvz + ivz.
    pub data: Vec<f64>,
    /// Grid dimensions `[nx, ny, nz, nvx, nvy, nvz]` (spatial then velocity axes).
    pub shape: [usize; 6],
    /// Simulation time at which this snapshot was recorded.
    pub time: f64,
}

/// Number of distinct velocity streams n_s(x) at each spatial point.
/// Nonzero where the sheet has folded (caustic formation).
pub struct StreamCountField {
    /// Flat row-major stream counts. Index = ix*ny*nz + iy*nz + iz.
    pub data: Vec<u32>,
    /// Grid dimensions `[nx, ny, nz]` (number of spatial cells per axis).
    pub shape: [usize; 3],
}
