//! Semi-Lagrangian advector. Traces characteristics backwards from each grid point and
//! interpolates. Eliminates CFL constraint; Δt limited only by accuracy requirements.

use super::super::{advecator::Advector, phasespace::PhaseSpaceRepr, types::*};

/// Semi-Lagrangian advector with cubic spline interpolation.
pub struct SemiLagrangian {
    pub interpolation_order: usize,
}

impl Default for SemiLagrangian {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiLagrangian {
    pub fn new() -> Self {
        Self {
            interpolation_order: 3,
        }
    }
}

impl Advector for SemiLagrangian {
    fn drift(&self, repr: &mut dyn PhaseSpaceRepr, dt: f64) {
        // UniformGrid6D ignores the displacement field; it computes shifts from its velocity grid.
        let dummy = DisplacementField {
            dx: vec![],
            dy: vec![],
            dz: vec![],
            shape: [0, 0, 0],
        };
        repr.advect_x(&dummy, dt);
    }

    fn kick(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64) {
        repr.advect_v(acceleration, dt);
    }

    fn step(&self, repr: &mut dyn PhaseSpaceRepr, acceleration: &AccelerationField, dt: f64) {
        self.kick(repr, acceleration, dt);
        self.drift(repr, dt);
    }
}

/// 4-point Catmull-Rom cubic spline interpolation at fractional position `x` in
/// periodic array of length `n`. `x` is a fractional cell index in `[0, n)`.
#[inline]
pub fn cubic_spline_interpolate(data: &[f64], x: f64, n: usize) -> f64 {
    let i = x.floor() as isize;
    let t = x - x.floor();
    let wrap = |j: isize| (j.rem_euclid(n as isize)) as usize;
    let p0 = data[wrap(i - 1)];
    let p1 = data[wrap(i)];
    let p2 = data[wrap(i + 1)];
    let p3 = data[wrap(i + 2)];
    let a0 = p1;
    let a1 = 0.5 * (-p0 + p2);
    let a2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let a3 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    a0 + t * (a1 + t * (a2 + t * a3))
}

/// 4-point Catmull-Rom cubic spline interpolation with clamped (open) boundary.
#[inline]
fn cubic_spline_interpolate_open(data: &[f64], x: f64, n: usize) -> f64 {
    let i = x.floor() as isize;
    let t = x - x.floor();
    let clamp = |j: isize| j.clamp(0, n as isize - 1) as usize;
    let p0 = data[clamp(i - 1)];
    let p1 = data[clamp(i)];
    let p2 = data[clamp(i + 1)];
    let p3 = data[clamp(i + 2)];
    let a0 = p1;
    let a1 = 0.5 * (-p0 + p2);
    let a2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let a3 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    a0 + t * (a1 + t * (a2 + t * a3))
}

/// 1D semi-Lagrangian shift of a grid line.
///
/// - `data`: array of `n` values on a periodic/open 1D grid spanning `[−l, l]`
/// - `disp`: displacement (physical units) — the whole line shifts by this amount
/// - `cell_size`: grid spacing (= 2l/n)
/// - `n`: number of cells
/// - `l`: half-extent of the domain
/// - `periodic`: true → wrap-around; false → absorbing (out-of-bounds → 0)
pub fn sl_shift_1d(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    periodic: bool,
) -> Vec<f64> {
    let mut out = vec![0.0; n];
    sl_shift_1d_into(data, disp, cell_size, n, l, periodic, &mut out);
    out
}

/// Evaluate the semi-Lagrangian shift for a single output cell index.
#[inline]
fn sl_shift_1d_at(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    periodic: bool,
    i: usize,
) -> f64 {
    let center_i = -l + (i as f64 + 0.5) * cell_size;
    let departure_phys = center_i - disp;
    let departure_idx = (departure_phys + l) / cell_size - 0.5;
    if periodic {
        cubic_spline_interpolate(data, departure_idx, n)
    } else if departure_idx < -0.5 || departure_idx >= n as f64 - 0.5 {
        0.0
    } else {
        let clamped = departure_idx.clamp(0.0, n as f64 - 1.0 - 1e-10);
        cubic_spline_interpolate_open(data, clamped, n)
    }
}

/// Fast sliding-window 1D semi-Lagrangian shift for periodic boundaries.
///
/// For a uniform displacement, the fractional departure offset `t` is identical
/// for all output points and the stencil slides by exactly one cell per iteration.
/// Catmull-Rom weights are computed once, and only 1 new data load per output
/// point (amortized) instead of 4.
#[inline]
fn sl_shift_1d_into_fast(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    _l: f64,
    out: &mut [f64],
) {
    if n == 0 {
        return;
    }
    // departure_idx for output point i = i - disp / cell_size
    let dep0 = -(disp / cell_size);
    let t = dep0 - dep0.floor();

    // Precompute Catmull-Rom weights (constant for all output points)
    let t2 = t * t;
    let t3 = t2 * t;
    let w0 = -0.5 * t + t2 - 0.5 * t3;
    let w1 = 1.0 - 2.5 * t2 + 1.5 * t3;
    let w2 = 0.5 * t + 2.0 * t2 - 1.5 * t3;
    let w3 = -0.5 * t2 + 0.5 * t3;

    let n_isize = n as isize;
    let i0 = dep0.floor() as isize;
    let wrap = |j: isize| (j.rem_euclid(n_isize)) as usize;

    // Initialize sliding window
    let mut p0 = data[wrap(i0 - 1)];
    let mut p1 = data[wrap(i0)];
    let mut p2 = data[wrap(i0 + 1)];
    let mut p3 = data[wrap(i0 + 2)];
    out[0] = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;

    // Slide: 1 new load per output point instead of 4
    for i in 1..n {
        p0 = p1;
        p1 = p2;
        p2 = p3;
        p3 = data[wrap(i0 + i as isize + 2)];
        out[i] = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
    }
}

/// Fast sliding-window 1D semi-Lagrangian shift for open (clamped) boundaries.
///
/// Same sliding-window optimization as the periodic variant: weights are constant
/// for uniform displacement, so only 1 new data load per output point. Boundary
/// lookups use clamping instead of wrapping.
#[inline]
fn sl_shift_1d_into_fast_open(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    out: &mut [f64],
) {
    if n == 0 {
        return;
    }
    // departure_idx for output point 0
    let center0 = -l + 0.5 * cell_size;
    let dep0_phys = center0 - disp;
    let dep0 = (dep0_phys + l) / cell_size - 0.5;

    // Check if the entire shift is out of bounds
    let dep_last = dep0 + (n - 1) as f64;
    if dep_last < -0.5 || dep0 >= n as f64 - 0.5 {
        for val in out.iter_mut().take(n) {
            *val = 0.0;
        }
        return;
    }

    let t = dep0 - dep0.floor();
    let t2 = t * t;
    let t3 = t2 * t;
    let w0 = -0.5 * t + t2 - 0.5 * t3;
    let w1 = 1.0 - 2.5 * t2 + 1.5 * t3;
    let w2 = 0.5 * t + 2.0 * t2 - 1.5 * t3;
    let w3 = -0.5 * t2 + 0.5 * t3;

    let n_isize = n as isize;
    let i0 = dep0.floor() as isize;
    let clamp = |j: isize| j.clamp(0, n_isize - 1) as usize;

    // Initialize sliding window
    let mut p0 = data[clamp(i0 - 1)];
    let mut p1 = data[clamp(i0)];
    let mut p2 = data[clamp(i0 + 1)];
    let mut p3 = data[clamp(i0 + 2)];

    // First point: check if in bounds
    if dep0 < -0.5 || dep0 >= n as f64 - 0.5 {
        out[0] = 0.0;
    } else {
        out[0] = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
    }

    // Slide: 1 new load per output point
    for i in 1..n {
        p0 = p1;
        p1 = p2;
        p2 = p3;
        p3 = data[clamp(i0 + i as isize + 2)];
        let dep_i = dep0 + i as f64;
        if dep_i < -0.5 || dep_i >= n as f64 - 0.5 {
            out[i] = 0.0;
        } else {
            out[i] = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
        }
    }
}

/// Like [`sl_shift_1d`], but writes results into a pre-allocated output buffer,
/// avoiding allocation. `out` must have length >= `n`.
///
/// Dispatches to fast sliding-window paths for both periodic and open boundaries.
pub fn sl_shift_1d_into(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    periodic: bool,
    out: &mut [f64],
) {
    if periodic {
        sl_shift_1d_into_fast(data, disp, cell_size, n, l, out);
    } else {
        sl_shift_1d_into_fast_open(data, disp, cell_size, n, l, out);
    }
}
