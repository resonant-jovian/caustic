//! Semi-Lagrangian advector. Traces characteristics backwards from each grid point and
//! interpolates. Eliminates CFL constraint; Δt limited only by accuracy requirements.

use super::super::{types::*, phasespace::PhaseSpaceRepr, advecator::Advector};

/// Semi-Lagrangian advector with cubic spline interpolation.
pub struct SemiLagrangian {
    pub interpolation_order: usize,
}

impl SemiLagrangian {
    pub fn new() -> Self {
        Self { interpolation_order: 3 }
    }
}

impl Advector for SemiLagrangian {
    fn drift(&self, repr: &mut dyn PhaseSpaceRepr, dt: f64) {
        // UniformGrid6D ignores the displacement field; it computes shifts from its velocity grid.
        let dummy = DisplacementField { dx: vec![], dy: vec![], dz: vec![], shape: [0, 0, 0] };
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
    a0 + a1 * t + a2 * t * t + a3 * t * t * t
}

/// 4-point Catmull-Rom cubic spline interpolation with clamped (open) boundary.
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
    a0 + a1 * t + a2 * t * t + a3 * t * t * t
}

/// 1D semi-Lagrangian shift of a grid line.
///
/// - `data`: array of `n` values on a periodic/open 1D grid spanning `[−l, l]`
/// - `disp`: displacement (physical units) — the whole line shifts by this amount
/// - `cell_size`: grid spacing (= 2l/n)
/// - `n`: number of cells
/// - `l`: half-extent of the domain
/// - `periodic`: true → wrap-around; false → absorbing (out-of-bounds → 0)
pub(crate) fn sl_shift_1d(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    periodic: bool,
) -> Vec<f64> {
    (0..n).map(|i| {
        // Center of output cell i: -l + (i + 0.5) * cell_size
        // Departure point (back-trace): center_i - disp
        let center_i = -l + (i as f64 + 0.5) * cell_size;
        let departure_phys = center_i - disp;
        // Fractional cell index of departure point
        let departure_idx = (departure_phys + l) / cell_size - 0.5;
        if periodic {
            cubic_spline_interpolate(data, departure_idx, n)
        } else {
            // Absorbing BC: anything outside [−0.5, n−0.5) maps to zero
            if departure_idx < -0.5 || departure_idx >= n as f64 - 0.5 {
                0.0
            } else {
                let clamped = departure_idx.clamp(0.0, n as f64 - 1.0 - 1e-10);
                cubic_spline_interpolate_open(data, clamped, n)
            }
        }
    }).collect()
}
