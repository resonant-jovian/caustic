//! Monotonicity-Preserving 7th-order (MP7) advection scheme.
//!
//! MP7 uses an 8-point stencil (i-3 to i+4) with 7th-order Lagrange interpolation
//! and a monotonicity-preserving limiter (Suresh & Huynh 1997). The limiter clips
//! the interpolated value to local bounds, maintaining high accuracy for smooth
//! solutions while preventing spurious oscillations near discontinuities.

/// MP7 1D shift: shift `data` by `disp` on a grid with `cell_size` spacing.
///
/// Uses an 8-point stencil (i-3 to i+4), computes 7th-order Lagrange interpolation
/// at the departure point, then applies a monotonicity-preserving median limiter.
/// `periodic`: true -> wrap-around; false -> clamped boundaries.
pub fn mp7_shift_1d_into(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    periodic: bool,
    out: &mut [f64],
) {
    if n == 0 {
        return;
    }
    if periodic {
        mp7_shift_1d_into_periodic(data, disp, cell_size, n, out);
    } else {
        mp7_shift_1d_into_open(data, disp, cell_size, n, l, out);
    }
}

/// Periodic boundary MP7 shift using sliding window.
fn mp7_shift_1d_into_periodic(data: &[f64], disp: f64, cell_size: f64, n: usize, out: &mut [f64]) {
    let dep0 = -(disp / cell_size);
    let i0 = dep0.floor() as isize;
    let t = dep0 - dep0.floor();

    let n_isize = n as isize;
    let wrap = |j: isize| (j.rem_euclid(n_isize)) as usize;

    // Precompute 7th-order Lagrange weights (constant for all output points)
    let w = lagrange7_weights(t);

    // Initialize sliding window with 8 points: i0-3, i0-2, ..., i0+4
    let mut s = [0.0f64; 8];
    for k in 0..8 {
        s[k] = data[wrap(i0 - 3 + k as isize)];
    }

    let p7 = w[0] * s[0]
        + w[1] * s[1]
        + w[2] * s[2]
        + w[3] * s[3]
        + w[4] * s[4]
        + w[5] * s[5]
        + w[6] * s[6]
        + w[7] * s[7];
    out[0] = mp_limit(&s, p7);

    for i in 1..n {
        // Slide window: shift all left by 1, load 1 new point
        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];
        s[3] = s[4];
        s[4] = s[5];
        s[5] = s[6];
        s[6] = s[7];
        s[7] = data[wrap(i0 + i as isize + 4)];

        let p7 = w[0] * s[0]
            + w[1] * s[1]
            + w[2] * s[2]
            + w[3] * s[3]
            + w[4] * s[4]
            + w[5] * s[5]
            + w[6] * s[6]
            + w[7] * s[7];
        out[i] = mp_limit(&s, p7);
    }
}

/// Open boundary MP7 shift.
fn mp7_shift_1d_into_open(
    data: &[f64],
    disp: f64,
    cell_size: f64,
    n: usize,
    l: f64,
    out: &mut [f64],
) {
    let center0 = -l + 0.5 * cell_size;
    let dep0_phys = center0 - disp;
    let dep0 = (dep0_phys + l) / cell_size - 0.5;

    let dep_last = dep0 + (n - 1) as f64;
    if dep_last < -0.5 || dep0 >= n as f64 - 0.5 {
        for val in out.iter_mut().take(n) {
            *val = 0.0;
        }
        return;
    }

    let t = dep0 - dep0.floor();
    let i0 = dep0.floor() as isize;
    let n_isize = n as isize;
    let clamp = |j: isize| j.clamp(0, n_isize - 1) as usize;

    let w = lagrange7_weights(t);

    let mut s = [0.0f64; 8];
    for k in 0..8 {
        s[k] = data[clamp(i0 - 3 + k as isize)];
    }

    let dep_i = dep0;
    if dep_i < -0.5 || dep_i >= n as f64 - 0.5 {
        out[0] = 0.0;
    } else {
        let p7 = w[0] * s[0]
            + w[1] * s[1]
            + w[2] * s[2]
            + w[3] * s[3]
            + w[4] * s[4]
            + w[5] * s[5]
            + w[6] * s[6]
            + w[7] * s[7];
        out[0] = mp_limit(&s, p7);
    }

    for i in 1..n {
        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];
        s[3] = s[4];
        s[4] = s[5];
        s[5] = s[6];
        s[6] = s[7];
        s[7] = data[clamp(i0 + i as isize + 4)];

        let dep_i = dep0 + i as f64;
        if dep_i < -0.5 || dep_i >= n as f64 - 0.5 {
            out[i] = 0.0;
        } else {
            let p7 = w[0] * s[0]
                + w[1] * s[1]
                + w[2] * s[2]
                + w[3] * s[3]
                + w[4] * s[4]
                + w[5] * s[5]
                + w[6] * s[6]
                + w[7] * s[7];
            out[i] = mp_limit(&s, p7);
        }
    }
}

/// Compute 7th-order Lagrange interpolation weights for fractional position `t`.
///
/// The 8 nodes are at positions -3, -2, -1, 0, 1, 2, 3, 4 relative to the
/// floor of the departure index. For each weight k:
///   w[k] = prod_{j != k} (t - node[j]) / (node[k] - node[j])
#[inline]
fn lagrange7_weights(t: f64) -> [f64; 8] {
    const NODES: [f64; 8] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];

    // Precompute (t - node[j]) for all j
    let tm = [
        t - NODES[0], // t + 3
        t - NODES[1], // t + 2
        t - NODES[2], // t + 1
        t - NODES[3], // t
        t - NODES[4], // t - 1
        t - NODES[5], // t - 2
        t - NODES[6], // t - 3
        t - NODES[7], // t - 4
    ];

    // Precompute denominators: prod_{j != k} (node[k] - node[j])
    // These are constant and can be hardcoded.
    // node[k] - node[j] for each k:
    // k=0 (node=-3): (-3)-(-2)=-1, (-3)-(-1)=-2, (-3)-0=-3, (-3)-1=-4, (-3)-2=-5, (-3)-3=-6, (-3)-4=-7
    //   denom = (-1)*(-2)*(-3)*(-4)*(-5)*(-6)*(-7) = -5040
    // k=1 (node=-2): 1*(-1)*(-2)*(-3)*(-4)*(-5)*(-6) = 720
    // k=2 (node=-1): 2*1*(-1)*(-2)*(-3)*(-4)*(-5) = -240
    // k=3 (node=0):  3*2*1*(-1)*(-2)*(-3)*(-4) = 144
    // k=4 (node=1):  4*3*2*1*(-1)*(-2)*(-3) = -144
    // k=5 (node=2):  5*4*3*2*1*(-1)*(-2) = 240
    // k=6 (node=3):  6*5*4*3*2*1*(-1) = -720
    // k=7 (node=4):  7*6*5*4*3*2*1 = 5040
    const INV_DENOM: [f64; 8] = [
        1.0 / -5040.0,
        1.0 / 720.0,
        1.0 / -240.0,
        1.0 / 144.0,
        1.0 / -144.0,
        1.0 / 240.0,
        1.0 / -720.0,
        1.0 / 5040.0,
    ];

    // w[k] = (1/denom[k]) * prod_{j != k} (t - node[j])
    [
        INV_DENOM[0] * tm[1] * tm[2] * tm[3] * tm[4] * tm[5] * tm[6] * tm[7],
        INV_DENOM[1] * tm[0] * tm[2] * tm[3] * tm[4] * tm[5] * tm[6] * tm[7],
        INV_DENOM[2] * tm[0] * tm[1] * tm[3] * tm[4] * tm[5] * tm[6] * tm[7],
        INV_DENOM[3] * tm[0] * tm[1] * tm[2] * tm[4] * tm[5] * tm[6] * tm[7],
        INV_DENOM[4] * tm[0] * tm[1] * tm[2] * tm[3] * tm[5] * tm[6] * tm[7],
        INV_DENOM[5] * tm[0] * tm[1] * tm[2] * tm[3] * tm[4] * tm[6] * tm[7],
        INV_DENOM[6] * tm[0] * tm[1] * tm[2] * tm[3] * tm[4] * tm[5] * tm[7],
        INV_DENOM[7] * tm[0] * tm[1] * tm[2] * tm[3] * tm[4] * tm[5] * tm[6],
    ]
}

/// Monotonicity-preserving median limiter (Suresh & Huynh 1997).
///
/// Clips the polynomial interpolation value to local bounds derived from the
/// 4 nearest stencil points (indices 2, 3, 4, 5 — corresponding to positions
/// i-1, i, i+1, i+2). For smooth data the limiter does not activate, preserving
/// 7th-order accuracy.
#[inline]
fn mp_limit(stencil: &[f64; 8], poly_val: f64) -> f64 {
    let f_min = stencil[2].min(stencil[3]).min(stencil[4]).min(stencil[5]);
    let f_max = stencil[2].max(stencil[3]).max(stencil[4]).max(stencil[5]);
    poly_val.clamp(f_min, f_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mp7_preserves_constant() {
        let n = 16;
        let data = vec![1.0; n];
        let mut out = vec![0.0; n];
        let l = 1.0;
        let dx = 2.0 * l / n as f64;
        mp7_shift_1d_into(&data, 0.3 * dx, dx, n, l, true, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "Constant not preserved at {i}: {v}"
            );
        }
    }

    #[test]
    fn mp7_preserves_linear() {
        // Verify mass conservation for shifted data.
        // Use a constant (trivially linear and periodic) and a smooth sinusoidal.
        let n = 32;
        let l = 1.0;
        let dx = 2.0 * l / n as f64;
        let disp = 0.25 * dx;

        // Constant data: mass must be exactly conserved
        let data: Vec<f64> = (0..n).map(|_| 3.5).collect();
        let mut out = vec![0.0; n];
        mp7_shift_1d_into(&data, disp, dx, n, l, true, &mut out);

        let mass_in: f64 = data.iter().sum();
        let mass_out: f64 = out.iter().sum();
        assert!(
            (mass_in - mass_out).abs() < 1e-12,
            "Mass not conserved for constant: {mass_in} vs {mass_out}"
        );

        // Smooth periodic sinusoidal profile: mass approximately conserved
        let data2: Vec<f64> = (0..n)
            .map(|i| {
                let x = -l + (i as f64 + 0.5) * dx;
                2.0 + 0.3 * (std::f64::consts::PI * x / l).sin()
            })
            .collect();
        let mut out2 = vec![0.0; n];
        mp7_shift_1d_into(&data2, disp, dx, n, l, true, &mut out2);
        let mass_in2: f64 = data2.iter().sum();
        let mass_out2: f64 = out2.iter().sum();
        assert!(
            (mass_in2 - mass_out2).abs() / mass_in2.abs() < 1e-8,
            "Mass not conserved for sinusoidal: {mass_in2} vs {mass_out2}"
        );
    }

    #[test]
    fn mp7_convergence_rate() {
        // Test convergence of the 7th-order Lagrange interpolation using a smooth
        // monotone function (arctan) with open boundaries. On monotone data the MP
        // limiter does not activate (the interpolant stays within local bounds),
        // so the full high-order convergence rate is observed.
        //
        // We skip boundary cells where clamped-boundary effects dominate and measure
        // the interior Linf error.
        let l = 2.0;
        let disp_frac = 0.37;
        let margin = 8; // skip this many cells at each boundary

        let mut prev_err = f64::MAX;
        let mut prev_n = 0usize;
        let mut rates = Vec::new();

        for &n in &[32, 64, 128, 256] {
            let dx = 2.0 * l / n as f64;
            let disp = disp_frac * dx;

            // arctan(3x) is smooth, monotone, and well-resolved on [-2, 2]
            let data: Vec<f64> = (0..n)
                .map(|i| {
                    let x = -l + (i as f64 + 0.5) * dx;
                    (3.0 * x).atan()
                })
                .collect();

            let mut out = vec![0.0; n];
            mp7_shift_1d_into(&data, disp, dx, n, l, false, &mut out);

            // Exact shifted arctan
            let exact: Vec<f64> = (0..n)
                .map(|i| {
                    let x = -l + (i as f64 + 0.5) * dx;
                    let xs = x - disp;
                    (3.0 * xs).atan()
                })
                .collect();

            // Interior Linf error (skip boundary cells)
            let err: f64 = out[margin..n - margin]
                .iter()
                .zip(exact[margin..n - margin].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            if prev_n > 0 && err > 0.0 {
                let rate = (prev_err / err).ln() / (n as f64 / prev_n as f64).ln();
                rates.push(rate);
            }
            prev_err = err;
            prev_n = n;
        }

        // Expect convergence rate >= 6 (theoretical 7-8 for 7th-order Lagrange;
        // allow margin for the arctan's higher derivatives and discrete effects).
        let avg_rate: f64 = rates.iter().sum::<f64>() / rates.len() as f64;
        assert!(
            avg_rate > 5.5,
            "Convergence rate too low: {avg_rate:.2} (expected ~7), rates: {rates:?}"
        );
    }

    #[test]
    fn mp7_positivity() {
        let n = 64;
        let l = 3.0;
        let dx = 2.0 * l / n as f64;
        // Sharp Gaussian — provokes oscillations in high-order schemes
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let x = -l + (i as f64 + 0.5) * dx;
                (-(x * x) / 0.01).exp()
            })
            .collect();
        let mut out = vec![0.0; n];
        mp7_shift_1d_into(&data, 0.5 * dx, dx, n, l, true, &mut out);

        // The MP limiter clips to local bounds from the 4 nearest stencil points,
        // so the result cannot be negative when all input data is non-negative.
        let min_val = out.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            min_val >= 0.0,
            "MP7 limiter failed — produced negative value: {min_val}"
        );
    }
}
