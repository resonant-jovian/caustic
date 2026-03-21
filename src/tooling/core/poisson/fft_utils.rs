//! Shared 3D FFT utilities with scratch-buffer-based implementation.
//!
//! Eliminates per-line heap allocations by using a pre-allocated scratch buffer
//! for axis transposes. Non-contiguous axes are made contiguous via transpose,
//! FFT'd in-place, then transposed back.

use rayon::prelude::*;
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Tiled 2D transpose: src[r * cols + c] → dst[c * rows + r].
///
/// 8×8 tiles (1KB of `Complex<f64>`) fit L1 cache, reducing thrashing
/// for large matrices compared to naive row-by-row transpose.
#[inline]
pub fn transpose_tiled<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    const TILE: usize = 8;
    for r0 in (0..rows).step_by(TILE) {
        let rend = (r0 + TILE).min(rows);
        for c0 in (0..cols).step_by(TILE) {
            let cend = (c0 + TILE).min(cols);
            for r in r0..rend {
                let src_base = r * cols;
                for c in c0..cend {
                    dst[c * rows + r] = src[src_base + c];
                }
            }
        }
    }
}

/// Perform a full complex 3D FFT (forward or inverse) using precomputed plans
/// and a scratch buffer to avoid per-line allocations.
///
/// `buf` is a flat row-major complex buffer of size `nx * ny * nz`.
/// `scratch` must be at least `nx * ny * nz` elements.
/// `plans` contains one precomputed plan per axis: [x, y, z].
pub fn fft_3d_c2c_scratch(
    buf: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
    shape: [usize; 3],
    plans: &[Arc<dyn rustfft::Fft<f64>>; 3],
) {
    let [nx, ny, nz] = shape;

    // Axis 2 (z): contiguous rows — in-place, no scratch needed
    buf.par_chunks_mut(nz).for_each(|row| {
        plans[2].process(row);
    });

    // Axis 1 (y): transpose buf(x,y,z) → scratch(x,z,y), FFT contiguous ny-rows, transpose back
    scratch[..nx * ny * nz]
        .par_chunks_mut(nz * ny)
        .enumerate()
        .for_each(|(ix, slab)| {
            let src = &buf[ix * ny * nz..(ix + 1) * ny * nz];
            transpose_tiled(src, slab, ny, nz);
        });
    // FFT on contiguous ny-rows in scratch
    scratch[..nx * nz * ny].par_chunks_mut(ny).for_each(|row| {
        plans[1].process(row);
    });
    // Transpose back
    buf.par_chunks_mut(ny * nz)
        .enumerate()
        .for_each(|(ix, slab)| {
            let src = &scratch[ix * nz * ny..(ix + 1) * nz * ny];
            transpose_tiled(src, slab, nz, ny);
        });

    // Axis 0 (x): transpose buf(x,y,z) → scratch(y,z,x), FFT contiguous nx-rows, transpose back
    scratch[..nx * ny * nz]
        .par_chunks_mut(nz * nx)
        .enumerate()
        .for_each(|(iy, slab)| {
            for iz in 0..nz {
                for ix in 0..nx {
                    slab[iz * nx + ix] = buf[ix * ny * nz + iy * nz + iz];
                }
            }
        });
    // FFT on contiguous nx-rows
    scratch[..ny * nz * nx].par_chunks_mut(nx).for_each(|row| {
        plans[0].process(row);
    });
    // Transpose back
    buf.par_chunks_mut(ny * nz)
        .enumerate()
        .for_each(|(ix, slab)| {
            for iy in 0..ny {
                for iz in 0..nz {
                    slab[iy * nz + iz] = scratch[iy * nz * nx + iz * nx + ix];
                }
            }
        });
}

/// 3D FFT (forward) of a real array, returning complex array.
/// Uses precomputed plans and scratch buffer.
pub fn fft_3d_forward_scratch(
    data: &[f64],
    scratch: &mut [Complex<f64>],
    shape: [usize; 3],
    plans: &[Arc<dyn rustfft::Fft<f64>>; 3],
) -> Vec<Complex<f64>> {
    let n_total: usize = shape.iter().product();
    assert_eq!(data.len(), n_total);

    let mut buf: Vec<Complex<f64>> = data.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_3d_c2c_scratch(&mut buf, scratch, shape, plans);
    buf
}

/// 3D IFFT of a complex array, returning real part.
/// Uses precomputed plans and scratch buffer.
pub fn fft_3d_inverse_scratch(
    data: &[Complex<f64>],
    scratch: &mut [Complex<f64>],
    shape: [usize; 3],
    plans: &[Arc<dyn rustfft::Fft<f64>>; 3],
) -> Vec<f64> {
    let n_total: usize = shape.iter().product();
    assert_eq!(data.len(), n_total);
    let scale = 1.0 / n_total as f64;

    let mut buf = data.to_vec();
    fft_3d_c2c_scratch(&mut buf, scratch, shape, plans);
    buf.par_iter().map(|c| c.re * scale).collect()
}
