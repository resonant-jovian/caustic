//! FFT-based Poisson solvers. Periodic (FftPoisson) and isolated (FftIsolated via
//! James zero-padding method). Both O(N³ log N).

use super::super::{types::*, solver::PoissonSolver, init::domain::Domain};

/// Periodic-BC Poisson solver. O(N³ log N). For cosmological boxes.
pub struct FftPoisson {
    pub shape: [usize; 3],
}

impl FftPoisson {
    pub fn new(domain: &Domain) -> Self {
        todo!()
    }
}

impl PoissonSolver for FftPoisson {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        todo!("FFT rho, divide by -k^2, IFFT: Phi_hat = rho_hat/(k^2/(4*pi*G)). Zero k=0 mode.")
    }
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        todo!("spectral diff: gx = iFFT(i*kx * Phi_hat), gy, gz similarly")
    }
}

/// Isolated-BC Poisson solver (James 1977 zero-padding). Correct vacuum BC.
pub struct FftIsolated {
    pub shape: [usize; 3],
}

impl FftIsolated {
    pub fn new(domain: &Domain) -> Self {
        todo!()
    }
}

impl PoissonSolver for FftIsolated {
    fn solve(&self, density: &DensityField, g: f64) -> PotentialField {
        todo!("zero-pad rho into (2N)^3 box, FFT, multiply by Green's function, IFFT, extract Phi")
    }
    fn compute_acceleration(&self, potential: &PotentialField) -> AccelerationField {
        todo!("spectral diff on (2N)^3 padded domain")
    }
}
