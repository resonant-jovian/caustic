use crate::sim::Simulation;
use crate::tooling::core::algos::lagrangian::SemiLagrangian;
use crate::tooling::core::algos::uniform::UniformGrid6D;
use crate::tooling::core::init::domain::Domain;
use crate::tooling::core::phasespace::PhaseSpaceRepr as _;
use crate::tooling::core::poisson::fft::FftPoisson;
use crate::tooling::core::time::strang::StrangSplitting;
use crate::tooling::core::types::{DensityField, PhaseSpaceSnapshot};

/// Relative drift: |a - b| / |b|, returns 0.0 if |b| < 1e-30.
pub fn relative_drift(current: f64, reference: f64) -> f64 {
    let r = reference.abs();
    if r > 1e-30 {
        (current - reference).abs() / r
    } else {
        0.0
    }
}

/// Standard post-run assertions: no NaN in density, positive mass, enough diagnostics.
pub fn assert_valid_output(density: &DensityField, diag_len: usize) {
    assert!(
        !density.data.iter().any(|x| x.is_nan()),
        "Density contains NaN"
    );
    assert!(
        density.data.iter().sum::<f64>() > 0.0,
        "Density must be positive"
    );
    assert!(diag_len >= 2, "Need at least 2 diagnostic entries");
}

/// Build a standard sim: FftPoisson + SemiLagrangian + StrangSplitting + CFL=1.0.
#[allow(clippy::unwrap_used)]
pub fn build_standard_sim(domain: Domain, snap: PhaseSpaceSnapshot, t_final: f64) -> Simulation {
    let poisson = FftPoisson::new(&domain);
    Simulation::builder()
        .domain(domain)
        .poisson_solver(poisson)
        .advector(SemiLagrangian::new())
        .integrator(StrangSplitting::new(1.0))
        .initial_conditions(snap)
        .time_final(t_final)
        .build()
        .unwrap()
}

/// Compute initial density from a snapshot (creates temporary UniformGrid6D).
pub fn snapshot_density(snap: &PhaseSpaceSnapshot, domain: &Domain) -> DensityField {
    let grid = UniformGrid6D::from_snapshot(
        PhaseSpaceSnapshot {
            data: snap.data.clone(),
            shape: snap.shape,
            time: 0.0,
        },
        domain.clone(),
    );
    grid.compute_density()
}

/// Compute total mass from density: sum(rho) * dx^3.
pub fn density_mass(density: &DensityField, dx: [f64; 3]) -> f64 {
    density.data.iter().sum::<f64>() * dx[0] * dx[1] * dx[2]
}
