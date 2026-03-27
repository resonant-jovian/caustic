//! Long-term conservation validation.
//!
//! Runs Plummer equilibrium for many dynamical times and tracks
//! energy, Casimir C₂, and entropy drift rates.

#[cfg(test)]
mod tests {
    use crate::sim::Simulation;
    use crate::tooling::core::algos::lagrangian::SemiLagrangian;
    use crate::tooling::core::init::domain::{Domain, SpatialBoundType, VelocityBoundType};
    use crate::tooling::core::init::isolated::{PlummerIC, sample_on_grid};
    use crate::tooling::core::poisson::fft::FftPoisson;
    use crate::tooling::core::time::strang::StrangSplitting;

    #[test]
    fn long_conservation_plummer() {
        let domain = Domain::builder()
            .spatial_extent(8.0)
            .velocity_extent(3.0)
            .spatial_resolution(8)
            .velocity_resolution(8)
            .t_final(5.0)
            .spatial_bc(SpatialBoundType::Periodic)
            .velocity_bc(VelocityBoundType::Open)
            .build()
            .unwrap();

        let ic = PlummerIC::new(1.0, 1.0, 1.0);
        let snap = sample_on_grid(&ic, &domain);

        let poisson = FftPoisson::new(&domain);
        let mut sim = Simulation::builder()
            .domain(domain)
            .poisson_solver(poisson)
            .advector(SemiLagrangian::new())
            .integrator(StrangSplitting::new())
            .initial_conditions(snap)
            .time_final(5.0)
            .build()
            .unwrap();

        let pkg = sim.run().unwrap();
        let summary = pkg.conservation_summary;

        // After many steps, energy drift should stay bounded
        assert!(
            summary.max_energy_drift < 0.5,
            "Long-run energy drift too large: {:.2e}",
            summary.max_energy_drift,
        );

        // Casimir should be reasonably conserved
        assert!(
            summary.max_casimir_drift < 0.5,
            "Casimir drift: {:.2e}",
            summary.max_casimir_drift
        );
    }
}
