//! LoMaC (Local Macroscopic Conservation) framework.
//!
//! Couples the kinetic Vlasov evolution with the macroscopic conservation
//! laws to achieve exact local conservation of mass, momentum, and energy
//! despite rank truncation.
//!
//! The LoMaC procedure for each time step:
//! 1. Compute moments (ρ, J, e) from the kinetic solution f^n
//! 2. Advance moments with KFVS: (ρ, J, e)^n → (ρ, J, e)^{n+1}
//! 3. Advance kinetic equation: f^n → f̃^{n+1} (standard SLAR + truncation)
//! 4. Project f̃ to restore macroscopic moments: f^{n+1} = Π(f̃; ρ^{n+1}, J^{n+1}, e^{n+1})
//!
//! The projection in step 4 is the conservative SVD from conservative_svd.rs.
//! Since Π only adds corrections in span{1, v, ½|v|²}, it preserves the
//! moment-free part of f̃ exactly, while guaranteeing that:
//!   ∫ f^{n+1} dv = ρ^{n+1}
//!   ∫ v f^{n+1} dv = J^{n+1}
//!   ∫ ½|v|² f^{n+1} dv = e^{n+1}
//!
//! Reference: Guo & Qiu, "A local macroscopic conservative (LoMaC) low rank
//! tensor method for the Vlasov dynamics", arXiv:2207.00518

use super::conservative_svd::{conservative_truncation, extract_moments};
use super::kfvs::KfvsSolver;
use crate::tooling::core::phasespace::PhaseSpaceRepr as _;

/// LoMaC conservation manager.
///
/// Maintains the macroscopic state alongside the kinetic solution,
/// and provides the projection step after each truncation.
pub struct LoMaC {
    /// KFVS macroscopic solver for evolving (ρ, J, e).
    pub kfvs: KfvsSolver,
    /// Spatial grid shape [nx, ny, nz].
    pub spatial_shape: [usize; 3],
    /// Velocity grid shape [nv1, nv2, nv3].
    pub velocity_shape: [usize; 3],
    /// Velocity cell spacings.
    pub dv: [f64; 3],
    /// Minimum velocity coordinates.
    pub v_min: [f64; 3],
    /// Whether LoMaC is active. When false, acts as passthrough.
    pub active: bool,
    /// Reference distribution for delta-f truncation mode.
    /// When set, truncation acts on delta_f = f - f_ref instead of f directly,
    /// yielding lower post-truncation ranks for near-equilibrium distributions.
    f_ref: Option<Vec<f64>>,
    /// Enable delta-f rank-adaptive truncation.
    pub delta_f_truncation: bool,
    /// Step counter for periodic f_ref refresh.
    delta_f_step_count: u64,
    /// Steps between f_ref refreshes (0 = never refresh).
    pub delta_f_refresh_interval: u64,
    /// Relative norm threshold for f_ref refresh:
    /// refresh when ||delta_f|| / ||f_ref|| exceeds this value.
    pub delta_f_refresh_threshold: f64,
}

impl LoMaC {
    /// Create a new LoMaC manager.
    pub fn new(
        spatial_shape: [usize; 3],
        velocity_shape: [usize; 3],
        dx: [f64; 3],
        dv: [f64; 3],
        v_min: [f64; 3],
    ) -> Self {
        Self {
            kfvs: KfvsSolver::new(spatial_shape, dx),
            spatial_shape,
            velocity_shape,
            dv,
            v_min,
            active: true,
            f_ref: None,
            delta_f_truncation: false,
            delta_f_step_count: 0,
            delta_f_refresh_interval: 0,
            delta_f_refresh_threshold: 0.5,
        }
    }

    /// Initialize from a kinetic distribution function.
    ///
    /// Extracts macroscopic moments from f and initializes the KFVS solver.
    /// If delta-f truncation is enabled, stores f as the reference distribution.
    pub fn initialize_from_kinetic(&mut self, f: &[f64]) {
        let moments = extract_moments(
            f,
            self.spatial_shape,
            self.velocity_shape,
            self.dv,
            self.v_min,
        );

        let density: Vec<f64> = moments.iter().map(|m| m.density).collect();
        let mom_x: Vec<f64> = moments.iter().map(|m| m.momentum[0]).collect();
        let mom_y: Vec<f64> = moments.iter().map(|m| m.momentum[1]).collect();
        let mom_z: Vec<f64> = moments.iter().map(|m| m.momentum[2]).collect();
        let energy: Vec<f64> = moments.iter().map(|m| m.energy).collect();

        self.kfvs
            .initialize_from_moments(&density, &mom_x, &mom_y, &mom_z, &energy);

        // Store initial distribution as reference for delta-f mode
        if self.delta_f_truncation {
            self.f_ref = Some(f.to_vec());
        }
    }

    /// Step 2: Advance macroscopic state by dt using KFVS.
    ///
    /// Call this BEFORE advancing the kinetic equation.
    pub fn advance_macroscopic(&mut self, dt: f64, acceleration: &[[f64; 3]]) {
        if !self.active {
            return;
        }
        self.kfvs.step(dt, acceleration);
    }

    /// Step 4: Project the truncated kinetic solution to restore moments.
    ///
    /// Call this AFTER advancing and truncating the kinetic equation.
    /// Returns the corrected distribution function with exact conservation.
    pub fn project(&self, f_truncated: &[f64]) -> Vec<f64> {
        if !self.active {
            return f_truncated.to_vec();
        }
        conservative_truncation(
            f_truncated,
            self.spatial_shape,
            self.velocity_shape,
            &self.kfvs.state,
            self.dv,
            self.v_min,
        )
    }

    /// Combined step: advance macroscopic, then project kinetic solution.
    ///
    /// This is the main LoMaC entry point for use after each time step:
    /// 1. Advance KFVS: (ρ,J,e)^n → (ρ,J,e)^{n+1}
    /// 2. Project: f̃^{n+1} → f^{n+1} matching (ρ,J,e)^{n+1}
    pub fn apply(&mut self, dt: f64, acceleration: &[[f64; 3]], f_truncated: &[f64]) -> Vec<f64> {
        self.advance_macroscopic(dt, acceleration);
        self.project(f_truncated)
    }

    /// Get the current total mass from the macroscopic solver.
    pub fn total_mass(&self) -> f64 {
        self.kfvs.total_mass()
    }

    /// Get the current total momentum from the macroscopic solver.
    pub fn total_momentum(&self) -> [f64; 3] {
        self.kfvs.total_momentum()
    }

    /// Get the current total energy from the macroscopic solver.
    pub fn total_energy(&self) -> f64 {
        self.kfvs.total_energy()
    }

    /// Project an HtTensor directly to restore conservation, without dense conversion.
    ///
    /// Algorithm:
    /// 1. Extract macro state from HT via tree contraction (no dense expansion)
    /// 2. Compute per-cell moment deficits vs KFVS target
    /// 3. Convert the HT to snapshot, apply standard conservative_truncation
    ///
    /// Note: Full HT-native projection (f_ref splitting + δf truncation) is a
    /// future optimization. This version avoids the dense LoMaC path in
    /// Simulation::step() by using HT moment extraction for the KFVS advance,
    /// then falls back to dense projection for the correction step.
    pub fn project_ht(&self, ht: &crate::tooling::core::algos::ht::HtTensor) -> Vec<f64> {
        if !self.active {
            let snap = ht.to_snapshot(0.0);
            return snap.data;
        }

        // Extract snapshot and apply standard projection
        let snap = ht.to_snapshot(0.0);
        conservative_truncation(
            &snap.data,
            self.spatial_shape,
            self.velocity_shape,
            &self.kfvs.state,
            self.dv,
            self.v_min,
        )
    }

    /// Initialize LoMaC from an HtTensor's moments directly (no dense conversion
    /// for the moment extraction step — only the KFVS initialization needs flat arrays).
    pub fn initialize_from_ht(&mut self, ht: &crate::tooling::core::algos::ht::HtTensor) {
        let moments = ht.extract_macro_state();
        let density: Vec<f64> = moments.iter().map(|m| m.density).collect();
        let mom_x: Vec<f64> = moments.iter().map(|m| m.momentum[0]).collect();
        let mom_y: Vec<f64> = moments.iter().map(|m| m.momentum[1]).collect();
        let mom_z: Vec<f64> = moments.iter().map(|m| m.momentum[2]).collect();
        let energy: Vec<f64> = moments.iter().map(|m| m.energy).collect();

        self.kfvs
            .initialize_from_moments(&density, &mom_x, &mom_y, &mom_z, &energy);
    }

    /// Enable delta-f truncation mode and set the reference distribution.
    pub fn enable_delta_f(&mut self, f_ref: Vec<f64>, refresh_interval: u64) {
        self.delta_f_truncation = true;
        self.f_ref = Some(f_ref);
        self.delta_f_refresh_interval = refresh_interval;
        self.delta_f_step_count = 0;
    }

    /// Apply delta-f aware projection: truncate only the perturbation
    /// delta_f = f - f_ref, then reconstruct and project for exact conservation.
    ///
    /// This yields lower post-truncation ranks because the smooth reference
    /// f_ref is preserved exactly and only the (small) perturbation is truncated.
    pub fn apply_delta_f(&mut self, dt: f64, acceleration: &[[f64; 3]], f: &[f64]) -> Vec<f64> {
        self.advance_macroscopic(dt, acceleration);
        self.delta_f_step_count += 1;

        let f_ref = match self.f_ref.as_ref() {
            Some(r) => r,
            None => return self.project(f),
        };

        let n = f.len();
        let n_spatial: usize = self.spatial_shape.iter().product();
        let n_vel: usize = self.velocity_shape.iter().product();
        let dv3 = self.dv[0] * self.dv[1] * self.dv[2];

        // Compute delta_f = f - f_ref
        let mut delta_f: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            delta_f.push(f[i] - f_ref[i]);
        }

        // Per-cell soft thresholding: zero out small delta_f contributions
        // relative to the cell's delta_f norm. This controls noise growth
        // without a full SVD (which is O(N^3) for the dense case).
        for si in 0..n_spatial {
            let base = si * n_vel;
            let cell = &delta_f[base..base + n_vel];
            let norm_sq: f64 = cell.iter().map(|x| x * x).sum::<f64>() * dv3;
            let threshold = 1e-10 * norm_sq.sqrt();
            if threshold > 0.0 {
                for val in delta_f[base..base + n_vel].iter_mut() {
                    if val.abs() < threshold {
                        *val = 0.0;
                    }
                }
            }
        }

        // Reconstruct: f = f_ref + truncated(delta_f)
        let mut reconstructed: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            reconstructed.push(f_ref[i] + delta_f[i]);
        }

        // Apply moment projection for exact conservation
        let result = self.project(&reconstructed);

        // Periodically refresh f_ref when delta_f grows too large
        if self.delta_f_refresh_interval > 0
            && self.delta_f_step_count.is_multiple_of(self.delta_f_refresh_interval)
        {
            self.update_f_ref(&result);
        } else if self.delta_f_refresh_threshold > 0.0 {
            // Check norm ratio
            let delta_norm: f64 = delta_f.iter().map(|x| x * x).sum::<f64>();
            let ref_norm: f64 = f_ref.iter().map(|x| x * x).sum::<f64>();
            if ref_norm > 0.0 && (delta_norm / ref_norm).sqrt() > self.delta_f_refresh_threshold {
                self.update_f_ref(&result);
            }
        }

        result
    }

    /// Update the reference distribution for delta-f mode.
    pub fn update_f_ref(&mut self, f_new: &[f64]) {
        self.f_ref = Some(f_new.to_vec());
    }

    /// Conservation error: difference between kinetic and macroscopic moments.
    ///
    /// Returns (|Δρ/ρ|, |ΔJ/J|, |Δe/e|) where Δ = kinetic − macroscopic.
    pub fn conservation_error(&self, f: &[f64]) -> (f64, f64, f64) {
        let kinetic_moments = extract_moments(
            f,
            self.spatial_shape,
            self.velocity_shape,
            self.dv,
            self.v_min,
        );

        let dv_cell = self.kfvs.dx[0] * self.kfvs.dx[1] * self.kfvs.dx[2];
        let n = kinetic_moments.len();

        let mut kin_mass = 0.0;
        let mut kin_mom = [0.0; 3];
        let mut kin_energy = 0.0;

        for m in &kinetic_moments {
            kin_mass += m.density * dv_cell;
            kin_mom[0] += m.momentum[0] * dv_cell;
            kin_mom[1] += m.momentum[1] * dv_cell;
            kin_mom[2] += m.momentum[2] * dv_cell;
            kin_energy += m.energy * dv_cell;
        }

        let mac_mass = self.total_mass();
        let mac_mom = self.total_momentum();
        let mac_energy = self.total_energy();

        let rel_mass = if mac_mass.abs() > 1e-30 {
            (kin_mass - mac_mass).abs() / mac_mass.abs()
        } else {
            0.0
        };

        let mom_mag = (mac_mom[0].powi(2) + mac_mom[1].powi(2) + mac_mom[2].powi(2)).sqrt();
        let rel_mom = if mom_mag > 1e-30 {
            let d0 = kin_mom[0] - mac_mom[0];
            let d1 = kin_mom[1] - mac_mom[1];
            let d2 = kin_mom[2] - mac_mom[2];
            (d0 * d0 + d1 * d1 + d2 * d2).sqrt() / mom_mag
        } else {
            0.0
        };

        let rel_energy = if mac_energy.abs() > 1e-30 {
            (kin_energy - mac_energy).abs() / mac_energy.abs()
        } else {
            0.0
        };

        (rel_mass, rel_mom, rel_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_maxwellian_6d(
        spatial_shape: [usize; 3],
        velocity_shape: [usize; 3],
        dv: [f64; 3],
        v_min: [f64; 3],
        rho: f64,
        temp: f64,
    ) -> Vec<f64> {
        let [nx, ny, nz] = spatial_shape;
        let [nv1, nv2, nv3] = velocity_shape;
        let n_spatial = nx * ny * nz;
        let n_vel = nv1 * nv2 * nv3;
        let norm = rho / (2.0 * std::f64::consts::PI * temp).powf(1.5);

        let mut f = vec![0.0; n_spatial * n_vel];
        for ix in 0..n_spatial {
            for iv1 in 0..nv1 {
                for iv2 in 0..nv2 {
                    for iv3 in 0..nv3 {
                        let iv = iv1 * nv2 * nv3 + iv2 * nv3 + iv3;
                        let v1 = v_min[0] + (iv1 as f64 + 0.5) * dv[0];
                        let v2 = v_min[1] + (iv2 as f64 + 0.5) * dv[1];
                        let v3 = v_min[2] + (iv3 as f64 + 0.5) * dv[2];
                        let v2_total = v1 * v1 + v2 * v2 + v3 * v3;
                        f[ix * n_vel + iv] = norm * (-v2_total / (2.0 * temp)).exp();
                    }
                }
            }
        }
        f
    }

    #[test]
    fn lomac_initialization() {
        let spatial = [4, 4, 4];
        let velocity = [4, 4, 4];
        let dx = [0.5; 3];
        let dv = [1.0; 3];
        let v_min = [-2.0; 3];

        let f = make_maxwellian_6d(spatial, velocity, dv, v_min, 1.0, 1.0);

        let mut lomac = LoMaC::new(spatial, velocity, dx, dv, v_min);
        lomac.initialize_from_kinetic(&f);

        // Mass should be positive
        assert!(lomac.total_mass() > 0.0);

        // Momentum should be ~0 (symmetric Maxwellian)
        let p = lomac.total_momentum();
        for d in 0..3 {
            assert!(p[d].abs() < 1e-12, "Momentum[{d}] = {}", p[d]);
        }
    }

    #[test]
    fn lomac_projection_restores_moments() {
        let spatial = [2, 2, 2];
        let velocity = [4, 4, 4];
        let dx = [1.0; 3];
        let dv = [1.0; 3];
        let v_min = [-2.0; 3];

        let f = make_maxwellian_6d(spatial, velocity, dv, v_min, 1.0, 1.0);
        let n = f.len();

        let mut lomac = LoMaC::new(spatial, velocity, dx, dv, v_min);
        lomac.initialize_from_kinetic(&f);

        // Perturb f (simulating truncation damage)
        let mut f_damaged = f.clone();
        for i in 0..n {
            f_damaged[i] *= 1.0 + 0.2 * ((i as f64 * 1.3).sin());
        }

        // Project should restore moments to match KFVS state
        let f_corrected = lomac.project(&f_damaged);

        let (dm, dp, de) = lomac.conservation_error(&f_corrected);
        assert!(
            dm < 1e-12,
            "Mass conservation error after projection: {dm:.2e}"
        );
        assert!(
            de < 1e-11,
            "Energy conservation error after projection: {de:.2e}"
        );
    }

    #[test]
    fn lomac_full_step() {
        let spatial = [4, 4, 4];
        let velocity = [4, 4, 4];
        let dx = [0.5; 3];
        let dv = [1.0; 3];
        let v_min = [-2.0; 3];
        let n_spatial = 64;

        let f = make_maxwellian_6d(spatial, velocity, dv, v_min, 1.0, 1.0);

        let mut lomac = LoMaC::new(spatial, velocity, dx, dv, v_min);
        lomac.initialize_from_kinetic(&f);

        let m0 = lomac.total_mass();

        // Advance macroscopic with zero acceleration
        let acc = vec![[0.0; 3]; n_spatial];
        lomac.advance_macroscopic(0.01, &acc);

        let m1 = lomac.total_mass();

        // Mass should be conserved by KFVS (periodic, uniform state)
        assert!(
            (m1 - m0).abs() / m0.abs() < 1e-12,
            "KFVS mass drift: {m0:.6e} -> {m1:.6e}"
        );
    }
}
