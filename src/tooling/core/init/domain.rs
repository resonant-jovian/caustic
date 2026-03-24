//! Domain specification for the 6D Vlasov--Poisson computational box.
//!
//! Defines spatial extents, velocity extents, grid resolutions, boundary
//! conditions, and the simulation time range.  All configuration-facing
//! lengths and times are stored as `rust_decimal::Decimal` for exact
//! arithmetic; hot-path accessors (`dx()`, `dv()`, `lx()`, `lv()`) return
//! cached `f64` values computed once at construction.  Use
//! [`DomainBuilder`] for ergonomic construction with validation.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Spatial half-extents of the simulation domain in each dimension.
///
/// The domain spans `[-neg_xi, xi]` along axis i when `neg_xi` is `Some`,
/// or `[0, xi]` when `neg_xi` is `None`.
#[derive(Clone)]
pub struct SpatialDom {
    /// Positive half-extent along the first spatial axis.
    pub x1: Decimal,
    /// Positive half-extent along the second spatial axis.
    pub x2: Decimal,
    /// Positive half-extent along the third spatial axis.
    pub x3: Decimal,
    /// Negative half-extent along x1; `None` means the domain starts at 0.
    pub neg_x1: Option<Decimal>,
    /// Negative half-extent along x2; `None` means the domain starts at 0.
    pub neg_x2: Option<Decimal>,
    /// Negative half-extent along x3; `None` means the domain starts at 0.
    pub neg_x3: Option<Decimal>,
}

/// Velocity half-extents of the simulation domain in each dimension.
///
/// Layout mirrors [`SpatialDom`]: the domain spans `[-neg_vi, vi]` when
/// the negative extent is `Some`.
#[derive(Clone)]
pub struct VelocityDom {
    /// Positive half-extent along the first velocity axis.
    pub v1: Decimal,
    /// Positive half-extent along the second velocity axis.
    pub v2: Decimal,
    /// Positive half-extent along the third velocity axis.
    pub v3: Decimal,
    /// Negative half-extent along v1; `None` means the domain starts at 0.
    pub neg_v1: Option<Decimal>,
    /// Negative half-extent along v2; `None` means the domain starts at 0.
    pub neg_v2: Option<Decimal>,
    /// Negative half-extent along v3; `None` means the domain starts at 0.
    pub neg_v3: Option<Decimal>,
}

/// Number of grid cells in each spatial dimension.
#[derive(Clone)]
pub struct SpatialRes {
    /// Cell count along the first spatial axis.
    pub x1: i128,
    /// Cell count along the second spatial axis.
    pub x2: i128,
    /// Cell count along the third spatial axis.
    pub x3: i128,
}

/// Number of grid cells in each velocity dimension.
#[derive(Clone)]
pub struct VelocityRes {
    /// Cell count along the first velocity axis.
    pub v1: i128,
    /// Cell count along the second velocity axis.
    pub v2: i128,
    /// Cell count along the third velocity axis.
    pub v3: i128,
}

/// Time range for the simulation.
#[derive(Clone)]
pub struct TimeRange {
    /// Initial simulation time (usually 0).
    pub t0: Decimal,
    /// Final simulation time; the integrator advances until `t >= t_final`.
    pub t_final: Decimal,
}

/// Integration timestep. May be adaptive (use `OptionalParams`) or fixed.
#[derive(Clone)]
pub struct Timestep {
    /// Fixed time-step size; ignored when adaptive stepping is enabled.
    pub delta_t: Decimal,
}

/// Spatial boundary condition type.
#[derive(Clone)]
pub enum SpatialBoundType {
    /// Periodic: opposite faces are identified. Standard for cosmological simulations.
    Periodic,
    /// Open / absorbing: particles leaving the box are removed.
    Open,
    /// Reflecting: particles bounce off the domain boundary.
    Reflecting,
    /// Isolated: zero-padding (2× box) Poisson solver; vacuum boundary conditions.
    Isolated,
}

/// Velocity boundary condition type.
#[derive(Clone)]
pub enum VelocityBoundType {
    /// Absorbing: particles leaving velocity box are removed from the simulation.
    Open,
    /// Hard wall: velocity domain is clipped to the box.
    Truncated,
}

/// Complete 6D computational domain specification, ready for use by solver components.
///
/// Combines spatial/velocity extents, resolutions, boundary conditions, and the
/// time range.  Construct via [`DomainBuilder`] (see [`Domain::builder`]).
/// Frequently-needed floating-point quantities (`dx`, `dv`, `lx`, `lv`) are
/// cached at construction time to avoid repeated `Decimal`-to-`f64` conversion.
#[derive(Clone)]
pub struct Domain {
    /// Spatial half-extents (Decimal).
    pub spatial: SpatialDom,
    /// Velocity half-extents (Decimal).
    pub velocity: VelocityDom,
    /// Spatial grid resolution (cell counts per axis).
    pub spatial_res: SpatialRes,
    /// Velocity grid resolution (cell counts per axis).
    pub velocity_res: VelocityRes,
    /// Simulation start and end times.
    pub time_range: TimeRange,
    /// Boundary condition applied along spatial axes.
    pub spatial_bc: SpatialBoundType,
    /// Boundary condition applied along velocity axes.
    pub velocity_bc: VelocityBoundType,
    // Cached f64 conversions (computed once at construction)
    cached_dx: [f64; 3],
    cached_dv: [f64; 3],
    cached_lx: [f64; 3],
    cached_lv: [f64; 3],
}

impl Domain {
    /// Create a `DomainBuilder` with default settings.
    pub fn builder() -> DomainBuilder {
        DomainBuilder::new()
    }

    /// Cell size in each spatial dimension: Δx = 2L / N.
    #[inline]
    pub fn dx(&self) -> [f64; 3] {
        self.cached_dx
    }

    /// Cell size in each velocity dimension: Δv = 2Lv / Nv.
    #[inline]
    pub fn dv(&self) -> [f64; 3] {
        self.cached_dv
    }

    /// Spatial half-extents as f64: [L_x1, L_x2, L_x3].
    #[inline]
    pub fn lx(&self) -> [f64; 3] {
        self.cached_lx
    }

    /// Velocity half-extents as f64: [L_v1, L_v2, L_v3].
    #[inline]
    pub fn lv(&self) -> [f64; 3] {
        self.cached_lv
    }

    fn compute_cache(
        spatial: &SpatialDom,
        velocity: &VelocityDom,
        spatial_res: &SpatialRes,
        velocity_res: &VelocityRes,
    ) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        let lx = [
            spatial.x1.to_f64().unwrap_or(1.0),
            spatial.x2.to_f64().unwrap_or(1.0),
            spatial.x3.to_f64().unwrap_or(1.0),
        ];
        let lv = [
            velocity.v1.to_f64().unwrap_or(1.0),
            velocity.v2.to_f64().unwrap_or(1.0),
            velocity.v3.to_f64().unwrap_or(1.0),
        ];
        let dx = [
            2.0 * lx[0] / spatial_res.x1 as f64,
            2.0 * lx[1] / spatial_res.x2 as f64,
            2.0 * lx[2] / spatial_res.x3 as f64,
        ];
        let dv = [
            2.0 * lv[0] / velocity_res.v1 as f64,
            2.0 * lv[1] / velocity_res.v2 as f64,
            2.0 * lv[2] / velocity_res.v3 as f64,
        ];
        (dx, dv, lx, lv)
    }

    /// Spatial cell volume dx1 * dx2 * dx3.
    #[inline]
    pub fn cell_volume_3d(&self) -> f64 {
        let dx = self.cached_dx;
        dx[0] * dx[1] * dx[2]
    }

    /// Full 6D phase-space cell volume dx1*dx2*dx3 * dv1*dv2*dv3.
    #[inline]
    pub fn cell_volume_6d(&self) -> f64 {
        let dx = self.cached_dx;
        let dv = self.cached_dv;
        dx[0] * dx[1] * dx[2] * dv[0] * dv[1] * dv[2]
    }

    /// Total number of 6D cells: Nx³ × Nv³.
    pub fn total_cells(&self) -> usize {
        (self.spatial_res.x1 as usize)
            * (self.spatial_res.x2 as usize)
            * (self.spatial_res.x3 as usize)
            * (self.velocity_res.v1 as usize)
            * (self.velocity_res.v2 as usize)
            * (self.velocity_res.v3 as usize)
    }
}

/// Fluent builder for [`Domain`].
///
/// All fields are mandatory; calling [`build`](DomainBuilder::build) returns
/// an error if any are missing.  Symmetric helpers (e.g. `spatial_extent`)
/// set all three axes to the same value; for anisotropic boxes construct
/// `SpatialDom` / `VelocityDom` directly.
pub struct DomainBuilder {
    spatial: Option<SpatialDom>,
    velocity: Option<VelocityDom>,
    spatial_res: Option<SpatialRes>,
    velocity_res: Option<VelocityRes>,
    time_range: Option<TimeRange>,
    spatial_bc: Option<SpatialBoundType>,
    velocity_bc: Option<VelocityBoundType>,
}

impl Default for DomainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainBuilder {
    /// Create a new builder with all fields unset.
    pub fn new() -> Self {
        DomainBuilder {
            spatial: None,
            velocity: None,
            spatial_res: None,
            velocity_res: None,
            time_range: None,
            spatial_bc: None,
            velocity_bc: None,
        }
    }

    /// Set symmetric spatial extent: domain spans [−L, L]³.
    pub fn spatial_extent(mut self, l: f64) -> Self {
        let L = Decimal::from_f64_retain(l).unwrap_or(Decimal::ZERO);
        self.spatial = Some(SpatialDom {
            x1: L,
            x2: L,
            x3: L,
            neg_x1: Some(-L),
            neg_x2: Some(-L),
            neg_x3: Some(-L),
        });
        self
    }

    /// Set symmetric velocity extent: domain spans [−Lv, Lv]³.
    pub fn velocity_extent(mut self, lv: f64) -> Self {
        let Lv = Decimal::from_f64_retain(lv).unwrap_or(Decimal::ZERO);
        self.velocity = Some(VelocityDom {
            v1: Lv,
            v2: Lv,
            v3: Lv,
            neg_v1: Some(-Lv),
            neg_v2: Some(-Lv),
            neg_v3: Some(-Lv),
        });
        self
    }

    /// Set spatial resolution (same in all dimensions).
    pub fn spatial_resolution(mut self, n: i128) -> Self {
        self.spatial_res = Some(SpatialRes {
            x1: n,
            x2: n,
            x3: n,
        });
        self
    }

    /// Set velocity resolution (same in all dimensions).
    pub fn velocity_resolution(mut self, n: i128) -> Self {
        self.velocity_res = Some(VelocityRes {
            v1: n,
            v2: n,
            v3: n,
        });
        self
    }

    /// Set final simulation time.
    pub fn t_final(mut self, t: f64) -> Self {
        self.time_range = Some(TimeRange {
            t0: Decimal::ZERO,
            t_final: Decimal::from_f64_retain(t).unwrap_or(Decimal::ZERO),
        });
        self
    }

    /// Set spatial boundary condition.
    pub fn spatial_bc(mut self, bc: SpatialBoundType) -> Self {
        self.spatial_bc = Some(bc);
        self
    }

    /// Set velocity boundary condition.
    pub fn velocity_bc(mut self, bc: VelocityBoundType) -> Self {
        self.velocity_bc = Some(bc);
        self
    }

    /// Set symmetric spatial extent from a Decimal value (avoids lossy f64→Decimal conversion).
    pub fn spatial_extent_decimal(mut self, l: Decimal) -> Self {
        self.spatial = Some(SpatialDom {
            x1: l,
            x2: l,
            x3: l,
            neg_x1: Some(-l),
            neg_x2: Some(-l),
            neg_x3: Some(-l),
        });
        self
    }

    /// Set symmetric velocity extent from a Decimal value.
    pub fn velocity_extent_decimal(mut self, lv: Decimal) -> Self {
        self.velocity = Some(VelocityDom {
            v1: lv,
            v2: lv,
            v3: lv,
            neg_v1: Some(-lv),
            neg_v2: Some(-lv),
            neg_v3: Some(-lv),
        });
        self
    }

    /// Set final simulation time from a Decimal value.
    pub fn t_final_decimal(mut self, t: Decimal) -> Self {
        self.time_range = Some(TimeRange {
            t0: Decimal::ZERO,
            t_final: t,
        });
        self
    }

    /// Validate that all fields are set and construct the [`Domain`].
    ///
    /// Returns an error if any required field (spatial extent, velocity extent,
    /// resolutions, t_final, or boundary conditions) has not been provided.
    pub fn build(self) -> anyhow::Result<Domain> {
        let spatial = self
            .spatial
            .ok_or_else(|| anyhow::anyhow!("missing spatial extent"))?;
        let velocity = self
            .velocity
            .ok_or_else(|| anyhow::anyhow!("missing velocity extent"))?;
        let spatial_res = self
            .spatial_res
            .ok_or_else(|| anyhow::anyhow!("missing spatial resolution"))?;
        let velocity_res = self
            .velocity_res
            .ok_or_else(|| anyhow::anyhow!("missing velocity resolution"))?;
        let time_range = self
            .time_range
            .ok_or_else(|| anyhow::anyhow!("missing t_final"))?;
        let spatial_bc = self
            .spatial_bc
            .ok_or_else(|| anyhow::anyhow!("missing spatial BC"))?;
        let velocity_bc = self
            .velocity_bc
            .ok_or_else(|| anyhow::anyhow!("missing velocity BC"))?;
        let (cached_dx, cached_dv, cached_lx, cached_lv) =
            Domain::compute_cache(&spatial, &velocity, &spatial_res, &velocity_res);
        Ok(Domain {
            spatial,
            velocity,
            spatial_res,
            velocity_res,
            time_range,
            spatial_bc,
            velocity_bc,
            cached_dx,
            cached_dv,
            cached_lx,
            cached_lv,
        })
    }
}
