//! Domain specification structs: spatial/velocity extents, resolutions, boundary conditions.
//! All extent/time fields use `rust_decimal::Decimal` for exact arithmetic.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Spatial extent of the simulation domain in each dimension.
#[derive(Clone)]
pub struct SpatialDom {
    pub x1: Decimal,
    pub x2: Decimal,
    pub x3: Decimal,
    pub neg_x1: Option<Decimal>,
    pub neg_x2: Option<Decimal>,
    pub neg_x3: Option<Decimal>,
}

/// Velocity extent of the simulation domain in each dimension.
#[derive(Clone)]
pub struct VelocityDom {
    pub v1: Decimal,
    pub v2: Decimal,
    pub v3: Decimal,
    pub neg_v1: Option<Decimal>,
    pub neg_v2: Option<Decimal>,
    pub neg_v3: Option<Decimal>,
}

/// Number of grid cells in each spatial dimension.
#[derive(Clone)]
pub struct SpatialRes {
    pub x1: i128,
    pub x2: i128,
    pub x3: i128,
}

/// Number of grid cells in each velocity dimension.
#[derive(Clone)]
pub struct VelocityRes {
    pub v1: i128,
    pub v2: i128,
    pub v3: i128,
}

/// Time range for the simulation.
#[derive(Clone)]
pub struct TimeRange {
    pub t0: Decimal,
    pub t_final: Decimal,
}

/// Integration timestep. May be adaptive (use OptionalParams) or fixed.
#[derive(Clone)]
pub struct Timestep {
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

/// Complete computational domain specification, ready for use by solver components.
#[derive(Clone)]
pub struct Domain {
    pub spatial: SpatialDom,
    pub velocity: VelocityDom,
    pub spatial_res: SpatialRes,
    pub velocity_res: VelocityRes,
    pub time_range: TimeRange,
    pub spatial_bc: SpatialBoundType,
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

/// Builder for `Domain` following a fluent API.
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

    /// Validate and construct the `Domain`.
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
