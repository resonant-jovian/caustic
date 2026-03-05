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
}

impl Domain {
    /// Create a `DomainBuilder` with default settings.
    pub fn builder() -> DomainBuilder {
        DomainBuilder::new()
    }

    /// Cell size in each spatial dimension: Δx = 2L / N.
    pub fn dx(&self) -> [f64; 3] {
        let l1 = self.spatial.x1.to_f64().unwrap();
        let l2 = self.spatial.x2.to_f64().unwrap();
        let l3 = self.spatial.x3.to_f64().unwrap();
        let n1 = self.spatial_res.x1 as f64;
        let n2 = self.spatial_res.x2 as f64;
        let n3 = self.spatial_res.x3 as f64;
        [2.0 * l1 / n1, 2.0 * l2 / n2, 2.0 * l3 / n3]
    }

    /// Cell size in each velocity dimension: Δv = 2Lv / Nv.
    pub fn dv(&self) -> [f64; 3] {
        let lv1 = self.velocity.v1.to_f64().unwrap();
        let lv2 = self.velocity.v2.to_f64().unwrap();
        let lv3 = self.velocity.v3.to_f64().unwrap();
        let nv1 = self.velocity_res.v1 as f64;
        let nv2 = self.velocity_res.v2 as f64;
        let nv3 = self.velocity_res.v3 as f64;
        [2.0 * lv1 / nv1, 2.0 * lv2 / nv2, 2.0 * lv3 / nv3]
    }

    /// Total number of 6D cells: Nx³ × Nv³. Warn if > ~10⁹.
    pub fn total_cells(&self) -> usize {
        let result = (self.spatial_res.x1 as usize)
            * (self.spatial_res.x2 as usize)
            * (self.spatial_res.x3 as usize)
            * (self.velocity_res.v1 as usize)
            * (self.velocity_res.v2 as usize)
            * (self.velocity_res.v3 as usize);
        if result > 1_000_000_000 {
            eprintln!("Warning: total_cells = {} > 10^9", result);
        }
        result
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
        let L = Decimal::from_f64_retain(l).expect("spatial extent: l should be a decimal");
        self.spatial = Some(SpatialDom {
            x1: L, x2: L, x3: L,
            neg_x1: Some(-L), neg_x2: Some(-L), neg_x3: Some(-L),
        });
        self
    }

    /// Set symmetric velocity extent: domain spans [−Lv, Lv]³.
    pub fn velocity_extent(mut self, lv: f64) -> Self {
        let Lv = Decimal::from_f64_retain(lv).expect("velocity extent");
        self.velocity = Some(VelocityDom {
            v1: Lv, v2: Lv, v3: Lv,
            neg_v1: Some(-Lv), neg_v2: Some(-Lv), neg_v3: Some(-Lv),
        });
        self
    }

    /// Set spatial resolution (same in all dimensions).
    pub fn spatial_resolution(mut self, n: i128) -> Self {
        self.spatial_res = Some(SpatialRes { x1: n, x2: n, x3: n });
        self
    }

    /// Set velocity resolution (same in all dimensions).
    pub fn velocity_resolution(mut self, n: i128) -> Self {
        self.velocity_res = Some(VelocityRes { v1: n, v2: n, v3: n });
        self
    }

    /// Set final simulation time.
    pub fn t_final(mut self, t: f64) -> Self {
        self.time_range = Some(TimeRange {
            t0: Decimal::ZERO,
            t_final: Decimal::from_f64_retain(t).expect("t_final"),
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

    /// Validate and construct the `Domain`.
    pub fn build(self) -> anyhow::Result<Domain> {
        Ok(Domain {
            spatial: self.spatial.ok_or_else(|| anyhow::anyhow!("missing spatial extent"))?,
            velocity: self.velocity.ok_or_else(|| anyhow::anyhow!("missing velocity extent"))?,
            spatial_res: self.spatial_res.ok_or_else(|| anyhow::anyhow!("missing spatial resolution"))?,
            velocity_res: self.velocity_res.ok_or_else(|| anyhow::anyhow!("missing velocity resolution"))?,
            time_range: self.time_range.ok_or_else(|| anyhow::anyhow!("missing t_final"))?,
            spatial_bc: self.spatial_bc.ok_or_else(|| anyhow::anyhow!("missing spatial BC"))?,
            velocity_bc: self.velocity_bc.ok_or_else(|| anyhow::anyhow!("missing velocity BC"))?,
        })
    }
}
