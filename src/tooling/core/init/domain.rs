//! Domain specification structs: spatial/velocity extents, resolutions, boundary conditions.
//! All extent/time fields use `rust_decimal::Decimal` for exact arithmetic.

use rust_decimal::Decimal;

/// Spatial extent of the simulation domain in each dimension.
pub struct SpatialDom {
    pub x1: Decimal,
    pub x2: Decimal,
    pub x3: Decimal,
    pub neg_x1: Option<Decimal>,
    pub neg_x2: Option<Decimal>,
    pub neg_x3: Option<Decimal>,
}

/// Velocity extent of the simulation domain in each dimension.
pub struct VelocityDom {
    pub v1: Decimal,
    pub v2: Decimal,
    pub v3: Decimal,
    pub neg_v1: Option<Decimal>,
    pub neg_v2: Option<Decimal>,
    pub neg_v3: Option<Decimal>,
}

/// Number of grid cells in each spatial dimension.
pub struct SpatialRes {
    pub x1: i128,
    pub x2: i128,
    pub x3: i128,
}

/// Number of grid cells in each velocity dimension.
pub struct VelocityRes {
    pub v1: i128,
    pub v2: i128,
    pub v3: i128,
}

/// Time range for the simulation.
pub struct TimeRange {
    pub t0: Decimal,
    pub t_final: Decimal,
}

/// Integration timestep. May be adaptive (use OptionalParams) or fixed.
pub struct Timestep {
    pub delta_t: Decimal,
}

/// Spatial boundary condition type.
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
pub enum VelocityBoundType {
    /// Absorbing: particles leaving velocity box are removed from the simulation.
    Open,
    /// Hard wall: velocity domain is clipped to the box.
    Truncated,
}

/// Complete computational domain specification, ready for use by solver components.
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
        todo!()
    }

    /// Cell size in each spatial dimension: Δx = 2L / N.
    pub fn dx(&self) -> [f64; 3] {
        todo!("2*L / N")
    }

    /// Cell size in each velocity dimension: Δv = 2Lv / Nv.
    pub fn dv(&self) -> [f64; 3] {
        todo!("2*Lv / Nv")
    }

    /// Total number of 6D cells: Nx³ × Nv³. Warn if > ~10⁹.
    pub fn total_cells(&self) -> usize {
        todo!("Nx^3 * Nv^3 -- warn if > ~10^9")
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
        todo!()
    }

    /// Set symmetric spatial extent: domain spans [−L, L]³.
    pub fn spatial_extent(mut self, l: f64) -> Self {
        todo!()
    }

    /// Set symmetric velocity extent: domain spans [−Lv, Lv]³.
    pub fn velocity_extent(mut self, lv: f64) -> Self {
        todo!()
    }

    /// Set spatial resolution (same in all dimensions).
    pub fn spatial_resolution(mut self, n: i128) -> Self {
        todo!()
    }

    /// Set velocity resolution (same in all dimensions).
    pub fn velocity_resolution(mut self, n: i128) -> Self {
        todo!()
    }

    /// Set final simulation time.
    pub fn t_final(mut self, t: f64) -> Self {
        todo!()
    }

    /// Set spatial boundary condition.
    pub fn spatial_bc(mut self, bc: SpatialBoundType) -> Self {
        todo!()
    }

    /// Set velocity boundary condition.
    pub fn velocity_bc(mut self, bc: VelocityBoundType) -> Self {
        todo!()
    }

    /// Validate and construct the `Domain`. Returns error if Lv < escape velocity estimate.
    pub fn build(self) -> anyhow::Result<Domain> {
        todo!("validate Lv >= escape velocity estimate")
    }
}
