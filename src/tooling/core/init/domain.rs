use rust_decimal::Decimal;
pub struct SpatialDom {
    x1: Decimal,
    x2: Decimal,
    x3: Decimal,
    neg_x1: Option<Decimal>,
    neg_x2: Option<Decimal>,
    neg_x3: Option<Decimal>,
}
pub struct VelocityDom {
    v1: Decimal,
    v2: Decimal,
    v3: Decimal,
    neg_v1: Option<Decimal>,
    neg_v2: Option<Decimal>,
    neg_v3: Option<Decimal>,
}
pub struct SpatialRes {
    x1: i128,
    x2: i128,
    x3: i128,
}
pub struct VelocityRes {
    v1: i128,
    v2: i128,
    v3: i128,
}
pub struct TimeRange {
    t0: Decimal,
    t_final: Decimal,
}
pub struct Timestep {
    Δt: Decimal,
}
pub enum SpatialBoundType {

}
pub enum VelocityBoundType {

}
