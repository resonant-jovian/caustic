use rust_decimal::Decimal;

pub const G: Decimal = Decimal::ONE;

pub enum BCSpatial {
    Periodic,
    Open,
    Isolated,
    Reflecting,
}
