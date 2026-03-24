/// Custom serde for `Decimal` that reads/writes as f64 in TOML/JSON.
///
/// Use with `#[serde(with = "decimal_serde")]` on `Decimal` fields.
pub mod decimal_serde {
    use rust_decimal::Decimal;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(val: &Decimal, s: S) -> Result<S::Ok, S::Error> {
        use rust_decimal::prelude::ToPrimitive;
        val.to_f64().unwrap_or(0.0).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Decimal, D::Error> {
        let f = f64::deserialize(d)?;
        Decimal::from_f64_retain(f).ok_or_else(|| serde::de::Error::custom("invalid decimal"))
    }
}
