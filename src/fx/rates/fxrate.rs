use crate::dual::dual::DualsOrF64;
use crate::fx::rates::fxpair::FXPair;
use chrono::NaiveDateTime;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};

/// An FX rate containing `FXPair`, `rate` and `settlement` info.
#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FXRate {
    pub(crate) pair: FXPair,
    pub(crate) rate: DualsOrF64,
    pub(crate) settlement: Option<NaiveDateTime>,
}

impl FXRate {
    pub fn try_new(
        lhs: &str,
        rhs: &str,
        rate: DualsOrF64,
        settlement: Option<NaiveDateTime>,
    ) -> Result<Self, PyErr> {
        Ok(FXRate {
            pair: FXPair::try_new(lhs, rhs)?,
            rate,
            settlement,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fxrate_creation() {
        FXRate::try_new("usd", "eur", DualsOrF64::F64(1.20), None).unwrap();
    }
}
