// SPDX-License-Identifier: LicenseRef-Rateslib-Dual
//
// Copyright (c) 2026 Siffrorna Technology Limited
// This code cannot be used or copied externally
//
// Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
// Source-available, not open source.
//
// See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
// and/or contact info (at) rateslib (dot) com
////////////////////////////////////////////////////////////////////////////////////////////////////

use crate::fx::rates::ccy::Ccy;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A container of a two-pair `Ccy` cross.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FXPair(pub(crate) Ccy, pub(crate) Ccy);

impl FXPair {
    /// Constructs a new `FXPair`, as a combination of two distinct `Ccy`s.
    pub fn try_new(lhs: &str, rhs: &str) -> Result<Self, PyErr> {
        let lhs_ = Ccy::try_new(lhs)?;
        let rhs_ = Ccy::try_new(rhs)?;
        if lhs_ == rhs_ {
            return Err(PyValueError::new_err(
                "`FXPair` must be created from two distinct currencies, not same.",
            ));
        }
        Ok(FXPair(lhs_, rhs_))
    }
}

impl fmt::Display for FXPair {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.0.name, self.1.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fxpair_creation() {
        let a = FXPair::try_new("usd", "eur").unwrap();
        let b = FXPair::try_new("USD", "EUR").unwrap();
        assert_eq!(a, b)
    }

    #[test]
    fn fxpair_creation_error() {
        match FXPair::try_new("usd", "USD") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }
}
