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

use internment::Intern;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};

/// A currency identified by 3-ascii ISO code.
#[pyclass(module = "rateslib.rs", from_py_object)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Ccy {
    pub(crate) name: Intern<String>,
}

impl Ccy {
    /// Constructs a new `Ccy`.
    ///
    /// Use **only** 3-ascii names. e.g. *"usd"*, aligned with ISO representation. `name` is converted
    /// to lowercase to promote performant equality between "USD" and "usd".
    ///
    /// Panics if `name` is not 3 bytes in length.
    pub fn try_new(name: &str) -> Result<Self, PyErr> {
        let ccy: String = name.to_string().to_lowercase();
        if ccy.len() != 3 {
            return Err(PyValueError::new_err(
                "`Ccy` must be 3 ascii character in length, e.g. 'usd'.",
            ));
        }
        Ok(Ccy {
            name: Intern::new(ccy),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccy_creation() {
        let a = Ccy::try_new("usd").unwrap();
        let b = Ccy::try_new("USD").unwrap();
        assert_eq!(a, b)
    }

    #[test]
    fn ccy_creation_error() {
        match Ccy::try_new("FOUR") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }
}
