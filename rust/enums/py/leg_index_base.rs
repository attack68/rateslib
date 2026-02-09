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

use crate::enums::parameters::LegIndexBase;
use crate::json::{DeserializedObj, JSON};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl LegIndexBase {
    // JSON
    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::LegIndexBase(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `LegIndexBase` to JSON.",
            )),
        }
    }

    // Pickling
    #[new]
    fn new_py(item: usize) -> PyResult<LegIndexBase> {
        match item {
            _ if item == LegIndexBase::Initial as usize => Ok(LegIndexBase::Initial),
            _ if item == LegIndexBase::PeriodOnPeriod as usize => Ok(LegIndexBase::PeriodOnPeriod),
            _ => Err(PyValueError::new_err(
                "unreachable code on LegIndexBase pickle. Please report",
            )),
        }
    }
    fn __getnewargs__<'py>(&self) -> PyResult<(usize,)> {
        Ok((*self as usize,))
    }

    fn __repr__(&self) -> String {
        format!("<rl.LegIndexBase.{:?} at {:p}>", self, self)
    }
}
