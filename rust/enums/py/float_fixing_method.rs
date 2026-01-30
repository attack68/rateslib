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

//! Wrapper module to export to Python using pyo3 bindings.

use crate::enums::FloatFixingMethod;
use crate::json::{DeserializedObj, JSON};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Deserialize, Serialize};

/// Python wrapper for Adjuster to facilitate complex enum pickling.
#[pyclass(module = "rateslib.rs", name = "FloatFixingMethod", eq)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum PyFloatFixingMethod {
    #[pyo3(constructor = (_u8=0))]
    RFRPaymentDelay { _u8: u8 },
    #[pyo3(constructor = (param, _u8=1))]
    RFRObservationShift { param: i32, _u8: u8 },
    #[pyo3(constructor = (param, _u8=2))]
    RFRLockout { param: i32, _u8: u8 },
    #[pyo3(constructor = (param, _u8=3))]
    RFRLookback { param: i32, _u8: u8 },
    #[pyo3(constructor = (_u8=4))]
    RFRPaymentDelayAverage { _u8: u8 },
    #[pyo3(constructor = (param, _u8=5))]
    RFRObservationShiftAverage { param: i32, _u8: u8 },
    #[pyo3(constructor = (param, _u8=6))]
    RFRLockoutAverage { param: i32, _u8: u8 },
    #[pyo3(constructor = (param, _u8=7))]
    RFRLookbackAverage { param: i32, _u8: u8 },
    #[pyo3(constructor = (param, _u8=8))]
    IBOR { param: i32, _u8: u8 },
}

/// Used for providing pickle support for PyFloatFixingMethod
enum PyFloatFixingMethodNewArgs {
    NoArgs(u8),
    I32(i32, u8),
}

impl<'py> IntoPyObject<'py> for PyFloatFixingMethodNewArgs {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            PyFloatFixingMethodNewArgs::NoArgs(x) => Ok((x,).into_pyobject(py).unwrap()),
            PyFloatFixingMethodNewArgs::I32(x, y) => Ok((x, y).into_pyobject(py).unwrap()),
        }
    }
}

impl<'py> FromPyObject<'py, 'py> for PyFloatFixingMethodNewArgs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let ext: PyResult<(u8,)> = obj.extract();
        if ext.is_ok() {
            let (x,) = ext.unwrap();
            return Ok(PyFloatFixingMethodNewArgs::NoArgs(x));
        }
        let ext: PyResult<(i32, u8)> = obj.extract();
        if ext.is_ok() {
            let (x, y) = ext.unwrap();
            return Ok(PyFloatFixingMethodNewArgs::I32(x, y));
        }
        Err(PyValueError::new_err("Undefined behaviour"))
    }
}

impl From<FloatFixingMethod> for PyFloatFixingMethod {
    fn from(value: FloatFixingMethod) -> Self {
        match value {
            FloatFixingMethod::RFRPaymentDelay {} => {
                PyFloatFixingMethod::RFRPaymentDelay { _u8: 0 }
            }
            FloatFixingMethod::RFRObservationShift(n) => {
                PyFloatFixingMethod::RFRObservationShift { param: n, _u8: 1 }
            }
            FloatFixingMethod::RFRLockout(n) => {
                PyFloatFixingMethod::RFRLockout { param: n, _u8: 2 }
            }
            FloatFixingMethod::RFRLookback(n) => {
                PyFloatFixingMethod::RFRLookback { param: n, _u8: 3 }
            }
            FloatFixingMethod::RFRPaymentDelayAverage {} => {
                PyFloatFixingMethod::RFRPaymentDelayAverage { _u8: 4 }
            }
            FloatFixingMethod::RFRObservationShiftAverage(n) => {
                PyFloatFixingMethod::RFRObservationShiftAverage { param: n, _u8: 5 }
            }
            FloatFixingMethod::RFRLockoutAverage(n) => {
                PyFloatFixingMethod::RFRLockoutAverage { param: n, _u8: 6 }
            }
            FloatFixingMethod::RFRLookbackAverage(n) => {
                PyFloatFixingMethod::RFRLookbackAverage { param: n, _u8: 7 }
            }
            FloatFixingMethod::IBOR(n) => PyFloatFixingMethod::IBOR { param: n, _u8: 8 },
        }
    }
}

impl From<PyFloatFixingMethod> for FloatFixingMethod {
    fn from(value: PyFloatFixingMethod) -> Self {
        match value {
            PyFloatFixingMethod::RFRPaymentDelay { _u8: _ } => {
                FloatFixingMethod::RFRPaymentDelay {}
            }
            PyFloatFixingMethod::RFRObservationShift { param: n, _u8: _ } => {
                FloatFixingMethod::RFRObservationShift(n)
            }
            PyFloatFixingMethod::RFRLockout { param: n, _u8: _ } => {
                FloatFixingMethod::RFRLockout(n)
            }
            PyFloatFixingMethod::RFRLookback { param: n, _u8: _ } => {
                FloatFixingMethod::RFRLookback(n)
            }
            PyFloatFixingMethod::RFRPaymentDelayAverage { _u8: _ } => {
                FloatFixingMethod::RFRPaymentDelayAverage {}
            }
            PyFloatFixingMethod::RFRObservationShiftAverage { param: n, _u8: _ } => {
                FloatFixingMethod::RFRObservationShiftAverage(n)
            }
            PyFloatFixingMethod::RFRLockoutAverage { param: n, _u8: _ } => {
                FloatFixingMethod::RFRLockoutAverage(n)
            }
            PyFloatFixingMethod::RFRLookbackAverage { param: n, _u8: _ } => {
                FloatFixingMethod::RFRLookbackAverage(n)
            }
            PyFloatFixingMethod::IBOR { param: n, _u8: _ } => FloatFixingMethod::IBOR(n),
        }
    }
}

#[pymethods]
impl PyFloatFixingMethod {
    /// Return a parameter associated with the fixing method.
    ///
    /// Returns
    /// -------
    /// int
    #[pyo3(name = "method_param")]
    fn method_param_py(&self) -> i32 {
        let fixing_method: FloatFixingMethod = (*self).into();
        fixing_method.method_param()
    }

    fn __str__(&self) -> String {
        match self {
            PyFloatFixingMethod::RFRPaymentDelay { _u8: _ } => "rfr_payment_delay".to_string(),
            PyFloatFixingMethod::RFRObservationShift { param: _, _u8: _ } => {
                "rfr_observation_shift".to_string()
            }
            PyFloatFixingMethod::RFRLockout { param: _, _u8: _ } => "rfr_lockout".to_string(),
            PyFloatFixingMethod::RFRLookback { param: _, _u8: _ } => "rfr_lookback".to_string(),
            PyFloatFixingMethod::RFRPaymentDelayAverage { _u8: _ } => {
                "rfr_payment_delay_avg".to_string()
            }
            PyFloatFixingMethod::RFRObservationShiftAverage { param: _, _u8: _ } => {
                "rfr_observation_shift_avg".to_string()
            }
            PyFloatFixingMethod::RFRLockoutAverage { param: _, _u8: _ } => {
                "rfr_lockout_avg".to_string()
            }
            PyFloatFixingMethod::RFRLookbackAverage { param: _, _u8: _ } => {
                "rfr_lookback_avg".to_string()
            }
            PyFloatFixingMethod::IBOR { param: _, _u8: _ } => "ibor".to_string(),
        }
    }

    fn __getnewargs__(&self) -> PyFloatFixingMethodNewArgs {
        match self {
            PyFloatFixingMethod::RFRPaymentDelay { _u8: u } => {
                PyFloatFixingMethodNewArgs::NoArgs(*u)
            }
            PyFloatFixingMethod::RFRObservationShift { param: n, _u8: u } => {
                PyFloatFixingMethodNewArgs::I32(*n, *u)
            }
            PyFloatFixingMethod::RFRLockout { param: n, _u8: u } => {
                PyFloatFixingMethodNewArgs::I32(*n, *u)
            }
            PyFloatFixingMethod::RFRLookback { param: n, _u8: u } => {
                PyFloatFixingMethodNewArgs::I32(*n, *u)
            }
            PyFloatFixingMethod::RFRPaymentDelayAverage { _u8: u } => {
                PyFloatFixingMethodNewArgs::NoArgs(*u)
            }
            PyFloatFixingMethod::RFRObservationShiftAverage { param: n, _u8: u } => {
                PyFloatFixingMethodNewArgs::I32(*n, *u)
            }
            PyFloatFixingMethod::RFRLockoutAverage { param: n, _u8: u } => {
                PyFloatFixingMethodNewArgs::I32(*n, *u)
            }
            PyFloatFixingMethod::RFRLookbackAverage { param: n, _u8: u } => {
                PyFloatFixingMethodNewArgs::I32(*n, *u)
            }
            PyFloatFixingMethod::IBOR { param: n, _u8: u } => {
                PyFloatFixingMethodNewArgs::I32(*n, *u)
            }
        }
    }

    #[new]
    fn new_py(args: PyFloatFixingMethodNewArgs) -> PyFloatFixingMethod {
        match args {
            PyFloatFixingMethodNewArgs::NoArgs(0) => {
                PyFloatFixingMethod::RFRPaymentDelay { _u8: 0 }
            }
            PyFloatFixingMethodNewArgs::I32(n, 1) => {
                PyFloatFixingMethod::RFRObservationShift { param: n, _u8: 1 }
            }
            PyFloatFixingMethodNewArgs::I32(n, 2) => {
                PyFloatFixingMethod::RFRLockout { param: n, _u8: 2 }
            }
            PyFloatFixingMethodNewArgs::I32(n, 3) => {
                PyFloatFixingMethod::RFRLookback { param: n, _u8: 3 }
            }
            PyFloatFixingMethodNewArgs::NoArgs(4) => {
                PyFloatFixingMethod::RFRPaymentDelayAverage { _u8: 4 }
            }
            PyFloatFixingMethodNewArgs::I32(n, 5) => {
                PyFloatFixingMethod::RFRObservationShiftAverage { param: n, _u8: 5 }
            }
            PyFloatFixingMethodNewArgs::I32(n, 6) => {
                PyFloatFixingMethod::RFRLockoutAverage { param: n, _u8: 6 }
            }
            PyFloatFixingMethodNewArgs::I32(n, 7) => {
                PyFloatFixingMethod::RFRLookbackAverage { param: n, _u8: 7 }
            }
            PyFloatFixingMethodNewArgs::I32(n, 8) => PyFloatFixingMethod::IBOR { param: n, _u8: 8 },
            _ => panic!("Undefined behaviour."),
        }
    }

    fn __repr__(&self) -> String {
        let fixing_method: FloatFixingMethod = (*self).into();
        format!("<rl.FloatFixingMethod.{:?} at {:p}>", fixing_method, self)
    }

    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::PyFloatFixingMethod(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `FloatFixingMethod` to JSON.",
            )),
        }
    }
}
