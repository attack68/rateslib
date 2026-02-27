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

use crate::enums::IROptionMetric;
use crate::json::{DeserializedObj, JSON};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Deserialize, Serialize};

/// Enumerable type for IR Option rate metrics.
///
/// .. rubric:: Variants
///
/// .. ipython:: python
///    :suppress:
///
///    from rateslib.rs import IROptionMetric
///    variants = [item for item in IROptionMetric.__dict__ if \
///        "__" != item[:2] and \
///        item not in ['to_json', 'method_param'] \
///    ]
///
/// .. ipython:: python
///
///    variants
///
#[pyclass(module = "rateslib.rs", name = "IROptionMetric", eq, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum PyIROptionMetric {
    #[pyo3(constructor = (_u8=0))]
    NormalVol { _u8: u8 },
    #[pyo3(constructor = (_u8=1))]
    LogNormalVol { _u8: u8 },
    #[pyo3(constructor = (_u8=2))]
    PercentNotional { _u8: u8 },
    #[pyo3(constructor = (_u8=3))]
    Cash { _u8: u8 },
    #[pyo3(constructor = (param, _u8=4))]
    BlackVolShift { param: i32, _u8: u8 },
}

/// Used for providing pickle support for PyIROptionMetric
enum PyIROptionMetricNewArgs {
    NoArgs(u8),
    I32(i32, u8),
}

impl<'py> IntoPyObject<'py> for PyIROptionMetricNewArgs {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            PyIROptionMetricNewArgs::NoArgs(x) => Ok((x,).into_pyobject(py).unwrap()),
            PyIROptionMetricNewArgs::I32(x, y) => Ok((x, y).into_pyobject(py).unwrap()),
        }
    }
}

impl<'py> FromPyObject<'py, 'py> for PyIROptionMetricNewArgs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let ext: PyResult<(u8,)> = obj.extract();
        if ext.is_ok() {
            let (x,) = ext.unwrap();
            return Ok(PyIROptionMetricNewArgs::NoArgs(x));
        }
        let ext: PyResult<(i32, u8)> = obj.extract();
        if ext.is_ok() {
            let (x, y) = ext.unwrap();
            return Ok(PyIROptionMetricNewArgs::I32(x, y));
        }
        Err(PyValueError::new_err("Undefined behaviour"))
    }
}

impl From<IROptionMetric> for PyIROptionMetric {
    fn from(value: IROptionMetric) -> Self {
        match value {
            IROptionMetric::NormalVol {} => PyIROptionMetric::NormalVol { _u8: 0 },
            IROptionMetric::LogNormalVol {} => PyIROptionMetric::LogNormalVol { _u8: 1 },
            IROptionMetric::PercentNotional {} => PyIROptionMetric::PercentNotional { _u8: 2 },
            IROptionMetric::Cash {} => PyIROptionMetric::Cash { _u8: 3 },
            IROptionMetric::BlackVolShift(n) => {
                PyIROptionMetric::BlackVolShift { param: n, _u8: 4 }
            }
        }
    }
}

impl From<PyIROptionMetric> for IROptionMetric {
    fn from(value: PyIROptionMetric) -> Self {
        match value {
            PyIROptionMetric::NormalVol { _u8: _ } => IROptionMetric::NormalVol {},
            PyIROptionMetric::LogNormalVol { _u8: _ } => IROptionMetric::LogNormalVol {},
            PyIROptionMetric::PercentNotional { _u8: _ } => IROptionMetric::PercentNotional {},
            PyIROptionMetric::Cash { _u8: _ } => IROptionMetric::Cash {},
            PyIROptionMetric::BlackVolShift { param: n, _u8: _ } => {
                IROptionMetric::BlackVolShift(n)
            }
        }
    }
}

#[pymethods]
impl PyIROptionMetric {
    /// Return the shift associated with the Black Vol metric.
    ///
    /// Returns
    /// -------
    /// int
    #[pyo3(name = "shift")]
    fn shift_py(&self) -> i32 {
        match self {
            PyIROptionMetric::BlackVolShift { param: n, _u8: _ } => *n,
            _ => 0_i32,
        }
    }

    fn __str__(&self) -> String {
        match self {
            PyIROptionMetric::NormalVol { _u8: _ } => "normal_vol".to_string(),
            PyIROptionMetric::LogNormalVol { _u8: _ } => "log_normal_vol".to_string(),
            PyIROptionMetric::PercentNotional { _u8: _ } => "percent_notional".to_string(),
            PyIROptionMetric::Cash { _u8: _ } => "cash".to_string(),
            PyIROptionMetric::BlackVolShift { param: n, _u8: _ } => {
                format!("black_vol_shift_{}", n)
            }
        }
    }

    fn __getnewargs__(&self) -> PyIROptionMetricNewArgs {
        match self {
            PyIROptionMetric::NormalVol { _u8: u } => PyIROptionMetricNewArgs::NoArgs(*u),
            PyIROptionMetric::LogNormalVol { _u8: u } => PyIROptionMetricNewArgs::NoArgs(*u),
            PyIROptionMetric::PercentNotional { _u8: u } => PyIROptionMetricNewArgs::NoArgs(*u),
            PyIROptionMetric::Cash { _u8: u } => PyIROptionMetricNewArgs::NoArgs(*u),
            PyIROptionMetric::BlackVolShift { param: n, _u8: u } => {
                PyIROptionMetricNewArgs::I32(*n, *u)
            }
        }
    }

    #[new]
    fn new_py(args: PyIROptionMetricNewArgs) -> PyIROptionMetric {
        match args {
            PyIROptionMetricNewArgs::NoArgs(0) => PyIROptionMetric::NormalVol { _u8: 0 },
            PyIROptionMetricNewArgs::NoArgs(1) => PyIROptionMetric::LogNormalVol { _u8: 1 },
            PyIROptionMetricNewArgs::NoArgs(2) => PyIROptionMetric::PercentNotional { _u8: 2 },
            PyIROptionMetricNewArgs::NoArgs(3) => PyIROptionMetric::Cash { _u8: 3 },
            PyIROptionMetricNewArgs::I32(n, 4) => {
                PyIROptionMetric::BlackVolShift { param: n, _u8: 4 }
            }
            _ => panic!("Undefined behaviour."),
        }
    }

    fn __repr__(&self) -> String {
        let metric: IROptionMetric = (*self).into();
        format!("<rl.IROptionMetric.{:?} at {:p}>", metric, self)
    }

    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::PyIROptionMetric(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `IROptionMetric` to JSON.",
            )),
        }
    }
}
