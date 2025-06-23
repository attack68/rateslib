//! Wrapper to allow de/serializable objects in Rust to be passed to/from Python using pyo3
//! bindings.

use crate::calendars::{Cal, NamedCal, UnionCal};
use crate::curves::curve_py::Curve;
use crate::dual::{Dual, Dual2};
use crate::fx::rates::FXRates;
use crate::json::JSON;
use pyo3::conversion::ToPyObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Container for all of the Python exposed Rust objects which are deserializable.
///
/// This allows a single `from_json` function to automatically detect the type and
/// convert it directly to a usable type in Python.
#[derive(Serialize, Deserialize, FromPyObject)]
pub(crate) enum DeserializedObj {
    Dual(Dual),
    Dual2(Dual2),
    Cal(Cal),
    UnionCal(UnionCal),
    NamedCal(NamedCal),
    FXRates(FXRates),
    Curve(Curve),
}

impl IntoPy<PyObject> for DeserializedObj {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            DeserializedObj::Dual(v) => Py::new(py, v).unwrap().to_object(py),
            DeserializedObj::Dual2(v) => Py::new(py, v).unwrap().to_object(py),
            DeserializedObj::Cal(v) => Py::new(py, v).unwrap().to_object(py),
            DeserializedObj::UnionCal(v) => Py::new(py, v).unwrap().to_object(py),
            DeserializedObj::NamedCal(v) => Py::new(py, v).unwrap().to_object(py),
            DeserializedObj::FXRates(v) => Py::new(py, v).unwrap().to_object(py),
            DeserializedObj::Curve(v) => Py::new(py, v).unwrap().to_object(py),
        }
    }
}

impl JSON for DeserializedObj {}

#[pyfunction]
#[pyo3(name = "from_json")]
pub(crate) fn from_json_py(_py: Python<'_>, json: &str) -> PyResult<DeserializedObj> {
    match DeserializedObj::from_json(json) {
        Ok(v) => Ok(v),
        Err(e) => Err(PyValueError::new_err(format!(
            "Could not create Class or Struct from given JSON.\n{}",
            e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::Dual;
    #[test]
    fn test_serialized_object() {
        let x = Dual::new(2.5, vec!["x".to_string()]);
        let json = DeserializedObj::Dual(x.clone()).to_json().unwrap();
        println!("{}", json);
        assert_eq!(json,"{\"Dual\":{\"real\":2.5,\"vars\":[\"x\"],\"dual\":{\"v\":1,\"dim\":[1],\"data\":[1.0]}}}");

        let y = DeserializedObj::from_json(&json).unwrap();
        match y {
            DeserializedObj::Dual(d) => assert_eq!(x, d),
            _ => assert!(false),
        }
    }
}
