//! Wrapper to allow de/serializable objects in Rust to be passed to/from Python using pyo3
//! bindings.
//!
//! Any pyclass that is serializable is added as a DeserializedObj and then converted to JSON.
//! Having been deserialized it is matched, unpacked and passed back to Python.
//!

use crate::curves::curve_py::Curve;
use crate::dual::{Dual, Dual2};
use crate::fx::rates::FXRates;
use crate::json::JSON;
use crate::scheduling::{
    Cal, Convention, Frequency, Imm, NamedCal, PyAdjuster, RollDay, Schedule, StubInference,
    UnionCal,
};
use crate::splines::{PPSplineDual, PPSplineDual2, PPSplineF64};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Container for all of the Python exposed Rust objects which are deserializable.
///
/// This allows a single `from_json` function to automatically detect the type and
/// convert it directly to a usable type in Python.
#[derive(Serialize, Deserialize, FromPyObject, IntoPyObject)]
pub(crate) enum DeserializedObj {
    Dual(Dual),
    Dual2(Dual2),
    Cal(Cal),
    UnionCal(UnionCal),
    NamedCal(NamedCal),
    FXRates(FXRates),
    Curve(Curve),
    PPSplineF64(PPSplineF64),
    PPSplineDual(PPSplineDual),
    PPSplineDual2(PPSplineDual2),
    StubInference(StubInference),
    Imm(Imm),
    RollDay(RollDay),
    Frequency(Frequency),
    PyAdjuster(PyAdjuster),
    Schedule(Schedule),
    Convention(Convention),
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
