use pyo3::exceptions::PyValueError;
use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::conversion::ToPyObject;
use crate::dual::dual1::Dual;
use crate::dual::dual2::Dual2;
use crate::calendars::calendar::{Cal, UnionCal};
use crate::json::JSON;

#[derive(Serialize, Deserialize, FromPyObject)]
pub enum Serialized {
    Dual(Dual),
    Dual2(Dual2),
    Cal(Cal),
    UnionCal(UnionCal),
}

impl IntoPy<PyObject> for Serialized {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Serialized::Dual(v) => Py::new(py, v).unwrap().to_object(py),
            Serialized::Dual2(v) => Py::new(py, v).unwrap().to_object(py),
            Serialized::Cal(v) => Py::new(py, v).unwrap().to_object(py),
            Serialized::UnionCal(v) => Py::new(py, v).unwrap().to_object(py),
        }
    }
}

impl JSON for Serialized {}


#[pyfunction]
#[pyo3(name = "from_json")]
pub fn from_json_py(_py: Python<'_>, json: &str) -> PyResult<Serialized> {
    match Serialized::from_json(json) {
        Ok(v) => Ok(v),
        Err(e) => Err(PyValueError::new_err(format!("Could not create Class or Struct from given JSON.\n{}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::dual1::Dual;
    #[test]
    fn test_serialized_object() {
        let x = Dual::new(2.5, vec!["x".to_string()]);
        let json = Serialized::Dual(x.clone()).to_json().unwrap();
        println!("{}", json);
        assert_eq!(json,"{\"Dual\":{\"real\":2.5,\"vars\":[\"x\"],\"dual\":{\"v\":1,\"dim\":[1],\"data\":[1.0]}}}");

        let y = Serialized::from_json(&json).unwrap();
        match y {
            Serialized::Dual(d) => assert_eq!(x, d),
            _ => assert!(false)
        }
    }
}
