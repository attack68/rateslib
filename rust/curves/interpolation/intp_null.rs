use crate::curves::nodes::NodesTimestamp;
use crate::curves::CurveInterpolation;
use crate::dual::Number;
use bincode::{deserialize, serialize};
use chrono::NaiveDateTime;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

/// Define a null interpolation object.
///
/// This is used by PyO3 binding to indicate interpolation occurs in Python.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct NullInterpolator {}

#[pymethods]
impl NullInterpolator {
    #[new]
    pub fn new() -> Self {
        NullInterpolator {}
    }

    // Pickling
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(PyTuple::empty_bound(py))
    }
}

impl CurveInterpolation for NullInterpolator {
    fn interpolated_value(&self, _nodes: &NodesTimestamp, _date: &NaiveDateTime) -> Number {
        panic!("NullInterpolator cannot be used to obtain interpolated values.");
        #[allow(unreachable_code)]
        Number::F64(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;
    use crate::curves::nodes::Nodes;
    use indexmap::IndexMap;

    fn nodes_timestamp_fixture() -> NodesTimestamp {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        NodesTimestamp::from(nodes)
    }

    #[test]
    #[should_panic]
    fn test_null_interpolation() {
        let nts = nodes_timestamp_fixture();
        let li = NullInterpolator::new();
        li.interpolated_value(&nts, &ndt(2000, 7, 1));
    }
}
