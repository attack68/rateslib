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

/// Define flat forward interpolation of nodes.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct FlatForwardInterpolator {}

#[pymethods]
impl FlatForwardInterpolator {
    #[new]
    pub fn new() -> Self {
        FlatForwardInterpolator {}
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

impl CurveInterpolation for FlatForwardInterpolator {
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> Number {
        let x = date.and_utc().timestamp();
        let index = self.node_index(nodes, x);
        macro_rules! interp {
            ($Variant: ident, $indexmap: expr) => {{
                let (_x1, y1) = $indexmap.get_index(index).unwrap();
                let (x2, y2) = $indexmap.get_index(index + 1_usize).unwrap();
                if x >= *x2 {
                    Number::$Variant(y2.clone())
                } else {
                    Number::$Variant(y1.clone())
                }
            }};
        }
        match nodes {
            NodesTimestamp::F64(m) => interp!(F64, m),
            NodesTimestamp::Dual(m) => interp!(Dual, m),
            NodesTimestamp::Dual2(m) => interp!(Dual2, m),
        }
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
    fn test_flat_forward() {
        let nts = nodes_timestamp_fixture();
        let li = FlatForwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(2000, 7, 1));
        assert_eq!(result, Number::F64(1.0));
    }

    #[test]
    fn test_flat_forward_left_out_of_bounds() {
        let nts = nodes_timestamp_fixture();
        let li = FlatForwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(1999, 7, 1));
        assert_eq!(result, Number::F64(1.0));
    }

    #[test]
    fn test_flat_forward_right_out_of_bounds() {
        let nts = nodes_timestamp_fixture();
        let li = FlatForwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(2005, 7, 1));
        assert_eq!(result, Number::F64(0.98));
    }

    #[test]
    fn test_flat_forward_equals_interval_value() {
        let nts = nodes_timestamp_fixture();
        let li = FlatForwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(2001, 1, 1));
        assert_eq!(result, Number::F64(0.99));
    }
}
