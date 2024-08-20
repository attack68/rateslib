use crate::curves::nodes::NodesTimestamp;
use crate::curves::CurveInterpolation;
use crate::dual::{DualsOrF64};
use chrono::NaiveDateTime;
use pyo3::{Bound, pyclass, pymethods, PyResult, Python};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;
use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Define flat backward interpolation of nodes.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct FlatBackwardInterpolator {}

#[pymethods]
impl FlatBackwardInterpolator {
    #[new]
    pub fn new() -> Self {
        FlatBackwardInterpolator {}
    }

    // Pickling
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<()> {
        Ok(())
    }
}

impl CurveInterpolation for FlatBackwardInterpolator {
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> DualsOrF64 {
        let x = date.and_utc().timestamp();
        let index = self.node_index(nodes, x);
        macro_rules! interp {
            ($Variant: ident, $indexmap: expr) => {{
                let (x1, y1) = $indexmap.get_index(index).unwrap();
                let (_x2, y2) = $indexmap.get_index(index + 1_usize).unwrap();
                if x <= *x1 {
                    DualsOrF64::$Variant(y1.clone())
                } else {
                    DualsOrF64::$Variant(y2.clone())
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
    fn test_flat_backward() {
        let nts = nodes_timestamp_fixture();
        let li = FlatBackwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(2000, 7, 1));
        assert_eq!(result, DualsOrF64::F64(0.99));
    }

    #[test]
    fn test_flat_backward_left_out_of_bounds() {
        let nts = nodes_timestamp_fixture();
        let li = FlatBackwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(1999, 7, 1));
        assert_eq!(result, DualsOrF64::F64(1.0));
    }

    #[test]
    fn test_flat_backward_right_out_of_bounds() {
        let nts = nodes_timestamp_fixture();
        let li = FlatBackwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(2005, 7, 1));
        assert_eq!(result, DualsOrF64::F64(0.98));
    }

    #[test]
    fn test_flat_backward_equals_interval_value() {
        let nts = nodes_timestamp_fixture();
        let li = FlatBackwardInterpolator::new();
        let result = li.interpolated_value(&nts, &ndt(2001, 1, 1));
        assert_eq!(result, DualsOrF64::F64(0.99));
    }
}
