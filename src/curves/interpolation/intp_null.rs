use crate::curves::nodes::NodesTimestamp;
use crate::curves::CurveInterpolation;
use crate::dual::DualsOrF64;
use chrono::NaiveDateTime;
use pyo3::{pyclass, pymethods};
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
}

impl CurveInterpolation for NullInterpolator {
    fn interpolated_value(&self, _nodes: &NodesTimestamp, _date: &NaiveDateTime) -> DualsOrF64 {
        panic!("NullInterpolator cannot be used to obtain interpolated values.");
        #[allow(unreachable_code)]
        DualsOrF64::F64(0.0)
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
