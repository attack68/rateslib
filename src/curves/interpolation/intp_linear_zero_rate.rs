use crate::curves::interpolation::utils::linear_zero_interp;
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

/// Define linear zero rate interpolation of nodes.
///
/// This interpolation can only be used with discount factors node values.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearZeroRateInterpolator {}

#[pymethods]

impl LinearZeroRateInterpolator {
    #[new]
    pub fn new() -> Self {
        LinearZeroRateInterpolator {}
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

impl CurveInterpolation for LinearZeroRateInterpolator {
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> Number {
        let x = date.and_utc().timestamp();
        let index = self.node_index(nodes, x);

        macro_rules! interp {
            ($Variant: ident, $indexmap: expr) => {{
                let (x0, _) = $indexmap.get_index(0_usize).unwrap();
                let (x2, y2) = $indexmap.get_index(index + 1_usize).unwrap();
                let (x1, y1) = $indexmap.get_index(index).unwrap();
                Number::$Variant(linear_zero_interp(
                    *x0 as f64, *x1 as f64, y1, *x2 as f64, y2, x as f64,
                ))
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
    fn test_log_linear() {
        let nts = nodes_timestamp_fixture();
        let ll = LinearZeroRateInterpolator::new();
        let result = ll.interpolated_value(&nts, &ndt(2001, 7, 1));
        // r1 = -ln(0.99) / 366, r2 = -ln(0.98) / 731
        // r = r1 + (181 / 365) * (r2 - r1)
        // expected = exp(-r * 547) r1 = 0.985044328
        assert_eq!(result, Number::F64(0.9850443279738612));
    }

    #[test]
    fn test_log_linear_first_period() {
        let nts = nodes_timestamp_fixture();
        let ll = LinearZeroRateInterpolator::new();
        let result = ll.interpolated_value(&nts, &ndt(2000, 7, 1));
        // r1 = r2, r2 = -ln(0.99) / 366
        // r = r1
        // expected = exp(-r * 182) = 0.99501476
        assert_eq!(result, Number::F64(0.9950147597711371));
    }
}
