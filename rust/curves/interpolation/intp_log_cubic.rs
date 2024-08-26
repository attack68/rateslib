use crate::curves::interpolation::utils::log_linear_interp;
use crate::curves::nodes::NodesTimestamp;
use crate::curves::CurveInterpolation;
use crate::dual::DualsOrF64;
use bincode::{deserialize, serialize};
use chrono::NaiveDateTime;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

/// Define log-linear interpolation of nodes.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogCubicInterpolator<T: NumberMapping> {
    spline: T
}

#[pymethods]
impl<T> LogCubicInterpolator
where          T: PartialOrd + Signed + Clone + Sum + Zero,
   for<'a> &'a T: Sub<&'a T, Output = T>,
   for<'a> &'a f64: Mul<&'a T, Output = T>,
{
    #[new]
    pub fn new(t: Vec<f64>, c: Option<Vec<T>>) -> Self {
        let spline: PPSpline<T> = PPSpline.new(3_usize, t, c);
        LogCubicInterpolator {
            spline
        }
    }

    // Pickling
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<(Vec<f64>, Option<Vec<T>>)> {
        Ok((self.t.clone(), ))
    }
}

impl CurveInterpolation for LogLinearInterpolator {
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> DualsOrF64 {
        let x = date.and_utc().timestamp();
        let index = self.node_index(nodes, x);

        macro_rules! interp {
            ($Variant: ident, $indexmap: expr) => {{
                let (x1, y1) = $indexmap.get_index(index).unwrap();
                let (x2, y2) = $indexmap.get_index(index + 1_usize).unwrap();
                DualsOrF64::$Variant(log_linear_interp(*x1 as f64, y1, *x2 as f64, y2, x as f64))
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
        let ll = LogLinearInterpolator::new();
        let result = ll.interpolated_value(&nts, &ndt(2000, 7, 1));
        // expected = exp(0 + (182 / 366) * (ln(0.99) - ln(1.0)) = 0.995015
        assert_eq!(result, DualsOrF64::F64(0.9950147597711371));
    }
}
