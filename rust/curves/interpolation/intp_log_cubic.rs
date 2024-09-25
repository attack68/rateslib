use crate::curves::nodes::NodesTimestamp;
use crate::curves::CurveInterpolation;
use crate::dual::{Number, ADOrder, NumberPPSpline, set_order_clone, Dual, Dual2};
use crate::splines::{PPSplineF64, PPSplineDual, PPSplineDual2};
use chrono::NaiveDateTime;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

/// Define log-cubic interpolation of nodes.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct LogCubicInterpolator {
    spline: NumberPPSpline
}

#[pymethods]
impl LogCubicInterpolator {
    #[new]
    pub fn new(t: Vec<f64>, ad: ADOrder, c: Option<Vec<Number>>) -> Self {
        match (c, ad) {
            (Some(v), ADOrder::Zero) => {
                let c_: Vec<f64> = v.iter().map(|x| set_order_clone(&x, ADOrder::Zero, vec![])).map(f64::from).collect();
                let spline = NumberPPSpline::F64(PPSplineF64::new(3, t, Some(c_)));
                LogCubicInterpolator {spline}
            }
            (Some(v), ADOrder::One) => {
                let c_: Vec<Dual> = v.iter().map(|x| set_order_clone(&x, ADOrder::One, vec![])).map(Dual::from).collect();
                let spline = NumberPPSpline::Dual(PPSplineDual::new(3, t, Some(c_)));
                LogCubicInterpolator {spline}
            }
            (Some(v), ADOrder::Two) => {
                let c_: Vec<Dual2> = v.iter().map(|x| set_order_clone(&x, ADOrder::Zero, vec![])).map(Dual2::from).collect();
                let spline = NumberPPSpline::Dual2(PPSplineDual2::new(3, t, Some(c_)));
                LogCubicInterpolator {spline}
            }
            (None, ADOrder::Zero) => {LogCubicInterpolator {spline: NumberPPSpline::F64(PPSplineF64::new(3, t, None))}},
            (None, ADOrder::One) => {LogCubicInterpolator {spline: NumberPPSpline::Dual(PPSplineDual::new(3, t, None))}},
            (None, ADOrder::Two) => {LogCubicInterpolator {spline: NumberPPSpline::Dual2(PPSplineDual2::new(3, t, None))}},
        }
    }
    //
    // // Pickling
    // pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
    //     *self = deserialize(state.as_bytes()).unwrap();
    //     Ok(())
    // }
    // pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    //     Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    // }
    // pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<(Vec<f64>, Option<Vec<T>>)> {
    //     Ok((self.t.clone(), ))
    // }
}

impl CurveInterpolation for LogCubicInterpolator {
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> Number {
        Number::F64(2.3)
    }

    /// Calibrate the interpolator to the Curve nodes if necessary
    fn calibrate(&self, nodes: &NodesTimestamp) -> Result<(), PyErr> {
        // will call csolve on the spline with the appropriate data.
        let t = self.spline.t().clone();
        let t_min = t[0]; let t_max = t[t.len()-1];

        let f = nodes.clone().iter().filter(|(k,v)| (k as f64) >= t_min && (k as f64) <= t_max);
        Ok(())
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
    fn test_log_cubic() {
        let nts = nodes_timestamp_fixture();
        let s = nts.get_index_as_f64(0).0;
        let m = nts.get_index_as_f64(1).0;
        let e = nts.get_index_as_f64(2).0;
        let t = vec![s, s, s, s, m, e, e, e, e];
        let ll = LogCubicInterpolator::new(t, ADOrder::Zero, None);
    }
}
