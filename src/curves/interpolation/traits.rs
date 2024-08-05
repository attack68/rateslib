use crate::dual::{DualsOrF64};
use chrono::NaiveDateTime;
use crate::curves::{
    LogLinearInterpolator,
    LinearInterpolator,
    LinearZeroRateInterpolator,
};
use crate::curves::nodes::{NodesTimestamp};
use crate::curves::interpolation::utils::index_left;
use pyo3::FromPyObject;
use std::cmp::PartialEq;

/// Interpolation
#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub enum CurveInterpolator {
    LogLinear(LogLinearInterpolator),
    Linear(LinearInterpolator),
    LinearZeroRate(LinearZeroRateInterpolator),
//     LinearIndex,
//     LinearZeroRate,
//     FlatForward,
//     FlatBackward,
}

/// Assigns methods for returning values from datetime indexed Curves.
pub trait CurveInterpolation {
    /// Get a value from the curve's `Nodes` expressed in its input form, i.e. discount factor or value.
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> DualsOrF64;

    /// Get the left side node key index of the given datetime
    fn node_index(&self, nodes: &NodesTimestamp, date_timestamp: i64) -> usize {
        // let timestamp = date.and_utc().timestamp();
        index_left(&nodes.keys(), &date_timestamp, None)
    }
}

impl CurveInterpolation for CurveInterpolator {
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> DualsOrF64 {
        match self {
            CurveInterpolator::LogLinear(i) => i.interpolated_value(nodes, date),
            CurveInterpolator::Linear(i) => i.interpolated_value(nodes, date),
            CurveInterpolator::LinearZeroRate(i) => i.interpolated_value(nodes, date),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use crate::curves::nodes::Nodes;
    use crate::calendars::{ndt};

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
