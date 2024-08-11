use crate::curves::interpolation::utils::index_left;
use crate::curves::nodes::{Nodes, NodesTimestamp};
use crate::dual::{ADOrder, DualsOrF64};
use chrono::NaiveDateTime;
use pyo3::PyErr;
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

/// Default struct for storing discount factors (DFs).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Curve<T: CurveInterpolation> {
    pub(crate) nodes: NodesTimestamp,
    interpolator: T,
    pub(crate) id: String,
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

impl<T: CurveInterpolation> Curve<T> {
    pub fn try_new(nodes: Nodes, interpolator: T, id: &str) -> Result<Self, PyErr> {
        let mut nodes = NodesTimestamp::from(nodes);
        nodes.sort_keys();
        Ok(Self {
            nodes,
            interpolator,
            id: id.to_string(),
        })
    }

    /// Get the `ADOrder` of the `Curve`.
    pub fn ad(&self) -> ADOrder {
        match self.nodes {
            NodesTimestamp::F64(_) => ADOrder::Zero,
            NodesTimestamp::Dual(_) => ADOrder::One,
            NodesTimestamp::Dual2(_) => ADOrder::Two,
        }
    }
}

impl<T: CurveInterpolation> Curve<T> {
    pub fn interpolated_value(&self, date: &NaiveDateTime) -> DualsOrF64 {
        self.interpolator.interpolated_value(&self.nodes, date)
    }

    pub fn node_index(&self, date_timestamp: i64) -> usize {
        self.interpolator.node_index(&self.nodes, date_timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;
    use crate::curves::LogLinearInterpolator;
    use indexmap::IndexMap;

    fn curve_fixture() -> Curve<LogLinearInterpolator> {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        let interpolator = LogLinearInterpolator::new();
        Curve::try_new(nodes, interpolator, "crv").unwrap()
    }

    #[test]
    fn test_get_index() {
        let c = curve_fixture();
        let result = c.node_index(ndt(2001, 7, 30).and_utc().timestamp());
        assert_eq!(result, 1_usize)
    }

    #[test]
    fn test_get_value() {
        let c = curve_fixture();
        let result = c.interpolated_value(&ndt(2000, 7, 1));
        assert_eq!(result, DualsOrF64::F64(0.9950147597711371))
    }

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
