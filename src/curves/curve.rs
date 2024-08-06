use crate::curves::nodes::{Nodes, NodesTimestamp};
use crate::curves::{CurveInterpolation};
use crate::dual::{ADOrder, DualsOrF64};
use chrono::NaiveDateTime;
use pyo3::{pyclass, PyErr};

/// Default struct for storing discount factors (DFs).
pub struct Curve<T: CurveInterpolation> {
    pub(crate) nodes: NodesTimestamp,
    interpolator: T,
    pub(crate) id: String,
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
    use crate::calendars::{ndt, CalType, Convention, NamedCal};
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
}
