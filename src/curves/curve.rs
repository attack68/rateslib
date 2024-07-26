use crate::calendars::calendar_py::Cals;
use crate::curves::interpolation::{LocalInterpolation, index_left};
use crate::dual::dual::{DualsOrF64, ADOrder, Dual, Dual2, set_order_with_conversion};
use crate::dual::get_variable_tags;
use chrono::NaiveDateTime;
use indexmap::IndexMap;
use pyo3::PyErr;
use std::collections::HashMap;
// use crate::calendars::dcfs::;

use crate::json::JSON;
use serde::{Serialize, Deserialize};

enum Nodes {
    F64(IndexMap<NaiveDateTime, f64>),
    Dual(IndexMap<NaiveDateTime, Dual>),
    Dual2(IndexMap<NaiveDateTime, Dual2>),
}

enum NodesTimestamp {
    F64(IndexMap<i64, f64>),
    Dual(IndexMap<i64, Dual>),
    Dual2(IndexMap<i64, Dual2>),
}

impl From<Nodes> for NodesTimestamp {
    fn from(value: Nodes) -> Self {
        match value {
            Nodes::F64(m) => {NodesTimestamp::F64(IndexMap::from_iter(m.into_iter().map(|(k,v)| (k.and_utc().timestamp(), v))))}
            Nodes::Dual(m) => {NodesTimestamp::Dual(IndexMap::from_iter(m.into_iter().map(|(k,v)| (k.and_utc().timestamp(), v))))}
            Nodes::Dual2(m) => {NodesTimestamp::Dual2(IndexMap::from_iter(m.into_iter().map(|(k,v)| (k.and_utc().timestamp(), v))))}
        }
    }
}

impl NodesTimestamp {
    fn keys_as_f64(&self) -> Vec<f64> {
        match self {
            NodesTimestamp::F64(m) => m.keys().cloned().map(|x| x as f64).collect(),
            NodesTimestamp::Dual(m) => m.keys().cloned().map(|x| x as f64).collect(),
            NodesTimestamp::Dual2(m) => m.keys().cloned().map(|x| x as f64).collect(),
        }
    }

    fn keys(&self) -> Vec<i64> {
        match self {
            NodesTimestamp::F64(m) => m.keys().cloned().collect(),
            NodesTimestamp::Dual(m) => m.keys().cloned().collect(),
            NodesTimestamp::Dual2(m) => m.keys().cloned().collect(),
        }
    }
}

pub struct Curve {
    nodes: NodesTimestamp,
//     calendar: Cals,
//     convention: Convention,
    interpolation: LocalInterpolation,
    id: String,
}

impl Curve {
    pub fn try_new(
        nodes: Nodes,
        interpolation: LocalInterpolation,
        id: &str,
    ) -> Result<Self, PyErr> {
        let nodes = NodesTimestamp::from(nodes);
        Ok( Self {nodes, interpolation, id: id.to_string()} )
    }

    /// Get the `ADOrder` of the `Curve`.
    pub fn ad(&self) -> ADOrder {
        match self.nodes {
            NodesTimestamp::F64(_) => ADOrder::Zero,
            NodesTimestamp::Dual(_) => ADOrder::One,
            NodesTimestamp::Dual2(_) => ADOrder::Two,
        }
    }

    /// Get the left side node key index of the given x value datetime
    fn get_index(&self, date: &NaiveDateTime) -> usize {
        let timestamp = date.and_utc().timestamp();
        index_left(&self.nodes.keys(), &timestamp, None)
    }

    pub fn get_value(&self, date: &NaiveDateTime) -> DualsOrF64 {
        let index = self.get_index(date);
        match &self.nodes {
            NodesTimestamp::F64(m) => {
                let (x1, y1) = m.get_index(index).unwrap();
                let (x2, y2) = m.get_index(index + 1_usize).unwrap();
                DualsOrF64::F64((y1 + y2) / 2.0_f64)
            }
            NodesTimestamp::Dual(m) => {
                let (x1, y1) = m.get_index(index).unwrap();
                let (x2, y2) = m.get_index(index + 1_usize).unwrap();
                DualsOrF64::F64(10.0)
            }
            NodesTimestamp::Dual2(m) => {
               let (x1, y1) = m.get_index(index).unwrap();
                let (x2, y2) = m.get_index(index + 1_usize).unwrap();
                DualsOrF64::F64(10.0)
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::calendar::ndt;

    fn curve_fixture() -> Curve {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        Curve::try_new(nodes, LocalInterpolation::LogLinear, "crv").unwrap()
    }

    #[test]
    fn test_get_index() {
        let c = curve_fixture();
        let result = c.get_index(&ndt(2001, 7, 30));
        assert_eq!(result, 1_usize)
    }

    #[test]
    fn test_get_value() {
        let c = curve_fixture();
        let result = c.get_value(&ndt(2001, 7, 30));
        assert_eq!(result, DualsOrF64::F64(0.985_f64))
    }
}