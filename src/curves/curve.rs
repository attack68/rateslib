use crate::calendars::calendar_py::Cals;
use crate::curves::interpolation::LocalInterpolation;
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

//     pub fn value(&self, &NaiveDateTime) -> DualsOrF64 {
//         match self.interpolation {
//             LocalInterpolation::LogLinear => {}
//             _ => panic!("not implemented!")
//         }
//     }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::calendar::ndt;

    #[test]
    fn construct_curve() {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        let curve = Curve::try_new(nodes, LocalInterpolation::LogLinear, "crv").unwrap();
    }
}