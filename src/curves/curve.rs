use crate::curves::interpolation::{CurveInterpolator};
use crate::dual::{DualsOrF64, ADOrder, Dual, Dual2, set_order, get_variable_tags};
use chrono::NaiveDateTime;
use indexmap::IndexMap;
use pyo3::PyErr;
use std::collections::HashMap;
// use crate::calendars::dcfs::;

use crate::json::JSON;
use serde::{Serialize, Deserialize};

pub(crate) enum Nodes {
    F64(IndexMap<NaiveDateTime, f64>),
    Dual(IndexMap<NaiveDateTime, Dual>),
    Dual2(IndexMap<NaiveDateTime, Dual2>),
}

pub(crate) enum NodesTimestamp {
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

    pub(crate) fn keys(&self) -> Vec<i64> {
        match self {
            NodesTimestamp::F64(m) => m.keys().cloned().collect(),
            NodesTimestamp::Dual(m) => m.keys().cloned().collect(),
            NodesTimestamp::Dual2(m) => m.keys().cloned().collect(),
        }
    }

    /// Refactors the `get_index` method of an IndexMap and type casts the return values.
    pub(crate) fn get_index_as_f64(&self, index: usize) -> (f64, DualsOrF64) {
        match self {
            NodesTimestamp::F64(m) => {
                let (k, v) = m.get_index(index).unwrap();
                (*k as f64, DualsOrF64::F64(*v))
            },
            NodesTimestamp::Dual(m) => {
                let (k, v) = m.get_index(index).unwrap();
                (*k as f64, DualsOrF64::Dual(v.clone()))
            },
            NodesTimestamp::Dual2(m) => {
                let (k, v) = m.get_index(index).unwrap();
                (*k as f64, DualsOrF64::Dual2(v.clone()))
            },
        }
    }
}

pub struct Curve {
    nodes: NodesTimestamp,
    interpolator: CurveInterpolator,
    id: String,
}

impl Curve {
    pub fn try_new(
        nodes: Nodes,
        interpolator: CurveInterpolator,
        id: &str,
    ) -> Result<Self, PyErr> {
        let nodes = NodesTimestamp::from(nodes);
        Ok( Self {nodes, interpolator, id: id.to_string()} )
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;

//     fn curve_fixture() -> Curve {
//         let nodes = Nodes::F64(IndexMap::from_iter(vec![
//             (ndt(2000, 1, 1), 1.0_f64),
//             (ndt(2001, 1, 1), 0.99_f64),
//             (ndt(2002, 1, 1), 0.98_f64),
//         ]));
//         Curve::try_new(nodes, LocalInterpolation::LogLinear, "crv").unwrap()
//     }

//     #[test]
//     fn test_get_index() {
//         let c = curve_fixture();
//         let result = c.get_index(&ndt(2001, 7, 30));
//         assert_eq!(result, 1_usize)
//     }
//
//     #[test]
//     fn test_get_value() {
//         let c = curve_fixture();
//         let result = c.get_value(&ndt(2001, 7, 30));
//         assert_eq!(result, DualsOrF64::F64(0.985_f64))
//     }
}