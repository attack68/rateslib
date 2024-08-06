use crate::dual::{Dual, Dual2};
use chrono::NaiveDateTime;
use indexmap::IndexMap;

/// Datetime indexed values of a specific [ADOrder](`crate::dual::ADOrder`).
pub enum Nodes {
    F64(IndexMap<NaiveDateTime, f64>),
    Dual(IndexMap<NaiveDateTime, Dual>),
    Dual2(IndexMap<NaiveDateTime, Dual2>),
}

pub enum NodesTimestamp {
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
    // fn keys_as_f64(&self) -> Vec<f64> {
    //     match self {
    //         NodesTimestamp::F64(m) => m.keys().cloned().map(|x| x as f64).collect(),
    //         NodesTimestamp::Dual(m) => m.keys().cloned().map(|x| x as f64).collect(),
    //         NodesTimestamp::Dual2(m) => m.keys().cloned().map(|x| x as f64).collect(),
    //     }
    // }

    pub(crate) fn sort_keys(&mut self) {
        match self {
            NodesTimestamp::F64(m) => m.sort_keys(),
            NodesTimestamp::Dual(m) => m.sort_keys(),
            NodesTimestamp::Dual2(m) => m.sort_keys(),
        }
    }

    pub(crate) fn keys(&self) -> Vec<i64> {
        match self {
            NodesTimestamp::F64(m) => m.keys().cloned().collect(),
            NodesTimestamp::Dual(m) => m.keys().cloned().collect(),
            NodesTimestamp::Dual2(m) => m.keys().cloned().collect(),
        }
    }

    //     /// Refactors the `get_index` method of an IndexMap and type casts the return values.
    //     pub(crate) fn get_index_as_f64(&self, index: usize) -> (f64, DualsOrF64) {
    //         match self {
    //             NodesTimestamp::F64(m) => {
    //                 let (k, v) = m.get_index(index).unwrap();
    //                 (*k as f64, DualsOrF64::F64(*v))
    //             },
    //             NodesTimestamp::Dual(m) => {
    //                 let (k, v) = m.get_index(index).unwrap();
    //                 (*k as f64, DualsOrF64::Dual(v.clone()))
    //             },
    //             NodesTimestamp::Dual2(m) => {
    //                 let (k, v) = m.get_index(index).unwrap();
    //                 (*k as f64, DualsOrF64::Dual2(v.clone()))
    //             },
    //         }
    //     }
}