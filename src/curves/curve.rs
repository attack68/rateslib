use crate::calendars::calendar_py::Cals;
use crate::curves::interpolation::LocalInterpolation;
use crate::dual::dual::{DualsOrF64, ADOrder};
use pyo3::PyErr;
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

pub struct Curve {
    nodes: Nodes,
    calendar: Cals,
    convention: Convention,
    interpolation: LocalInterpolation,
    id: String,
}

impl Curve {
    pub fn try_new(
        nodes: HashMap<NaiveDateTime, DualsOrF64>,
        ad: ADOrder
        id: &str,
    ) -> Result<Self, PyErr> {
        let mut nodes_order: IndexMap<NaiveDateTime, DualsOrF64> = IndexMap::new();
        for (k,v) in nodes {
            let _ = nodes_order.insert_sorted(k, v);
        }
    }

    pub fn value(&self, &NaiveDateTime) -> DualsOrF64 {
        match self.interpolation {
            LocalInterpolation::LogLinear => {}
            _ => panic!("not implemented!")
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
}