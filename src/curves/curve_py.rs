//! Wrapper module to export Rust curve data types to Python using pyo3 bindings.

use crate::curves::{Curve, CurveInterpolator};
use crate::curves::nodes::Nodes;
use crate::dual::{Dual, Dual2, DualsOrF64, set_order, ADOrder, get_variable_tags};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFloat};
use bincode::{deserialize, serialize};
use indexmap::IndexMap;
use chrono::NaiveDateTime;
use crate::json::json_py::DeserializedObj;
use crate::json::JSON;

#[pymethods]
impl Curve {
    #[new]
    fn new_py(
        mut nodes: IndexMap<NaiveDateTime, DualsOrF64>,
        interpolator: CurveInterpolator,
        ad: ADOrder,
        id: &str,
    ) -> PyResult<Self> {
        let nodes_ = nodes_into_order(nodes, ad, id);
        Self::try_new(nodes_, interpolator, id)
    }

    fn __get_item__(&self, date: NaiveDateTime) -> DualsOrF64 {
        self.interpolated_value(&date)
    }
}

// /// Convert the `nodes`of a `Curve` from a `HashMap` input form into the local data model.
// /// Will upcast f64 values to a new ADOrder adding curve variable tags by id.
// fn hashmap_into_nodes_timestamp(
//     h: HashMap<NaiveDateTime, DualsOrF64>,
//     ad: ADOrder,
//     id: &str,
// ) -> NodesTimestamp {
//     let vars: Vec<String> = get_variable_tags(id, h.keys().len());
//
//     /// First convert to IndexMap and sort key order.
//     // let mut im: IndexMap<NaiveDateTime, DualsOrF64> = IndexMap::from_iter(h.into_iter());
//     let mut im: IndexMap<i64, DualsOrF64> = IndexMap::from_iter(h.into_iter().map(|(k,v)| (k.and_utc().timestamp(), v)));
//     im.sort_keys();
//
//     match ad {
//         ADOrder::Zero => { NodesTimestamp::F64(IndexMap::from_iter(im.into_iter().map(|(k,v)| (k, f64::from(v))))) }
//         ADOrder::One => { NodesTimestamp::Dual(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
//         ADOrder::Two => { NodesTimestamp::Dual2(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual2::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
//     }
// }

fn nodes_into_order(
    mut nodes: IndexMap<NaiveDateTime, DualsOrF64>,
    ad: ADOrder,
    id: &str,
) -> Nodes {
    let vars: Vec<String> = get_variable_tags(id, nodes.keys().len());
    nodes.sort_keys();
    match ad {
        ADOrder::Zero => { Nodes::F64(IndexMap::from_iter(nodes.into_iter().map(|(k,v)| (k, f64::from(v))))) }
        ADOrder::One => { Nodes::Dual(IndexMap::from_iter(nodes.into_iter().enumerate().map(|(i,(k,v))| (k, Dual::from(set_order(v, ad, vec![vars[i].clone()])))))) }
        ADOrder::Two => { Nodes::Dual2(IndexMap::from_iter(nodes.into_iter().enumerate().map(|(i,(k,v))| (k, Dual2::from(set_order(v, ad, vec![vars[i].clone()])))))) }
    }
}
