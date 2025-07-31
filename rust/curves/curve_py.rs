//! Wrapper module to export Rust curve data types to Python using pyo3 bindings.

use crate::curves::nodes::{Nodes, NodesTimestamp};
use crate::curves::{
    CurveDF, CurveInterpolation, FlatBackwardInterpolator, FlatForwardInterpolator,
    LinearInterpolator, LinearZeroRateInterpolator, LogLinearInterpolator, Modifier,
    NullInterpolator,
};
use crate::dual::{get_variable_tags, set_order, ADOrder, Dual, Dual2, Number};
use crate::json::json_py::DeserializedObj;
use crate::json::JSON;
use crate::scheduling::{Calendar, Convention};
use bincode::config::legacy;
use bincode::serde::{decode_from_slice, encode_to_vec};
use chrono::NaiveDateTime;
use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

/// Interpolation
#[derive(Debug, Clone, PartialEq, FromPyObject, Deserialize, Serialize, IntoPyObject)]
pub(crate) enum CurveInterpolator {
    LogLinear(LogLinearInterpolator),
    Linear(LinearInterpolator),
    LinearZeroRate(LinearZeroRateInterpolator),
    FlatForward(FlatForwardInterpolator),
    FlatBackward(FlatBackwardInterpolator),
    Null(NullInterpolator),
}

// // removed upgrading to pyo3 0.23, see https://pyo3.rs/v0.23.0/migration#intopyobject-and-intopyobjectref-derive-macros
// impl IntoPy<PyObject> for CurveInterpolator {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         macro_rules! into_py {
//             ($obj: ident) => {
//                 Py::new(py, $obj).unwrap().to_object(py)
//             };
//         }
//
//         match self {
//             CurveInterpolator::LogLinear(i) => into_py!(i),
//             CurveInterpolator::Linear(i) => into_py!(i),
//             CurveInterpolator::LinearZeroRate(i) => into_py!(i),
//             CurveInterpolator::FlatForward(i) => into_py!(i),
//             CurveInterpolator::FlatBackward(i) => into_py!(i),
//             CurveInterpolator::Null(i) => into_py!(i),
//         }
//     }
// }

impl CurveInterpolation for CurveInterpolator {
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> Number {
        match self {
            CurveInterpolator::LogLinear(i) => i.interpolated_value(nodes, date),
            CurveInterpolator::Linear(i) => i.interpolated_value(nodes, date),
            CurveInterpolator::LinearZeroRate(i) => i.interpolated_value(nodes, date),
            CurveInterpolator::FlatBackward(i) => i.interpolated_value(nodes, date),
            CurveInterpolator::FlatForward(i) => i.interpolated_value(nodes, date),
            CurveInterpolator::Null(i) => i.interpolated_value(nodes, date),
        }
    }
}

#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct Curve {
    inner: CurveDF<CurveInterpolator, Calendar>,
}

#[pymethods]
impl Curve {
    #[new]
    #[pyo3(signature = (nodes, interpolator, ad, id, convention, modifier, calendar, index_base=None))]
    fn new_py(
        nodes: IndexMap<NaiveDateTime, Number>,
        interpolator: CurveInterpolator,
        ad: ADOrder,
        id: String,
        convention: Convention,
        modifier: Modifier,
        calendar: Calendar,
        index_base: Option<f64>,
    ) -> PyResult<Self> {
        let nodes_ = nodes_into_order(nodes, ad, &id);
        let inner = CurveDF::try_new(
            nodes_,
            interpolator,
            &id,
            convention,
            modifier,
            index_base,
            calendar,
        )?;
        Ok(Self { inner })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn nodes(&self) -> IndexMap<NaiveDateTime, Number> {
        let nodes = Nodes::from(self.inner.nodes.clone());
        match nodes {
            Nodes::F64(i) => IndexMap::from_iter(i.into_iter().map(|(k, v)| (k, Number::F64(v)))),
            Nodes::Dual(i) => IndexMap::from_iter(i.into_iter().map(|(k, v)| (k, Number::Dual(v)))),
            Nodes::Dual2(i) => {
                IndexMap::from_iter(i.into_iter().map(|(k, v)| (k, Number::Dual2(v))))
            }
        }
    }

    #[getter]
    fn ad(&self) -> ADOrder {
        self.inner.ad()
    }

    #[getter]
    fn interpolation(&self) -> String {
        match self.inner.interpolator {
            CurveInterpolator::Linear(_) => "linear".to_string(),
            CurveInterpolator::LogLinear(_) => "log_linear".to_string(),
            CurveInterpolator::LinearZeroRate(_) => "linear_zero_rate".to_string(),
            CurveInterpolator::FlatForward(_) => "flat_forward".to_string(),
            CurveInterpolator::FlatBackward(_) => "flat_backward".to_string(),
            CurveInterpolator::Null(_) => "null".to_string(),
        }
    }

    #[getter]
    fn convention(&self) -> Convention {
        self.inner.convention
    }

    #[getter]
    fn modifier(&self) -> Modifier {
        self.inner.modifier
    }

    #[pyo3(name = "index_value")]
    fn index_value_py(&self, date: NaiveDateTime) -> PyResult<Number> {
        self.inner.index_value(&date)
    }

    fn set_ad_order(&mut self, ad: ADOrder) -> PyResult<()> {
        let _ = self.inner.set_ad_order(ad);
        Ok(())
    }

    fn __getitem__(&self, date: NaiveDateTime) -> Number {
        self.inner.interpolated_value(&date)
    }

    fn __eq__(&self, other: Curve) -> bool {
        self.inner.eq(&other.inner)
    }

    // JSON
    /// Create a JSON string representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::Curve(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `Curve` to JSON.",
            )),
        }
    }

    // Pickling
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = decode_from_slice(state.as_bytes(), legacy()).unwrap().0;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &encode_to_vec(&self, legacy()).unwrap()))
    }
    pub fn __getnewargs__(
        &self,
    ) -> PyResult<(
        IndexMap<NaiveDateTime, Number>,
        CurveInterpolator,
        ADOrder,
        String,
        Convention,
        Modifier,
        Calendar,
        Option<f64>,
    )> {
        Ok((
            self.inner.nodes.index_map(),
            self.inner.interpolator.clone(),
            self.inner.ad(),
            self.inner.id.clone(),
            self.inner.convention,
            self.inner.modifier,
            self.inner.calendar.clone(),
            self.inner.index_base,
        ))
    }
}

// /// Convert the `nodes`of a `Curve` from a `HashMap` input form into the local data model.
// /// Will upcast f64 values to a new ADOrder adding curve variable tags by id.
// fn hashmap_into_nodes_timestamp(
//     h: HashMap<NaiveDateTime, Number>,
//     ad: ADOrder,
//     id: &str,
// ) -> NodesTimestamp {
//     let vars: Vec<String> = get_variable_tags(id, h.keys().len());
//
//     /// First convert to IndexMap and sort key order.
//     // let mut im: IndexMap<NaiveDateTime, Number> = IndexMap::from_iter(h.into_iter());
//     let mut im: IndexMap<i64, Number> = IndexMap::from_iter(h.into_iter().map(|(k,v)| (k.and_utc().timestamp(), v)));
//     im.sort_keys();
//
//     match ad {
//         ADOrder::Zero => { NodesTimestamp::F64(IndexMap::from_iter(im.into_iter().map(|(k,v)| (k, f64::from(v))))) }
//         ADOrder::One => { NodesTimestamp::Dual(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
//         ADOrder::Two => { NodesTimestamp::Dual2(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual2::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
//     }
// }

fn nodes_into_order(mut nodes: IndexMap<NaiveDateTime, Number>, ad: ADOrder, id: &str) -> Nodes {
    let vars: Vec<String> = get_variable_tags(id, nodes.keys().len());
    nodes.sort_keys();
    match ad {
        ADOrder::Zero => Nodes::F64(IndexMap::from_iter(
            nodes.into_iter().map(|(k, v)| (k, f64::from(v))),
        )),
        ADOrder::One => {
            Nodes::Dual(IndexMap::from_iter(nodes.into_iter().enumerate().map(
                |(i, (k, v))| (k, Dual::from(set_order(v, ad, vec![vars[i].clone()]))),
            )))
        }
        ADOrder::Two => {
            Nodes::Dual2(IndexMap::from_iter(nodes.into_iter().enumerate().map(
                |(i, (k, v))| (k, Dual2::from(set_order(v, ad, vec![vars[i].clone()]))),
            )))
        }
    }
}

#[pymethods]
impl Modifier {
    // Pickling
    #[new]
    fn new_py(ad: u8) -> PyResult<Modifier> {
        match ad {
            0_u8 => Ok(Modifier::Act),
            1_u8 => Ok(Modifier::F),
            2_u8 => Ok(Modifier::ModF),
            3_u8 => Ok(Modifier::P),
            4_u8 => Ok(Modifier::ModP),
            _ => Err(PyValueError::new_err(
                "unreachable code on Convention pickle.",
            )),
        }
    }
    pub fn __getnewargs__<'py>(&self) -> PyResult<(u8,)> {
        match self {
            Modifier::Act => Ok((0_u8,)),
            Modifier::F => Ok((1_u8,)),
            Modifier::ModF => Ok((2_u8,)),
            Modifier::P => Ok((3_u8,)),
            Modifier::ModP => Ok((4_u8,)),
        }
    }
}

#[pyfunction]
pub(crate) fn _get_modifier_str(modifier: Modifier) -> String {
    match modifier {
        Modifier::F => "F".to_string(),
        Modifier::ModF => "MF".to_string(),
        Modifier::P => "P".to_string(),
        Modifier::ModP => "MP".to_string(),
        Modifier::Act => "NONE".to_string(),
    }
}
