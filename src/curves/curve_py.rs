//! Wrapper module to export Rust curve data types to Python using pyo3 bindings.

use crate::curves::Curve;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFloat};
use bincode::{deserialize, serialize};
use crate::json::json_py::DeserializedObj;
use crate::json::JSON;

impl IntoPy<PyObject> for DualsOrF64 {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            DualsOrF64::F64(f) => PyFloat::new_bound(py, f).to_object(py),
            DualsOrF64::Dual(d) => Py::new(py, d).unwrap().to_object(py),
            DualsOrF64::Dual2(d) => Py::new(py, d).unwrap().to_object(py),
        }
    }
}

#[pymethods]
impl Curve {
    #[new]
    fn new_py(real: f64, vars: Vec<String>, dual: Vec<f64>) -> PyResult<Self> {
        Dual::try_new(real, vars, dual)
    }

    /// Create a :class:`~rateslib.dual.Dual` object with ``vars`` linked with another.
    ///
    /// Parameters
    /// ----------
    /// other: Dual
    ///     The other `Dual` from which `vars` are linked.
    /// real: float
    ///     The real coefficient of the dual number.
    /// vars: list[str]
    ///     The labels of the variables for which to record derivatives. If empty,
    ///     the dual number represents a constant, equivalent to a float.
    /// dual: list[float]
    ///     First derivative information contained as coefficient of linear manifold.
    ///     Defaults to an array of ones the length of ``vars`` if empty.
    ///
    /// Returns
    /// -------
    /// Dual
    ///
    /// Notes
    /// -----
    /// Variables are constantly checked when operations are performed between dual numbers. In Rust the variables
    /// are stored within an ARC pointer. It is much faster to check the equivalence of two ARC pointers than if the elements
    /// within a variables Set, say, are the same *and* in the same order. This method exists to create dual data types
    /// with shared ARC pointers directly.
    ///
    /// .. ipython:: python
    ///
    ///    from rateslib import Dual
    ///
    ///    x1 = Dual(1.0, ["x"], [])
    ///    x2 = Dual(2.0, ["x"], [])
    ///    # x1 and x2 have the same variables (["x"]) but it is a different object
    ///    x1.ptr_eq(x2)
    ///
    ///    x3 = Dual.vars_from(x1, 3.0, ["x"], [])
    ///    # x3 contains shared object variables with x1
    ///    x1.ptr_eq(x3)
    #[staticmethod]
    fn vars_from(other: &Dual, real: f64, vars: Vec<String>, dual: Vec<f64>) -> PyResult<Self> {
        Dual::try_new_from(other, real, vars, dual)
    }

    #[getter]
    #[pyo3(name = "real")]
    fn real_py(&self) -> PyResult<f64> {
        Ok(self.real())
    }

/// Convert the `nodes`of a `Curve` from a `HashMap` input form into the local data model.
/// Will upcast f64 values to a new ADOrder adding curve variable tags by id.
fn hashmap_into_nodes_timestamp(
    h: HashMap<NaiveDateTime, DualsOrF64>,
    ad: ADOrder,
    id: &str,
) -> NodesTimestamp {
    let vars: Vec<String> = get_variable_tags(id, h.keys().len());

    /// First convert to IndexMap and sort key order.
    let mut im: IndexMap<i64, DualsOrF64> = IndexMap::from_iter(h.into_iter().map(|(k,v)| (k.and_utc().timestamp(), v)));
    im.sort_keys();

    match ad {
        ADOrder::Zero => { NodesTimestamp::F64(IndexMap::from_iter(im.into_iter().map(|(k,v)| (k, f64::from(v))))) }
        ADOrder::One => { NodesTimestamp::Dual(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
        ADOrder::Two => { NodesTimestamp::Dual2(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual2::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
    }
}