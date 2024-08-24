//! Wrapper module to export Rust dual data types to Python using pyo3 bindings.

use crate::dual::dual::{ADOrder, Dual, Dual2, Gradient1, Gradient2, Number, Vars};
use crate::dual::dual_ops::math_funcs::MathFuncs;
use bincode::{deserialize, serialize};
use num_traits::{Pow, Signed};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFloat};
use std::sync::Arc;
// use pyo3::types::PyFloat;
use crate::json::json_py::DeserializedObj;
use crate::json::JSON;
use numpy::{Element, PyArray1, PyArray2, PyArrayDescr, ToPyArray};

unsafe impl Element for Dual {
    const IS_COPY: bool = false;
    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::object_bound(py)
    }
}
unsafe impl Element for Dual2 {
    const IS_COPY: bool = false;
    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::object_bound(py)
    }
}

impl IntoPy<PyObject> for Number {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Number::F64(f) => PyFloat::new_bound(py, f).to_object(py),
            Number::Dual(d) => Py::new(py, d).unwrap().to_object(py),
            Number::Dual2(d) => Py::new(py, d).unwrap().to_object(py),
        }
    }
}

// https://github.com/PyO3/pyo3/discussions/3911
// #[derive(Debug, Clone, PartialEq, PartialOrd, FromPyObject)]
// pub enum DualsOrPyFloat<'py> {
//     Dual(Dual),
//     Dual2(Dual2),
//     Float(&'py PyFloat),
// }

#[pymethods]
impl ADOrder {
    // Pickling
    #[new]
    fn new_py(ad: u8) -> PyResult<ADOrder> {
        match ad {
            0_u8 => Ok(ADOrder::Zero),
            1_u8 => Ok(ADOrder::One),
            2_u8 => Ok(ADOrder::Two),
            _ => Err(PyValueError::new_err("unreachable code on ADOrder pickle.")),
        }
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__<'py>(&self) -> PyResult<(u8,)> {
        match self {
            ADOrder::Zero => Ok((0_u8,)),
            ADOrder::One => Ok((1_u8,)),
            ADOrder::Two => Ok((2_u8,)),
        }
    }
}

#[pymethods]
impl Dual {
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

    #[getter]
    #[pyo3(name = "vars")]
    fn vars_py(&self) -> PyResult<Vec<&String>> {
        Ok(Vec::from_iter(self.vars().iter()))
    }

    #[getter]
    #[pyo3(name = "dual")]
    fn dual_py<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'_, PyArray1<f64>>> {
        Ok(self.dual().to_pyarray_bound(py))
    }

    #[getter]
    #[pyo3(name = "dual2")]
    fn dual2_py<'py>(&'py self, _py: Python<'py>) -> PyResult<&PyArray2<f64>> {
        Err(PyValueError::new_err(
            "`Dual` variable cannot possess `dual2` attribute.",
        ))
    }

    #[pyo3(name = "grad1")]
    fn grad1<'py>(
        &'py self,
        py: Python<'py>,
        vars: Vec<String>,
    ) -> PyResult<Bound<'_, PyArray1<f64>>> {
        Ok(self.gradient1(vars).to_pyarray_bound(py))
    }

    #[pyo3(name = "grad2")]
    fn grad2<'py>(&'py self, _py: Python<'py>, _vars: Vec<String>) -> PyResult<&PyArray2<f64>> {
        Err(PyValueError::new_err(
            "Cannot evaluate second order derivative on a Dual.",
        ))
    }

    /// Evaluate if the ARC pointers of two `Dual` data types are equivalent.
    ///
    /// Parameters
    /// ----------
    /// other: Dual
    ///     The comparison object.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(name = "ptr_eq")]
    fn ptr_eq_py(&self, other: &Dual) -> PyResult<bool> {
        Ok(Arc::ptr_eq(self.vars(), other.vars()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut _vars = Vec::from_iter(self.vars().iter().take(3).map(String::as_str)).join(", ");
        let mut _dual =
            Vec::from_iter(self.dual().iter().take(3).map(|x| format!("{:.1}", x))).join(", ");
        if self.vars().len() > 3 {
            _vars.push_str(", ...");
            _dual.push_str(", ...");
        }
        let fs = format!("<Dual: {:.6}, ({}), [{}]>", self.real(), _vars, _dual);
        Ok(fs)
    }

    fn __eq__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual(d) => Ok(d.eq(self)),
            Number::F64(f) => Ok(Dual::new(f, Vec::new()).eq(self)),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Cannot compare Dual with incompatible type (Dual2).",
            )),
        }
    }

    fn __lt__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual(d) => Ok(self < &d),
            Number::F64(f) => Ok(self < &f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Cannot compare Dual with incompatible type (Dual2).",
            )),
        }
    }

    fn __le__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual(d) => Ok(self <= &d),
            Number::F64(f) => Ok(self <= &f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Cannot compare Dual with incompatible type (Dual2).",
            )),
        }
    }

    fn __gt__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual(d) => Ok(self > &d),
            Number::F64(f) => Ok(self > &f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Cannot compare Dual with incompatible type (Dual2).",
            )),
        }
    }

    fn __ge__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual(d) => Ok(self >= &d),
            Number::F64(f) => Ok(self >= &f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Cannot compare Dual with incompatible type (Dual2).",
            )),
        }
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __add__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(self + d),
            Number::F64(f) => Ok(self + f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __radd__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(self + d),
            Number::F64(f) => Ok(self + f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __sub__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(self - d),
            Number::F64(f) => Ok(self - f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __rsub__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(d - self),
            Number::F64(f) => Ok(f - self),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __mul__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(self * d),
            Number::F64(f) => Ok(self * f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __rmul__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(d * self),
            Number::F64(f) => Ok(f * self),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __truediv__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(self / d),
            Number::F64(f) => Ok(self / f),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __rtruediv__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual(d) => Ok(d / self),
            Number::F64(f) => Ok(f / self),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Dual operation with incompatible type (Dual2).",
            )),
        }
    }

    fn __pow__(&self, power: Number, modulo: Option<i32>) -> PyResult<Self> {
        if modulo.unwrap_or(0) != 0 {
            panic!("Power function with mod not available for Dual.")
        }
        match power {
            Number::F64(f) => Ok(self.clone().pow(f)),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Power operation not defined with Dual type exponent.",
            )),
            Number::Dual2(_) => Err(PyTypeError::new_err(
                "Power operation not defined with Dual type exponent.",
            )),
        }
    }

    fn __exp__(&self) -> Self {
        self.exp()
    }

    fn __abs__(&self) -> f64 {
        self.abs().real()
    }

    fn __log__(&self) -> Self {
        self.log()
    }

    fn __norm_cdf__(&self) -> Self {
        self.norm_cdf()
    }

    fn __norm_inv_cdf__(&self) -> Self {
        self.inv_norm_cdf()
    }

    fn __float__(&self) -> f64 {
        self.real()
    }

    // JSON
    /// Create a JSON string representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::Dual(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err("Failed to serialize `Dual` to JSON.")),
        }
    }

    // Pickling
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(f64, Vec<String>, Vec<f64>)> {
        Ok((
            self.real,
            self.vars().iter().cloned().collect(),
            self.dual.to_vec(),
        ))
    }

    // Conversion
    #[pyo3(name = "to_dual2")]
    fn to_dual2_py(&self) -> Dual2 {
        self.clone().into()
    }
}

#[pymethods]
impl Dual2 {
    /// Python wrapper to construct a new `Dual2`.
    #[new]
    pub fn new_py(real: f64, vars: Vec<String>, dual: Vec<f64>, dual2: Vec<f64>) -> PyResult<Self> {
        Dual2::try_new(real, vars, dual, dual2)
    }

    /// Create a :class:`~rateslib.dual.Dual2` object with ``vars`` linked with another.
    ///
    /// Parameters
    /// ----------
    /// other: Dual
    ///     The other `Dual` from which `vars` are linked.
    /// real: float
    ///     The real coefficient of the dual number.
    /// vars: list(str)
    ///     The labels of the variables for which to record derivatives. If empty,
    ///     the dual number represents a constant, equivalent to a float.
    /// dual: list(float)
    ///     First derivative information contained as coefficient of linear manifold.
    ///     Defaults to an array of ones the length of ``vars`` if empty.
    /// dual2: list(float)
    ///     Second derivative information contained as coefficients of a quadratic manifold.
    ///     These values represent a 2d array but must be given as a 1d list of values in
    ///     row-major order.
    ///     Defaults to a 2-d array of zeros of size NxN where N is length of ``vars`` if not
    ///     given.
    ///
    /// Returns
    /// -------
    /// Dual2
    ///
    /// Notes
    /// --------
    /// For examples see also...
    ///
    /// .. seealso::
    ///    :meth:`~rateslib.dual.Dual.vars_from`: Create a *Dual* with ``vars`` linked to another.
    ///
    #[staticmethod]
    pub fn vars_from(
        other: &Dual2,
        real: f64,
        vars: Vec<String>,
        dual: Vec<f64>,
        dual2: Vec<f64>,
    ) -> PyResult<Self> {
        Dual2::try_new_from(other, real, vars, dual, dual2)
    }

    #[getter]
    #[pyo3(name = "real")]
    fn real_py(&self) -> PyResult<f64> {
        Ok(self.real)
    }

    #[getter]
    #[pyo3(name = "vars")]
    fn vars_py(&self) -> PyResult<Vec<&String>> {
        Ok(Vec::from_iter(self.vars.iter()))
    }

    #[getter]
    #[pyo3(name = "dual")]
    fn dual_py<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'_, PyArray1<f64>>> {
        Ok(self.dual.to_pyarray_bound(py))
    }

    #[getter]
    #[pyo3(name = "dual2")]
    fn dual2_py<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'_, PyArray2<f64>>> {
        Ok(self.dual2.to_pyarray_bound(py))
    }

    #[pyo3(name = "grad1")]
    fn grad1_py<'py>(
        &'py self,
        py: Python<'py>,
        vars: Vec<String>,
    ) -> PyResult<Bound<'_, PyArray1<f64>>> {
        Ok(self.gradient1(vars).to_pyarray_bound(py))
    }

    #[pyo3(name = "grad2")]
    fn grad2_py<'py>(
        &'py self,
        py: Python<'py>,
        vars: Vec<String>,
    ) -> PyResult<Bound<'_, PyArray2<f64>>> {
        Ok(self.gradient2(vars).to_pyarray_bound(py))
    }

    #[pyo3(name = "grad1_manifold")]
    fn grad1_manifold_py<'py>(
        &'py self,
        _py: Python<'py>,
        vars: Vec<String>,
    ) -> PyResult<Vec<Dual2>> {
        let out = self.gradient1_manifold(vars);
        Ok(out.into_raw_vec())
    }

    /// Evaluate if the ARC pointers of two `Dual2` data types are equivalent. See
    /// :meth:`~rateslib.dual.Dual.ptr_eq`.
    #[pyo3(name = "ptr_eq")]
    fn ptr_eq_py(&self, other: &Dual2) -> PyResult<bool> {
        Ok(self.ptr_eq(other))
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut _vars = Vec::from_iter(self.vars.iter().take(3).map(String::as_str)).join(", ");
        let mut _dual =
            Vec::from_iter(self.dual.iter().take(3).map(|x| format!("{:.1}", x))).join(", ");
        if self.vars.len() > 3 {
            _vars.push_str(", ...");
            _dual.push_str(", ...");
        }
        let fs = format!(
            "<Dual2: {:.6}, ({}), [{}], [[...]]>",
            self.real, _vars, _dual
        );
        Ok(fs)
    }

    fn __eq__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual2(d) => Ok(d.eq(self)),
            Number::F64(f) => Ok(Dual2::new(f, Vec::new()).eq(self)),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Cannot compare Dual2 with incompatible type (Dual).",
            )),
        }
    }

    fn __lt__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual2(d) => Ok(self < &d),
            Number::F64(f) => Ok(self < &f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Cannot compare Dual2 with incompatible type (Dual).",
            )),
        }
    }

    fn __le__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual2(d) => Ok(self <= &d),
            Number::F64(f) => Ok(self <= &f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Cannot compare Dual2 with incompatible type (Dual).",
            )),
        }
    }

    fn __gt__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual2(d) => Ok(self > &d),
            Number::F64(f) => Ok(self > &f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Cannot compare Dual2 with incompatible type (Dual).",
            )),
        }
    }

    fn __ge__(&self, other: Number) -> PyResult<bool> {
        match other {
            Number::Dual2(d) => Ok(self >= &d),
            Number::F64(f) => Ok(self >= &f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Cannot compare Dual2 with incompatible type (Dual).",
            )),
        }
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __add__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(self + d),
            Number::F64(f) => Ok(self + f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __radd__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(self + d),
            Number::F64(f) => Ok(self + f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __sub__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(self - d),
            Number::F64(f) => Ok(self - f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __rsub__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(d - self),
            Number::F64(f) => Ok(f - self),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __mul__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(self * d),
            Number::F64(f) => Ok(self * f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __rmul__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(d * self),
            Number::F64(f) => Ok(f * self),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __truediv__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(self / d),
            Number::F64(f) => Ok(self / f),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __rtruediv__(&self, other: Number) -> PyResult<Self> {
        match other {
            Number::Dual2(d) => Ok(d / self),
            Number::F64(f) => Ok(f / self),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Dual2 operation with incompatible type (Dual).",
            )),
        }
    }

    fn __pow__(&self, power: Number, modulo: Option<i32>) -> PyResult<Self> {
        if modulo.unwrap_or(0) != 0 {
            panic!("Power function with mod not available for Dual.")
        }
        match power {
            Number::F64(f) => Ok(self.clone().pow(f)),
            Number::Dual(_d) => Err(PyTypeError::new_err(
                "Power operation not defined with Dual type exponent.",
            )),
            Number::Dual2(_d) => Err(PyTypeError::new_err(
                "Power operation not defined with Dual type exponent.",
            )),
        }
    }

    fn __exp__(&self) -> Self {
        self.exp()
    }

    fn __abs__(&self) -> f64 {
        self.abs().real
    }

    fn __log__(&self) -> Self {
        self.log()
    }

    fn __norm_cdf__(&self) -> Self {
        self.norm_cdf()
    }

    fn __norm_inv_cdf__(&self) -> Self {
        self.inv_norm_cdf()
    }

    fn __float__(&self) -> f64 {
        self.real
    }

    // JSON
    /// Create a JSON string representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::Dual2(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `Dual2` to JSON.",
            )),
        }
    }

    // Pickling
    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    fn __getnewargs__(&self) -> PyResult<(f64, Vec<String>, Vec<f64>, Vec<f64>)> {
        Ok((
            self.real,
            self.vars().iter().cloned().collect(),
            self.dual.to_vec(),
            self.dual2.clone().into_raw_vec(),
        ))
    }

    // Conversion
    #[pyo3(name = "to_dual")]
    fn to_dual_py(&self) -> Dual {
        self.clone().into()
    }
}
