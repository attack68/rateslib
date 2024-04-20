//! Wrapper module to export Rust dual data types to Python using pyo3 bindings.

use crate::dual::dual1::{Dual, Gradient1, MathFuncs, Vars};
use crate::dual::dual2::{Dual2, Gradient2};
use num_traits::{Pow, Signed};
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
// use pyo3::types::PyFloat;
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

#[derive(Debug, Clone, PartialEq, PartialOrd, FromPyObject)]
pub enum DualsOrF64 {
    Dual(Dual),
    Dual2(Dual2),
    F64(f64),
}

// https://github.com/PyO3/pyo3/discussions/3911
// #[derive(Debug, Clone, PartialEq, PartialOrd, FromPyObject)]
// pub enum DualsOrPyFloat<'py> {
//     Dual(Dual),
//     Dual2(Dual2),
//     Float(&'py PyFloat),
// }

#[pymethods]
impl Dual {
    #[new]
    fn new_py(real: f64, vars: Vec<String>, dual: Vec<f64>) -> PyResult<Self> {
        Dual::try_new(real, vars, dual)
    }

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
        Err(PyValueError::new_err("`Dual` variable cannot possess `dual2` attribute."))
    }

    #[pyo3(name = "grad1")]
    fn grad1<'py>(&'py self, py: Python<'py>, vars: Vec<String>) -> PyResult<Bound<'_, PyArray1<f64>>> {
        Ok(self.gradient1(vars).to_pyarray_bound(py))
    }

    #[pyo3(name = "grad2")]
    fn grad2<'py>(&'py self, _py: Python<'py>, _vars: Vec<String>) -> PyResult<&PyArray2<f64>> {
        Err(PyValueError::new_err("Cannot evaluate second order derivative on a Dual."))
    }

    #[pyo3(name = "ptr_eq")]
    fn ptr_eq_py(&self, other: &Dual) -> PyResult<bool> {
        Ok(Arc::ptr_eq(self.vars(), other.vars()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut _vars = Vec::from_iter(self.vars().iter().take(3).map(String::as_str)).join(", ");
        let mut _dual = Vec::from_iter(self.dual().iter().take(3).map(|x| format!("{:.1}", x))).join(", ");
        if self.vars().len() > 3 {
            _vars.push_str(", ...");
            _dual.push_str(", ...");
        }
        let fs = format!("<Dual: {:.6}, ({}), [{}]>", self.real(), _vars, _dual);
        Ok(fs)
    }

    fn __eq__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(d.eq(self)),
            DualsOrF64::F64(f) => Ok(Dual::new(f, Vec::new()).eq(self)),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __lt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self < &d),
            DualsOrF64::F64(f) => Ok(self < &f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __le__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self <= &d),
            DualsOrF64::F64(f) => Ok(self <= &f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __gt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self > &d),
            DualsOrF64::F64(f) => Ok(self > &f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __ge__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self >= &d),
            DualsOrF64::F64(f) => Ok(self >= &f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __add__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __radd__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __sub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self - d),
            DualsOrF64::F64(f) => Ok(self - f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __rsub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(d - self),
            DualsOrF64::F64(f) => Ok(f - self),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __mul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self * d),
            DualsOrF64::F64(f) => Ok(self * f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __rmul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(d * self),
            DualsOrF64::F64(f) => Ok(f * self),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __truediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self / d),
            DualsOrF64::F64(f) => Ok(self / f),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __rtruediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(d / self),
            DualsOrF64::F64(f) => Ok(f / self),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __pow__(&self, power: DualsOrF64, modulo: Option<i32>) -> PyResult<Self> {
        if modulo.unwrap_or(0) != 0 {
            panic!("Power function with mod not available for Dual.")
        }
        match power {
            DualsOrF64::F64(f) => Ok(self.clone().pow(f)),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Power operation not defined with Dual type exponent.")),
            DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Power operation not defined with Dual type exponent."))
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

    fn __norm_cdf__(&self) -> Self { self.norm_cdf() }

    fn __norm_inv_cdf__(&self) -> Self { self.inv_norm_cdf() }

    fn __float__(&self) -> f64 {
        self.real()
    }
}

#[pymethods]
impl Dual2 {
    /// Python wrapper to construct a new `Dual2`.
    #[new]
    pub fn new_py(real: f64, vars: Vec<String>, dual: Vec<f64>, dual2: Vec<f64>) -> PyResult<Self> {
        Dual2::try_new(real, vars, dual, dual2)
    }

    /// Python wrapper to construct a new `Dual2` using the Arc pointer of another.
    #[staticmethod]
    pub fn vars_from(other: &Dual2, real: f64, vars: Vec<String>, dual: Vec<f64>, dual2: Vec<f64>) -> PyResult<Self> {
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
    fn grad1_py<'py>(&'py self, py: Python<'py>, vars: Vec<String>) -> PyResult<Bound<'_, PyArray1<f64>>> {
        Ok(self.gradient1(vars).to_pyarray_bound(py))
    }

    #[pyo3(name = "grad2")]
    fn grad2_py<'py>(&'py self, py: Python<'py>, vars: Vec<String>) -> PyResult<Bound<'_, PyArray2<f64>>> {
        Ok(self.gradient2(vars).to_pyarray_bound(py))
    }

    #[pyo3(name = "grad1_manifold")]
    fn grad1_manifold_py<'py>(&'py self, _py: Python<'py>, vars: Vec<String>) -> PyResult<Vec<Dual2>> {
        let out = self.gradient1_manifold(vars);
        Ok(out.into_raw_vec())
    }

    #[pyo3(name = "ptr_eq")]
    fn ptr_eq_py(&self, other: &Dual2) -> PyResult<bool> {
        Ok(self.ptr_eq(&other))
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut _vars = Vec::from_iter(self.vars.iter().take(3).map(String::as_str)).join(", ");
        let mut _dual = Vec::from_iter(self.dual.iter().take(3).map(|x| format!("{:.1}", x))).join(", ");
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

    fn __eq__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d.eq(self)),
            DualsOrF64::F64(f) => Ok(Dual2::new(f, Vec::new()).eq(self)),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __lt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self < &d),
            DualsOrF64::F64(f) => Ok(self < &f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __le__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self <= &d),
            DualsOrF64::F64(f) => Ok(self <= &f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __gt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self > &d),
            DualsOrF64::F64(f) => Ok(self > &f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __ge__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self >= &d),
            DualsOrF64::F64(f) => Ok(self >= &f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __add__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __radd__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __sub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self - d),
            DualsOrF64::F64(f) => Ok(self - f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __rsub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d - self),
            DualsOrF64::F64(f) => Ok(f - self),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __mul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self * d),
            DualsOrF64::F64(f) => Ok(self * f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __rmul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d * self),
            DualsOrF64::F64(f) => Ok(f * self),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __truediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self / d),
            DualsOrF64::F64(f) => Ok(self / f),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __rtruediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d / self),
            DualsOrF64::F64(f) => Ok(f / self),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __pow__(&self, power: DualsOrF64, modulo: Option<i32>) -> PyResult<Self> {
        if modulo.unwrap_or(0) != 0 {
            panic!("Power function with mod not available for Dual.")
        }
        match power {
            DualsOrF64::F64(f) => Ok(self.clone().pow(f)),
            DualsOrF64::Dual(_d) => Err(PyTypeError::new_err("Power operation not defined with Dual type exponent.")),
            DualsOrF64::Dual2(_d) => Err(PyTypeError::new_err("Power operation not defined with Dual type exponent."))
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

    fn __norm_cdf__(&self) -> Self { self.norm_cdf() }

    fn __norm_inv_cdf__(&self) -> Self { self.inv_norm_cdf() }

    fn __float__(&self) -> f64 {
        self.real
    }
}