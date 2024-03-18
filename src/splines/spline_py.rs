use crate::splines::spline_f64::{PPSpline, bsplev_single_f64, bspldnev_single_f64};
use crate::dual::dual1::Dual;
use crate::dual::dual2::Dual2;

use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::exceptions::{PyTypeError, PyValueError};
use numpy::{PyArray1, ToPyArray};
use ndarray::Array1;

macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub struct $name {
            inner: PPSpline<$type>,
        }
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(k: usize, t: Vec<f64>, c: Option<Vec<$type>>) -> Self {
                Self {
                    inner: PPSpline::new(k, t, c),
                }
            }

            #[getter]
            fn n(&self) -> PyResult<usize> {
                Ok(*self.inner.n())
            }

            #[getter]
            fn k(&self) -> PyResult<usize> {
                Ok(*self.inner.k())
            }

            #[getter]
            fn t(&self) -> PyResult<Vec<f64>> {
                Ok(self.inner.t().clone())
            }

            #[getter]
            fn c(&self) -> PyResult<Option<Vec<$type>>> {
                match self.inner.c() {
                    Some(val) => Ok(Some(val.clone().into_raw_vec())),
                    None => Ok(None)
                }
            }

            pub fn csolve(
                &mut self,
                tau: Vec<f64>,
                y: Vec<$type>,
                left_n: usize,
                right_n: usize,
                allow_lsq: bool
            ) {
                self.inner.csolve(&tau, &y, left_n, right_n, allow_lsq)
            }

            pub fn ppev_single(&self, x: f64) -> $type {
                self.inner.ppev_single(&x)
            }

            pub fn bsplev<'py>(&'py self, py: Python<'py>, x: &PyArray1<f64>, i: usize) -> PyResult<&PyArray1<f64>> {
                Ok(Array1::from_vec(self.inner.bsplev(&x.to_vec().expect(""), &i)).to_pyarray(py))
            }

            pub fn bspldnev<'py>(&'py self, py: Python<'py>, x: &PyArray1<f64>, i: usize, m: usize) -> PyResult<&PyArray1<f64>> {
                Ok(Array1::from_vec(self.inner.bspldnev(&x.to_vec().expect(""), &i, &m)).to_pyarray(py))
            }
        }
    };
}

create_interface!(PPSplineF64, f64);
create_interface!(PPSplineDual, Dual);
create_interface!(PPSplineDual2, Dual2);

#[pyfunction]
pub fn bsplev_single(x: f64, i: usize, k: usize, t: Vec<f64>, org_k: Option<usize>) -> PyResult<f64> {
    Ok(bsplev_single_f64(&x, i, &k, &t, org_k))
}

#[pyfunction]
pub fn bspldnev_single(x: f64, i: usize, k: usize, t: Vec<f64>, m: usize, org_k: Option<usize>) -> PyResult<f64> {
    Ok(bspldnev_single_f64(&x, i, &k, &t, m, org_k))
}