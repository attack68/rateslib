use crate::splines::spline_f64::{PPSpline, bsplev_single_f64, bspldnev_single_f64};
use crate::dual::dual1::Dual;
use crate::dual::dual2::Dual2;
use crate::dual::dual_py::{DualsOrF64};
use std::cmp::PartialEq;

use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;

// use numpy::{PyArray1, ToPyArray, PyArrayMethods};
// use ndarray::Array1;

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
            ) -> PyResult<()> {
                self.inner.csolve(&tau, &y, left_n, right_n, allow_lsq)
            }

            pub fn ppev_single(&self, x: DualsOrF64) -> PyResult<$type> {
                match x {
                    DualsOrF64::F64(f) => Ok(self.inner.ppdnev_single(&f, 0)),
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err(
                        "Cannot index PPSpline with `Dual`, use either `ppev_single(float(x))` or `ppev_single_dual(x)`."
                        )),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err(
                        "Cannot index PPSpline with `Dual2`, use either `ppev_single(float(x))` or `ppev_single_dual2(x)`.")),
                }
            }

            pub fn ppev_single_dual(&self, x: DualsOrF64) -> PyResult<Dual> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual(&Dual::new(f, vec![]), 0),
                    DualsOrF64::Dual(d) => self.inner.ppdnev_single_dual(&d, 0),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppev_single_dual2(x)`.")),
                }
            }

            pub fn ppev_single_dual2(&self, x: DualsOrF64) -> PyResult<Dual2> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual2(&Dual2::new(f, vec![]), 0),
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppev_single_dual(x)`.")),
                    DualsOrF64::Dual2(d) => self.inner.ppdnev_single_dual2(&d, 0),
                }
            }

            pub fn ppev<'py>(&'py self, x: Vec<f64>) -> PyResult<Vec<$type>> {
                let out: Vec<$type> = x.iter().map(|v| self.inner.ppdnev_single(&v, 0)).collect();
                Ok(out)
            }

            pub fn ppdnev_single(&self, x: DualsOrF64, m: usize) -> PyResult<$type> {
                match x {
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err("Splines cannot be indexed with Duals use `float(x)`.")),
                    DualsOrF64::F64(f) => Ok(self.inner.ppdnev_single(&f, m)),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Splines cannot be indexed with Duals use `float(x)`.")),
                }
            }

            pub fn ppdnev_single_dual(&self, x: DualsOrF64, m: usize) -> PyResult<Dual> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual(&Dual::new(f, vec![]), m),
                    DualsOrF64::Dual(d) => self.inner.ppdnev_single_dual(&d, m),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppdnev_single_dual2(x)`.")),
                }
            }

            pub fn ppdnev_single_dual2(&self, x: DualsOrF64, m: usize) -> PyResult<Dual2> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual2(&Dual2::new(f, vec![]), m),
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppdnev_single_dual(x)`.")),
                    DualsOrF64::Dual2(d) => self.inner.ppdnev_single_dual2(&d, m),
                }
            }

            pub fn ppdnev<'py>(&'py self, x: Vec<f64>, m: usize) -> PyResult<Vec<$type>> {
                let out: Vec<$type> = x.iter().map(|v| self.inner.ppdnev_single(&v, m)).collect();
                Ok(out)
            }

            pub fn bsplev<'py>(&'py self, x: Vec<f64>, i: usize) -> PyResult<Vec<f64>> {
                Ok(self.inner.bspldnev(&x, &i, &0))
            }

            pub fn bspldnev<'py>(&'py self, x: Vec<f64>, i: usize, m: usize) -> PyResult<Vec<f64>> {
                Ok(self.inner.bspldnev(&x, &i, &m))
            }

            pub fn __eq__(&self, other: &Self) -> PyResult<bool> {
                Ok(self.inner.eq(&other.inner))
            }

            pub fn __copy__(&self) -> Self {
                $name { inner: self.inner.clone() }
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
