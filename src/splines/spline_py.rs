use crate::dual::dual::{Dual, Dual2, DualsOrF64};
use crate::splines::spline_f64::{bspldnev_single_f64, bsplev_single_f64, PPSpline};
use std::cmp::PartialEq;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

// use numpy::{PyArray1, ToPyArray, PyArrayMethods};
// use ndarray::Array1;

macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[pyclass]
        pub(crate) struct $name {
            inner: PPSpline<$type>,
        }
        #[pymethods]
        impl $name {
            #[new]
            fn new(k: usize, t: Vec<f64>, c: Option<Vec<$type>>) -> Self {
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

            /// Solve the spline coefficients given the data sites.
            ///
            /// Parameters
            /// ----------
            /// tau: list[f64]
            ///     The data site `x`-coordinates.
            /// y: list[type]
            ///     The data site `y`-coordinates in appropriate type (float, *Dual* or *Dual2*)
            ///     for *self*.
            /// left_n: int
            ///     The number of derivatives to evaluate at the left side of the data sites,
            ///     i.e. defining an endpoint constraint.
            /// right_n: int
            ///     The number of derivatives to evaluate at the right side of the datasites,
            ///     i.e. defining an endpoint constraint.
            /// allow_lsq: bool
            ///     Whether to permit least squares solving using non-square matrices.
            ///
            /// Returns
            /// -------
            /// None
            fn csolve(
                &mut self,
                tau: Vec<f64>,
                y: Vec<$type>,
                left_n: usize,
                right_n: usize,
                allow_lsq: bool
            ) -> PyResult<()> {
                self.inner.csolve(&tau, &y, left_n, right_n, allow_lsq)
            }

            /// Evaluate a single *x* coordinate value on the pp spline.
            ///
            /// Parameters
            /// ----------
            /// x: float
            ///     The x-axis value at which to evaluate value.
            ///
            /// Returns
            /// -------
            /// float, Dual or Dual2 based on self
            ///
            /// Notes
            /// -----
            /// The value of the spline at *x* is the sum of the value of each b-spline
            /// evaluated at *x* multiplied by the spline coefficients, *c*.
            ///
            /// .. math::
            ///
            ///    \$(x) = \sum_{i=1}^n c_i B_{(i,k,\mathbf{t})}(x)
            ///
            fn ppev_single(&self, x: DualsOrF64) -> PyResult<$type> {
                match x {
                    DualsOrF64::F64(f) => Ok(self.inner.ppdnev_single(&f, 0)),
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err(
                        "Cannot index PPSpline with `Dual`, use either `ppev_single(float(x))` or `ppev_single_dual(x)`."
                        )),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err(
                        "Cannot index PPSpline with `Dual2`, use either `ppev_single(float(x))` or `ppev_single_dual2(x)`.")),
                }
            }

            /// Evaluate a single *x* coordinate value on the pp spline.
            ///
            /// Parameters
            /// ----------
            /// x: Dual
            ///     The x-axis value at which to evaluate value.
            ///
            /// Returns
            /// -------
            /// Dual
            ///
            /// Notes
            /// -----
            /// This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual`
            /// sensitivities. It also prohibits type mixing and will raise if *Dual2* data types
            /// are encountered.
            fn ppev_single_dual(&self, x: DualsOrF64) -> PyResult<Dual> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual(&Dual::new(f, vec![]), 0),
                    DualsOrF64::Dual(d) => self.inner.ppdnev_single_dual(&d, 0),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppev_single_dual2(x)`.")),
                }
            }

            /// Evaluate a single *x* coordinate value on the pp spline.
            ///
            /// Parameters
            /// ----------
            /// x: Dual2
            ///     The x-axis value at which to evaluate value.
            ///
            /// Returns
            /// -------
            /// Dual2
            ///
            /// Notes
            /// -----
            /// This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual2`
            /// sensitivities. It also prohibits type mixing and will raise if *Dual* data types
            /// are encountered.
            fn ppev_single_dual2(&self, x: DualsOrF64) -> PyResult<Dual2> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual2(&Dual2::new(f, vec![]), 0),
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppev_single_dual(x)`.")),
                    DualsOrF64::Dual2(d) => self.inner.ppdnev_single_dual2(&d, 0),
                }
            }

            /// Evaluate an array of *x* coordinates derivatives on the pp spline.
            ///
            /// Repeatedly applies :meth:`~rateslib.splines.PPSplineF64.ppev_single`, and
            /// is typically used for minor performance gains in chart plotting.
            ///
            /// .. warning::
            ///
            ///    The *x* coordinates supplied to this function are treated as *float*, or are
            ///    **converted** to *float*. Therefore it does not guarantee the preservation of AD
            ///    sensitivities. If you need to index by *x* values which are
            ///    :class:`~rateslib.dual.Dual` or :class:`~rateslib.dual.Dual2`, then
            ///    you should choose to iteratively map the
            ///    provided methods :meth:`~rateslib.splines.PPSplineF64.ppev_single_dual` or
            ///    :meth:`~rateslib.splines.PPSplineF64.ppev_single_dual2` respectively.
            ///
            /// Returns
            /// -------
            /// 1-d array of float
            fn ppev(&self, x: Vec<f64>) -> PyResult<Vec<$type>> {
                let out: Vec<$type> = x.iter().map(|v| self.inner.ppdnev_single(&v, 0)).collect();
                Ok(out)
            }

            /// Evaluate a single *x* coordinate derivative from the right on the pp spline.
            ///
            /// Parameters
            /// ----------
            /// x: float
            ///     The x-axis value at which to evaluate value.
            /// m: int
            ///     The order of derivative to calculate value for (0 is function value).
            ///
            /// Returns
            /// -------
            /// float, Dual, or Dual2, based on self
            ///
            /// Notes
            /// -----
            /// The value of derivatives of the spline at *x* is the sum of the value of each
            /// b-spline derivatives evaluated at *x* multiplied by the spline
            /// coefficients, *c*.
            ///
            /// Due to the definition of the splines this derivative will return the value
            /// from the right at points where derivatives are discontinuous.
            ///
            /// .. math::
            ///
            ///    \frac{d^m\$(x)}{d x^m} = \sum_{i=1}^n c_i \frac{d^m B_{(i,k,\mathbf{t})}(x)}{d x^m}
            fn ppdnev_single(&self, x: DualsOrF64, m: usize) -> PyResult<$type> {
                match x {
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err("Splines cannot be indexed with Duals use `float(x)`.")),
                    DualsOrF64::F64(f) => Ok(self.inner.ppdnev_single(&f, m)),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Splines cannot be indexed with Duals use `float(x)`.")),
                }
            }

            /// Evaluate a single *x* coordinate derivative from the right on the pp spline.
            ///
            /// Parameters
            /// ----------
            /// x: Dual
            ///     The x-axis value at which to evaluate value.
            /// m: int
            ///     The order of derivative to calculate value for (0 is function value).
            ///
            /// Returns
            /// -------
            /// Dual
            ///
            /// Notes
            /// -----
            /// This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual`
            /// sensitivities. It also prohibits type mixing and will raise if any *Dual2*
            /// data types are encountered.
            fn ppdnev_single_dual(&self, x: DualsOrF64, m: usize) -> PyResult<Dual> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual(&Dual::new(f, vec![]), m),
                    DualsOrF64::Dual(d) => self.inner.ppdnev_single_dual(&d, m),
                    DualsOrF64::Dual2(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppdnev_single_dual2(x)`.")),
                }
            }

            /// Evaluate a single *x* coordinate derivative from the right on the pp spline.
            ///
            /// Parameters
            /// ----------
            /// x: Dual2
            ///     The x-axis value at which to evaluate value.
            /// m: int
            ///     The order of derivative to calculate value for (0 is function value).
            ///
            /// Returns
            /// -------
            /// Dual2
            ///
            /// Notes
            /// -----
            /// This function guarantees preservation of accurate AD :class:`~rateslib.dual.Dual2`
            /// sensitivities. It also prohibits type mixing and will raise if any *Dual*
            /// data types are encountered.
            fn ppdnev_single_dual2(&self, x: DualsOrF64, m: usize) -> PyResult<Dual2> {
                match x {
                    DualsOrF64::F64(f) => self.inner.ppdnev_single_dual2(&Dual2::new(f, vec![]), m),
                    DualsOrF64::Dual(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppdnev_single_dual(x)`.")),
                    DualsOrF64::Dual2(d) => self.inner.ppdnev_single_dual2(&d, m),
                }
            }

            /// Evaluate an array of x coordinates derivatives on the pp spline.
            ///
            /// Repeatedly applies :meth:`~rateslib.splines.PPSplineF64.ppdnev_single`.
            ///
            /// .. warning::
            ///
            ///    The *x* coordinates supplied to this function are treated as *float*, or are
            ///    **converted** to *float*. Therefore it does not guarantee the preservation of AD
            ///    sensitivities.
            ///
            /// Parameters
            /// ----------
            /// x: 1-d array of float
            ///     x-axis coordinates.
            /// m: int
            ///     The order of derivative to calculate value for.
            ///
            /// Returns
            /// -------
            /// 1-d array of float
            fn ppdnev(&self, x: Vec<f64>, m: usize) -> PyResult<Vec<$type>> {
                let out: Vec<$type> = x.iter().map(|v| self.inner.ppdnev_single(&v, m)).collect();
                Ok(out)
            }

            /// Evaluate value of the *i* th b-spline at x coordinates.
            ///
            /// Repeatedly applies :meth:`~rateslib.splines.bsplev_single`.
            ///
            /// .. warning::
            ///
            ///    The *x* coordinates supplied to this function are treated as *float*, or are
            ///    **converted** to *float*. Therefore it does not guarantee the preservation of AD
            ///    sensitivities.
            ///
            /// Parameters
            /// ----------
            /// x: 1-d array of float
            ///     x-axis coordinates
            /// i: int
            ///     Index of the B-spline to evaluate.
            ///
            /// Returns
            /// -------
            /// 1-d array of float
            fn bsplev(&self, x: Vec<f64>, i: usize) -> PyResult<Vec<f64>> {
                Ok(self.inner.bspldnev(&x, &i, &0))
            }

            fn bspldnev(&self, x: Vec<f64>, i: usize, m: usize) -> PyResult<Vec<f64>> {
                Ok(self.inner.bspldnev(&x, &i, &m))
            }

            fn __eq__(&self, other: &Self) -> PyResult<bool> {
                Ok(self.inner.eq(&other.inner))
            }

            fn __copy__(&self) -> Self {
                $name { inner: self.inner.clone() }
            }
        }
    };
}

create_interface!(PPSplineF64, f64);
create_interface!(PPSplineDual, Dual);
create_interface!(PPSplineDual2, Dual2);

#[pyfunction]
pub(crate) fn bsplev_single(
    x: f64,
    i: usize,
    k: usize,
    t: Vec<f64>,
    org_k: Option<usize>,
) -> PyResult<f64> {
    Ok(bsplev_single_f64(&x, i, &k, &t, org_k))
}

#[pyfunction]
pub(crate) fn bspldnev_single(
    x: f64,
    i: usize,
    k: usize,
    t: Vec<f64>,
    m: usize,
    org_k: Option<usize>,
) -> PyResult<f64> {
    Ok(bspldnev_single_f64(&x, i, &k, &t, m, org_k))
}
