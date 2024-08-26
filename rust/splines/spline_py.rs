//! Wrapper to export spline functionality to Python

use crate::dual::{Dual, Dual2, Number};
use crate::splines::spline::{bspldnev_single_f64, bsplev_single_f64, PPSpline};
use std::cmp::PartialEq;

use numpy::{PyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Container for the three core spline types; `f64`, `Dual` and `Dual2`
#[derive(Clone, FromPyObject, Serialize, Deserialize)]
pub enum Spline {
    F64(PPSplineF64),
    Dual(PPSplineDual),
    Dual2(PPSplineDual2),
}

impl IntoPy<PyObject> for Spline {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Spline::F64(s) => Py::new(py, s).unwrap().to_object(py),
            Spline::Dual(s) => Py::new(py, s).unwrap().to_object(py),
            Spline::Dual2(s) => Py::new(py, s).unwrap().to_object(py),
        }
    }
}

macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[pyclass(module = "rateslib.rs")]
        #[derive(Clone, Deserialize, Serialize)]
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
            fn ppev_single(&self, x: Number) -> PyResult<$type> {
                match x {
                    Number::F64(f) => self.inner.ppdnev_single(&f, 0),
                    Number::Dual(_) => Err(PyTypeError::new_err(
                        "Cannot index PPSpline with `Dual`, use either `ppev_single(float(x))` or `ppev_single_dual(x)`."
                        )),
                    Number::Dual2(_) => Err(PyTypeError::new_err(
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
            fn ppev_single_dual(&self, x: Number) -> PyResult<Dual> {
                match x {
                    Number::F64(f) => self.inner.ppdnev_single_dual(&Dual::new(f, vec![]), 0),
                    Number::Dual(d) => self.inner.ppdnev_single_dual(&d, 0),
                    Number::Dual2(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppev_single_dual2(x)`.")),
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
            fn ppev_single_dual2(&self, x: Number) -> PyResult<Dual2> {
                match x {
                    Number::F64(f) => self.inner.ppdnev_single_dual2(&Dual2::new(f, vec![]), 0),
                    Number::Dual(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppev_single_dual(x)`.")),
                    Number::Dual2(d) => self.inner.ppdnev_single_dual2(&d, 0),
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
                let out: Vec<$type> = x.iter().map(|v| self.inner.ppdnev_single(&v, 0)).collect::<Result<Vec<$type>, _>>()?;
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
            fn ppdnev_single(&self, x: Number, m: usize) -> PyResult<$type> {
                match x {
                    Number::Dual(_) => Err(PyTypeError::new_err("Splines cannot be indexed with Duals use `float(x)`.")),
                    Number::F64(f) => self.inner.ppdnev_single(&f, m),
                    Number::Dual2(_) => Err(PyTypeError::new_err("Splines cannot be indexed with Duals use `float(x)`.")),
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
            fn ppdnev_single_dual(&self, x: Number, m: usize) -> PyResult<Dual> {
                match x {
                    Number::F64(f) => self.inner.ppdnev_single_dual(&Dual::new(f, vec![]), m),
                    Number::Dual(d) => self.inner.ppdnev_single_dual(&d, m),
                    Number::Dual2(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppdnev_single_dual2(x)`.")),
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
            fn ppdnev_single_dual2(&self, x: Number, m: usize) -> PyResult<Dual2> {
                match x {
                    Number::F64(f) => self.inner.ppdnev_single_dual2(&Dual2::new(f, vec![]), m),
                    Number::Dual(_) => Err(PyTypeError::new_err("Cannot mix `Dual2` and `Dual` types, use `ppdnev_single_dual(x)`.")),
                    Number::Dual2(d) => self.inner.ppdnev_single_dual2(&d, m),
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
                let out: Vec<$type> = x.iter().map(|v| self.inner.ppdnev_single(&v, m)).collect::<Result<Vec<$type>, _>>()?;
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

            /// Evaluate *m* order derivative on the *i* th b-spline at *x* coordinates.
            ///
            /// Repeatedly applies :meth:`~rateslib.splines.bspldnev_single`.
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
            /// i: int
            ///     The index of the B-spline to evaluate.
            /// m: int
            ///     The order of derivative to calculate value for.
            ///
            /// Returns
            /// -------
            /// 1-d array
            fn bspldnev(&self, x: Vec<f64>, i: usize, m: usize) -> PyResult<Vec<f64>> {
                Ok(self.inner.bspldnev(&x, &i, &m))
            }

            /// Evaluate the 2d spline collocation matrix at each data site.
            ///
            /// Parameters
            /// ----------
            /// tau: 1-d array of float
            ///     The data sites `x`-axis values which will instruct the pp spline.
            /// left_n: int
            ///     The order of derivative to use for the left most data site and top row
            ///     of the spline collocation matrix.
            /// right_n: int
            ///     The order of derivative to use for the right most data site and bottom row
            ///     of the spline collocation matrix.
            ///
            /// Returns
            /// -------
            /// 2-d array of float
            ///
            /// Notes
            /// -----
            /// The spline collocation matrix is defined as,
            ///
            /// .. math::
            ///
            ///    [\mathbf{B}_{k, \mathbf{t}}(\mathbf{\tau})]_{j,i} = B_{i,k,\mathbf{t}}(\tau_j)
            ///
            /// where each row is a call to :meth:`~rateslib.splines.PPSplineF64.bsplev`, except the top and bottom rows
            /// which can be specifically adjusted to account for
            /// ``left_n`` and ``right_n`` such that, for example, the first row might be,
            ///
            /// .. math::
            ///
            ///    [\mathbf{B}_{k, \mathbf{t}}(\mathbf{\tau})]_{1,i} = \frac{d^n}{dx}B_{i,k,\mathbf{t}}(\tau_1)
            fn bsplmatrix<'py>(
                &'py self,
                py: Python<'py>,
                tau: Vec<f64>,
                left_n: usize,
                right_n: usize
            ) -> PyResult<Bound<'_, PyArray2<f64>>> {
                Ok(self.inner.bsplmatrix(&tau, left_n, right_n).to_pyarray_bound(py))
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

/// Calculate the value of an indexed b-spline at *x*.
///
/// Parameters
/// ----------
/// x: float
///     The *x* value at which to evaluate the b-spline.
/// i: int
///     The index of the b-spline to evaluate.
/// k: int
///     The order of the b-spline (note that k=4 is a cubic spline).
/// t: sequence of float
///     The knot sequence of the pp spline.
/// org_k: int, optional
///     The original k input. Used only internally when recursively calculating
///     successive b-splines. Users will not typically use this parameters.
///
/// Notes
/// -----
/// B-splines can be recursively defined as:
///
/// .. math::
///
///    B_{i,k,\mathbf{t}}(x) = \frac{x-t_i}{t_{i+k-1}-t_i}B_{i,k-1,\mathbf{t}}(x) + \frac{t_{i+k}-x}{t_{i+k}-t_{i+1}}B_{i+1,k-1,\mathbf{t}}(x)
///
/// and such that the basic, stepwise, b-spline or order 1 are:
///
/// .. math::
///
///    B_{i,1,\mathbf{t}}(x) = \left \{ \begin{matrix} 1, & t_i \leq x < t_{i+1} \\ 0, & \text{otherwise} \end{matrix} \right .
///
/// For continuity on the right boundary the rightmost basic b-spline is also set equal
/// to 1 there: :math:`B_{n,1,\mathbf{t}}(t_{n+k})=1`.
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

/// Calculate the *m* th order derivative (from the right) of an indexed b-spline at *x*.
///
/// Parameters
/// ----------
/// x: float
///     The *x* value at which to evaluate the b-spline.
/// i: int
///     The index of the b-spline to evaluate.
/// k: int
///     The order of the b-spline (note that k=4 is a cubic spline).
/// t: sequence of float
///     The knot sequence of the pp spline.
/// m: int
///     The order of the derivative of the b-spline to evaluate.
/// org_k: int, optional
///     The original k input. Used only internally when recursively calculating
///     successive b-splines. Users will not typically use this parameter.
///
/// Notes
/// -----
/// B-splines derivatives can be recursively defined as:
///
/// .. math::
///
///    \frac{d}{dx}B_{i,k,\mathbf{t}}(x) = (k-1) \left ( \frac{B_{i,k-1,\mathbf{t}}(x)}{t_{i+k-1}-t_i} - \frac{B_{i+1,k-1,\mathbf{t}}(x)}{t_{i+k}-t_{i+1}} \right )
///
/// and such that the basic, stepwise, b-spline derivative is:
///
/// .. math::
///
///    \frac{d}{dx}B_{i,1,\mathbf{t}}(x) = 0
///
/// During this recursion the original order of the spline is registered so that under
/// the given knot sequence, :math:`\mathbf{t}`, lower order b-splines which are not
/// the rightmost will register a unit value. For example, the 4'th order knot sequence
/// [1,1,1,1,2,2,2,3,4,4,4,4] defines 8 b-splines. The rightmost is measured
/// across the knots [3,4,4,4,4]. When the knot sequence remains constant and the
/// order is lowered to 3 the rightmost, 9'th, b-spline is measured across [4,4,4,4],
/// which is effectively redundant since its domain has zero width. The 8'th b-spline
/// which is measured across the knots [3,4,4,4] is that which will impact calculations
/// and is therefore given the value 1 at the right boundary. This is controlled by
/// the information provided by ``org_k``.
///
/// Examples
/// --------
/// The derivative of the 4th b-spline of the following knot sequence
/// is discontinuous at `x` = 2.0.
///
/// .. ipython:: python
///
///    t = [1,1,1,1,2,2,2,3,4,4,4,4]
///    bspldnev_single(x=2.0, i=3, k=4, t=t, m=1)
///    bspldnev_single(x=1.99999999, i=3, k=4, t=t, m=1)
///
/// .. plot::
///
///    from rateslib.splines import *
///    import matplotlib.pyplot as plt
///    from datetime import datetime as dt
///    import numpy as np
///    t = [1,1,1,1,2,2,2,3,4,4,4,4]
///    spline = PPSpline(k=4, t=t)
///    x = np.linspace(1, 4, 76)
///    fix, ax = plt.subplots(1,1)
///    ax.plot(x, spline.bspldnev(x, 3, 0))
///    plt.show()
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
