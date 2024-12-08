use crate::dual::{Dual, Dual2};
use crate::splines::{PPSplineDual, PPSplineDual2, PPSplineF64};
use ndarray::{Array1, Array2};
use pyo3::{pyclass, FromPyObject, PyErr};
use serde::{Deserialize, Serialize};

/// Defines the order of gradients available in a calculation with AD.
#[pyclass(module = "rateslib.rs", eq, eq_int)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ADOrder {
    /// Floating point arithmetic only.
    Zero,
    /// Derivatives available to first order.
    One,
    /// Derivatives available to second order.
    Two,
}

/// Container for the three core numeric types; [f64], [Dual] and [Dual2].
#[derive(Debug, Clone, FromPyObject, Serialize, Deserialize)]
pub enum Number {
    Dual(Dual),
    Dual2(Dual2),
    F64(f64),
}

/// Container for [Vec] of each core numeric type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NumberVec {
    F64(Vec<f64>),
    Dual(Vec<Dual>),
    Dual2(Vec<Dual2>),
}

/// Container for [Array1] of each core numeric type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NumberArray1 {
    F64(Array1<f64>),
    Dual(Array1<Dual>),
    Dual2(Array1<Dual2>),
}

/// Container for [Array2] of each core numeric type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NumberArray2 {
    F64(Array2<f64>),
    Dual(Array2<Dual>),
    Dual2(Array2<Dual2>),
}

/// Container for [PPSpline] definitive type variants.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub enum NumberPPSpline {
    F64(PPSplineF64),
    Dual(PPSplineDual),
    Dual2(PPSplineDual2),
}

impl NumberPPSpline {
    /// Return a reference to the knot sequence of the inner pp-spline.
    pub fn t(&self) -> &Vec<f64> {
        match self {
            NumberPPSpline::F64(s) => s.inner.t(),
            NumberPPSpline::Dual(s) => s.inner.t(),
            NumberPPSpline::Dual2(s) => s.inner.t(),
        }
    }
}

/// Generic trait indicating a function exists to map one [Number] to another.
///
/// An example of this trait is used by certain [PPSpline] indicating that an x-value as
/// some [Number] can be mapped under spline interpolation to some y-value as another [Number].
pub trait NumberMapping {
    fn mapped_value(&self, x: &Number) -> Result<Number, PyErr>;
}
