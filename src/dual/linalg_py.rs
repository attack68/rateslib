use crate::dual::linalg::dsolve;
use crate::dual::dual2::Dual2;
use crate::dual::dual1::Dual;
use pyo3::prelude::*;
use ndarray::Array1;
use num_traits::identities::{Zero};
use num_traits::{Signed};
use std::cmp::PartialOrd;
use std::iter::Sum;
use std::ops::{Div, Mul, Sub};


fn dsolve_py<T>(a: Vec<T>, b: Vec<T>, allow_lsq: bool) -> Vec<T>
where T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a T, Output = T>,
    for<'a> &'a T: Mul<&'a f64, Output = T> + Sub<T, Output = T> + Mul<&'a T, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
{
    // requires row major order of numpy.
    // &'py PyArray1<Dual>
    let a1 = Array1::from_vec(a);
    let b_ = Array1::from_vec(b);
    let (r, c) = (a1.len() / b_.len(), b_.len());
    let a2 = a1
        .into_shape((r, c))
        .expect("Inputs `a` and `b` for dual solve were incorrect shapes");
    let out = dsolve(&a2.view(), &b_.view(), allow_lsq);
    out.into_raw_vec()
}

#[pyfunction]
#[pyo3(name = "_dsolve1")]
pub fn dsolve1_py(py: Python<'_>, a: Vec<Dual>, b: Vec<Dual>, allow_lsq: bool) -> PyResult<Vec<Dual>>
{
    Ok(dsolve_py(a, b, allow_lsq))
}

#[pyfunction]
#[pyo3(name = "_dsolve2")]
pub fn dsolve2_py(py: Python<'_>, a: Vec<Dual2>, b: Vec<Dual2>, allow_lsq: bool) -> PyResult<Vec<Dual2>> {
    Ok(dsolve_py(a, b, allow_lsq))
}