#[cfg(test)]
mod tests;

pub mod dual;
use dual::dual1::Dual;
use dual::dual2::Dual2;
use dual::linalg::dsolve;

pub mod splines;
use splines::PPSpline;

// pub mod interpolate;
// pub mod point;

use ndarray::Array1;
// use point::{PointVec, Dual};

// use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "dsolve")]
fn dsolve_py(py: Python<'_>, a: Vec<Dual>, b: Vec<Dual>, allow_lsq: bool) -> Vec<Dual> {
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

#[pymodule]
fn rateslibrs(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_class::<PointVec>()?;
    m.add_class::<Dual>()?;
    m.add_class::<Dual2>()?;

    // m.add_class::<PPSpline<f64>>()?;
    m.add_function(wrap_pyfunction!(dsolve_py, m)?)?;
    Ok(())
}
