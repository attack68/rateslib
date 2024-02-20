#[cfg(test)]
mod tests;

pub mod dual;
// pub mod interpolate;
// pub mod point;
use dual::dual1::Dual;
use dual::linalg::{dmul11_, dsolve};
use ndarray::{arr1, Array, Array1};
use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
// use point::{PointVec, Dual};

// use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

// wrapper of `axpy`
#[pyfunction]
#[pyo3(name = "dsolve")]
fn dsolve_py<'py>(py: Python<'py>, a: Vec<Dual>, b: Vec<Dual>, allow_lsq: bool) -> Vec<Dual> {
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
    m.add_function(wrap_pyfunction!(dsolve_py, m)?)?;
    Ok(())
}
