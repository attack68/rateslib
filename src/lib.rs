#[cfg(test)]
mod tests;

pub mod dual;
// pub mod interpolate;
// pub mod point;
use dual::dual1::{Dual};
use dual::linalg::dsolve;
use numpy::{PyArray, ToPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray1, IntoPyArray};
use ndarray::{Array1, Array, arr1};
// use point::{PointVec, Dual};

// use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;


// wrapper of `axpy`
#[pyfunction]
#[pyo3(name = "dsolve")]
fn dsolve_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, Dual>,
    b: PyReadonlyArray1<'py, Dual>,
) -> &'py PyArray1<Dual> {
    let a = a.as_array();
    let b = b.as_array();
    b.into_pyarray(py)
    // let z = dsolve(&a, &b, allow_lsq);
    // z.to_pyarray(py)
}

#[pymodule]
fn rateslibrs(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_class::<PointVec>()?;
    m.add_class::<Dual>()?;
    m.add_function(wrap_pyfunction!(dsolve_py, m)?)?;
    Ok(())
}