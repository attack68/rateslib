#[cfg(test)]
mod tests;

pub mod dual;
// pub mod interpolate;
// pub mod point;
use dual::dual1::Dual;
use dual::linalg::dsolve;
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
fn dsolve_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, Dual>,
    b: PyReadonlyArray1<'py, Dual>,
) -> &'py PyArray1<Dual> {
    println!("reading a as array");
    let a = a.as_array();
    println!("reading b as array");
    let b = b.as_array();
    println!("constructing c as array");
    let c = arr1(&[Dual::new(2.0, Vec::new(), Vec::new())])
            .to_pyarray(py);
    println!("returning c");
    c
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
