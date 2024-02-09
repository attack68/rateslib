#[cfg(test)]
mod tests;

pub mod dual;
pub mod interpolate;
// pub mod point;
use dual::dual1::Dual;
// use point::{PointVec, Dual};

// use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

#[pymodule]
fn rateslibrs(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_class::<PointVec>()?;
    m.add_class::<Dual>()?;
    return Ok(())
}