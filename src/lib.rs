//! This is the documentation for rateslib-rs

#[cfg(test)]
mod tests;

// type GenericError = Box<dyn std::error::Error + Send + Sync + 'static>;
// type GenericResult<T> = Result<T, GenericError>;

use pyo3::prelude::*;

pub mod dual;
use dual::dual1::Dual;
use dual::dual2::Dual2;
use dual::linalg_py::{dsolve1_py, dsolve2_py, fdsolve1_py, fdsolve2_py};

pub mod splines;
use splines::spline_py::{
    bspldnev_single, bsplev_single, PPSplineDual, PPSplineDual2, PPSplineF64,
};

pub mod curves;
use curves::interpolation_py::index_left_f64;

pub mod calendars;
use calendars::calendar::{Cal, Modifier, RollDay, UnionCal};
use calendars::calendar_py::get_calendar_by_name_py;

pub mod fx;
use fx::rates::{Ccy, FXRate, FXRates};

#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Automatic Differentiation
    m.add_class::<Dual>()?;
    m.add_class::<Dual2>()?;
    m.add_function(wrap_pyfunction!(dsolve1_py, m)?)?;
    m.add_function(wrap_pyfunction!(dsolve2_py, m)?)?;
    m.add_function(wrap_pyfunction!(fdsolve1_py, m)?)?;
    m.add_function(wrap_pyfunction!(fdsolve2_py, m)?)?;

    // Splines
    m.add_class::<PPSplineF64>()?;
    m.add_class::<PPSplineDual>()?;
    m.add_class::<PPSplineDual2>()?;
    m.add_function(wrap_pyfunction!(bsplev_single, m)?)?;
    m.add_function(wrap_pyfunction!(bspldnev_single, m)?)?;

    // Curves
    m.add_function(wrap_pyfunction!(index_left_f64, m)?)?;

    // Calendars
    m.add_class::<Cal>()?;
    m.add_class::<UnionCal>()?;
    m.add_class::<Modifier>()?;
    m.add_class::<RollDay>()?;
    m.add_function(wrap_pyfunction!(get_calendar_by_name_py, m)?)?;

    // FX
    m.add_class::<Ccy>()?;
    m.add_class::<FXRate>()?;
    m.add_class::<FXRates>()?;

    Ok(())
}
