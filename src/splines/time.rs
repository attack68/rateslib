use chrono::{DateTime, Utc};
use pyo3::{Python, ToPyObject, pymethod};

#[pymethod]
fn add_one_day(a: &DateTime) -> PyResult<>