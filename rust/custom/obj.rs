use crate::calendars::ndt;
use chrono::prelude::*;
use pyo3::{pyclass};

/// A empty struct with a method to push data to Python
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Default, Debug)]
pub struct DataProvider {
    #[pyo3(get, set)]
    pub(crate) dates: Vec<NaiveDateTime>,
    #[pyo3(get, set)]
    pub(crate) val: f64,
    #[pyo3(get, set)]
    pub(crate) vals: Vec<f64>,
}

impl DataProvider {
    pub fn new() -> Self {
        Self {
            dates: vec![
                ndt(2001, 1, 1),
                ndt(2001, 1, 2),
                ndt(2001, 1, 3),
                ndt(2001, 1, 4),
                ndt(2001, 1, 5),
            ],
            val: 3.5_f64,
            vals: vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64],
        }
    }
}
