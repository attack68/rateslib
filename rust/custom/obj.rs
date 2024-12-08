use crate::calendars::ndt;
use chrono::prelude::*;
use pyo3::{pyclass};

/// A empty struct with a method to push data to Python
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Default, Debug)]
pub struct DataProvider {
    pub(crate) dates: Vec<NaiveDateTime>,
}

impl DataProvider {
    pub fn new() -> Self {
        Self {
            dates: vec![
                ndt(2001, 1, 1),
                ndt(2001, 1, 2),
                ndt(2001, 1, 3),
            ]
        }
    }

    // pub fn get_dates(&self) -> Vec<NaiveDateTime> {
    //     self.dates.clone()
    // }
}
