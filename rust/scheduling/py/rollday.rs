use crate::scheduling::RollDay;
use pyo3::prelude::*;

#[pymethods]
impl RollDay {
    pub(crate) fn __str__(&self) -> String {
        match self {
            RollDay::Day(n) => format!("{n}"),
            RollDay::IMM() => "IMM".to_string(),
        }
    }
}
