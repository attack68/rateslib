use crate::scheduling::frequency::Imm;

use chrono::prelude::*;
use pyo3::prelude::*;

#[pymethods]
impl Imm {
    /// Return the next IMM date after ``date`` under the given definition.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     The input date.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "next")]
    fn next_py(&self, date: NaiveDateTime) -> NaiveDateTime {
        self.next(&date)
    }

    /// Check whether a date is an IMM date under the given definition.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     The input date.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(name = "validate")]
    fn validate_py(&self, date: NaiveDateTime) -> bool {
        self.validate(&date)
    }

    /// Return an IMM date from a given year and month under the given definition.
    ///
    /// Parameters
    /// ----------
    /// year: int
    ///     The year.
    /// month: int
    ///     The month.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "get")]
    fn from_ym_opt_py(&self, year: i32, month: u32) -> PyResult<NaiveDateTime> {
        self.from_ym_opt(year, month)
    }
}
