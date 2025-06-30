//! Wrapper module to export to Python using pyo3 bindings.

use crate::calendars::adjuster::{Adjuster, Adjustment};
use crate::calendars::CalType;
use chrono::NaiveDateTime;
use pyo3::prelude::*;

#[pymethods]
impl Adjuster {
    /// Return a `date` under a date adjustment rule.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     Date to adjust.
    /// calendar: Cal, UnionCal or NamedCal
    ///     The calendar to assist with date adjustment.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "adjust")]
    fn adjust_py(&self, date: NaiveDateTime, calendar: CalType) -> NaiveDateTime {
        self.adjust(&date, &calendar)
    }

    /// Return a vector of `dates` adjusted under a date adjustment rule.
    ///
    /// Parameters
    /// ----------
    /// dates: list[datetime]
    ///     Dates to adjust.
    /// calendar: Cal, UnionCal or NamedCal
    ///     The calendar to assist with date adjustment.
    ///
    /// Returns
    /// -------
    /// list[datetime]
    #[pyo3(name = "adjusts")]
    fn adjusts_py(&self, dates: Vec<NaiveDateTime>, calendar: CalType) -> Vec<NaiveDateTime> {
        self.adjusts(&dates, &calendar)
    }
}
