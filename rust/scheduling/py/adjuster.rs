//! Wrapper module to export to Python using pyo3 bindings.

use crate::scheduling::{Adjuster, Adjustment, Calendar};
use chrono::NaiveDateTime;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

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
    fn adjust_py(&self, date: NaiveDateTime, calendar: Calendar) -> NaiveDateTime {
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
    fn adjusts_py(&self, dates: Vec<NaiveDateTime>, calendar: Calendar) -> Vec<NaiveDateTime> {
        self.adjusts(&dates, &calendar)
    }
}

pub(crate) fn get_roll_adjuster_from_str(input: (&str, bool)) -> Result<Adjuster, PyErr> {
    let hmap: HashMap<(&str, bool), Adjuster> = HashMap::from([
        (("act", true), Adjuster::Actual {}),
        (("actual", true), Adjuster::Actual {}),
        (("f", true), Adjuster::FollowingSettle {}),
        (("mf", true), Adjuster::ModifiedFollowingSettle {}),
        (("p", true), Adjuster::PreviousSettle {}),
        (("mp", true), Adjuster::ModifiedPreviousSettle {}),
        (("act", false), Adjuster::Actual {}),
        (("actual", false), Adjuster::Actual {}),
        (("f", false), Adjuster::Following {}),
        (("mf", false), Adjuster::ModifiedFollowing {}),
        (("p", false), Adjuster::Previous {}),
        (("mp", false), Adjuster::ModifiedPrevious {}),
    ]);
    match hmap.get(&input) {
        None => Err(PyValueError::new_err(format!(
            "'{}', with '{}' settlement, is not found in the list of allowed roll adjusters.",
            input.0, input.1
        ))),
        Some(value) => Ok(*value),
    }
}
