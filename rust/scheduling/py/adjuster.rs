//! Wrapper module to export to Python using pyo3 bindings.

use crate::scheduling::{Adjuster, Adjustment, Calendar};
use chrono::NaiveDateTime;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// A list of rules for performing date adjustment.
#[pyclass(module = "rateslib.rs", name="Adjuster", eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PyAdjuster {
    /// Actual date without adjustment.
    #[pyo3(constructor = (_u8=0))]
    Actual { _u8: u8 },
    /// Following adjustment rule.
    #[pyo3(constructor = (_u8=1))]
    Following { _u8: u8 },
    /// Modified following adjustment rule.
    #[pyo3(constructor = (_u8=2))]
    ModifiedFollowing { _u8: u8 },
    /// Previous adjustment rule.
    #[pyo3(constructor = (_u8=3))]
    Previous { _u8: u8 },
    /// Modified previous adjustment rule.
    #[pyo3(constructor = (_u8=4))]
    ModifiedPrevious { _u8: u8 },
    /// Following adjustment rule, enforcing settlement calendar.
    #[pyo3(constructor = (_u8=5))]
    FollowingSettle { _u8: u8 },
    /// Modified following adjustment rule, enforcing settlement calendar.
    #[pyo3(constructor = (_u8=6))]
    ModifiedFollowingSettle { _u8: u8 },
    /// Previous adjustment rule, enforcing settlement calendar.
    #[pyo3(constructor = (_u8=7))]
    PreviousSettle { _u8: u8 },
    /// Modified previous adjustment rule, enforcing settlement calendar.
    #[pyo3(constructor = (_u8=8))]
    ModifiedPreviousSettle { _u8: u8 },
    /// A set number of business days, defined by a given calendar,
    /// using calendar lag rules and enforcing settlement calendars.
    #[pyo3(constructor = (number, _u8=9))]
    BusDaysLagSettle { number: i32, _u8: u8  },
    /// A set number of calendar days enforcing settlement calendars, defined by a
    /// given calendar.
    #[pyo3(constructor = (number, _u8=10))]
    CalDaysLagSettle { number: i32, _u8: u8 },
}

impl From<Adjuster> for PyAdjuster {
    fn from(value: Adjuster) -> Self {
        match value {
            Adjuster::Actual {} => PyAdjuster::Actual { _u8: 0 },
            Adjuster::Following {} => PyAdjuster::Following { _u8: 1 },
            Adjuster::ModifiedFollowing {} => PyAdjuster::ModifiedFollowing { _u8: 2 },
            Adjuster::Previous {} => PyAdjuster::Previous { _u8: 3 },
            Adjuster::ModifiedPrevious {} => PyAdjuster::ModifiedPrevious { _u8: 4 },
            Adjuster::FollowingSettle {} => PyAdjuster::FollowingSettle { _u8: 5 },
            Adjuster::ModifiedFollowingSettle {} => PyAdjuster::ModifiedFollowingSettle { _u8: 6 },
            Adjuster::PreviousSettle {} => PyAdjuster::PreviousSettle { _u8: 7 },
            Adjuster::ModifiedPreviousSettle {} => PyAdjuster::ModifiedPreviousSettle { _u8: 8 },
            Adjuster::BusDaysLagSettle { number: n } => PyAdjuster::BusDaysLagSettle { number: n, _u8: 9 },
            Adjuster::CalDaysLagSettle { number: n } => PyAdjuster::CalDaysLagSettle { number: n, _u8: 10 },
        }
    }
}

impl From<PyAdjuster> for Adjuster {
    fn from(value: PyAdjuster) -> Self {
        match value {
            PyAdjuster::Actual { _u8: _ } => Adjuster::Actual {  },
            PyAdjuster::Following { _u8: _ } => Adjuster::Following {  },
            PyAdjuster::ModifiedFollowing { _u8: _ } => Adjuster::ModifiedFollowing {  },
            PyAdjuster::Previous { _u8: _ } => Adjuster::Previous { },
            PyAdjuster::ModifiedPrevious { _u8: _ } => Adjuster::ModifiedPrevious { },
            PyAdjuster::FollowingSettle { _u8: _ } => Adjuster::FollowingSettle { },
            PyAdjuster::ModifiedFollowingSettle { _u8: _ } => Adjuster::ModifiedFollowingSettle { },
            PyAdjuster::PreviousSettle { _u8: _ } => Adjuster::PreviousSettle { },
            PyAdjuster::ModifiedPreviousSettle { _u8: _ } => Adjuster::ModifiedPreviousSettle { },
            PyAdjuster::BusDaysLagSettle { number: n, _u8: _ } => Adjuster::BusDaysLagSettle { number: n },
            PyAdjuster::CalDaysLagSettle { number: n, _u8: _ } => Adjuster::CalDaysLagSettle { number: n },
        }
    }
}

#[pymethods]
impl PyAdjuster {
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
        let adjuster: Adjuster = (*self).into();
        adjuster.adjust(&date, &calendar)
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
        let adjuster: Adjuster = (*self).into();
        adjuster.adjusts(&dates, &calendar)
    }

    fn __str__(&self) -> String {
        match self {
            PyAdjuster::Actual { _u8:  _ } => "NONE".to_string(),
            PyAdjuster::Following { _u8:  _ } => "F".to_string(),
            PyAdjuster::Previous { _u8:  _ } => "P".to_string(),
            PyAdjuster::ModifiedFollowing { _u8:  _ } => "MF".to_string(),
            PyAdjuster::ModifiedPrevious { _u8:  _ } => "MP".to_string(),
            PyAdjuster::FollowingSettle { _u8:  _ } => "FSETTLE".to_string(),
            PyAdjuster::PreviousSettle { _u8:  _ } => "PSETTLE".to_string(),
            PyAdjuster::ModifiedFollowingSettle { _u8:  _ } => "MFSETTLE".to_string(),
            PyAdjuster::ModifiedPreviousSettle { _u8:  _ } => "FSETTLE".to_string(),
            PyAdjuster::BusDaysLagSettle { number: n, _u8: _ } => format!("{n}B"),
            PyAdjuster::CalDaysLagSettle { number: n, _u8: _ } => format!("{n}D"),
        }
    }
}

// #[pymethods]
// impl Adjuster {
//     /// Return a `date` under a date adjustment rule.
//     ///
//     /// Parameters
//     /// ----------
//     /// date: datetime
//     ///     Date to adjust.
//     /// calendar: Cal, UnionCal or NamedCal
//     ///     The calendar to assist with date adjustment.
//     ///
//     /// Returns
//     /// -------
//     /// datetime
//     #[pyo3(name = "adjust")]
//     fn adjust_py(&self, date: NaiveDateTime, calendar: Calendar) -> NaiveDateTime {
//         self.adjust(&date, &calendar)
//     }
// 
//     /// Return a vector of `dates` adjusted under a date adjustment rule.
//     ///
//     /// Parameters
//     /// ----------
//     /// dates: list[datetime]
//     ///     Dates to adjust.
//     /// calendar: Cal, UnionCal or NamedCal
//     ///     The calendar to assist with date adjustment.
//     ///
//     /// Returns
//     /// -------
//     /// list[datetime]
//     #[pyo3(name = "adjusts")]
//     fn adjusts_py(&self, dates: Vec<NaiveDateTime>, calendar: Calendar) -> Vec<NaiveDateTime> {
//         self.adjusts(&dates, &calendar)
//     }
// 
//     fn __str__(&self) -> String {
//         match self {
//             Adjuster::Actual {} => "NONE".to_string(),
//             Adjuster::Following {} => "F".to_string(),
//             Adjuster::Previous {} => "P".to_string(),
//             Adjuster::ModifiedFollowing {} => "MF".to_string(),
//             Adjuster::ModifiedPrevious {} => "MP".to_string(),
//             Adjuster::FollowingSettle {} => "FSETTLE".to_string(),
//             Adjuster::PreviousSettle {} => "PSETTLE".to_string(),
//             Adjuster::ModifiedFollowingSettle {} => "MFSETTLE".to_string(),
//             Adjuster::ModifiedPreviousSettle {} => "FSETTLE".to_string(),
//             Adjuster::BusDaysLagSettle { number: n } => format!("{n}B"),
//             Adjuster::CalDaysLagSettle { number: n } => format!("{n}D"),
//         }
//     }
// }

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
