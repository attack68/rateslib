use crate::scheduling::frequency::{Frequency, Scheduling};

use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl Frequency {
    /// Return the next unadjusted date under the schedule frequency.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     Any unchecked date, which may or may not align with the `Frequency`.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "next")]
    fn next_py(&self, date: NaiveDateTime) -> NaiveDateTime {
        self.next(&date)
    }

    /// Return the next unadjusted date under the schedule frequency.
    ///
    /// Parameters
    /// ----------
    /// udate: datetime
    ///     The unadjusted start date of the frequency period. If this is not a valid unadjusted
    ///     date aligned with the Frequency then it will raise.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "unext")]
    fn unext_py(&self, udate: NaiveDateTime) -> PyResult<NaiveDateTime> {
        self.try_unext(&udate)
    }

    /// Return the previous unadjusted date under the schedule frequency.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     Any unchecked date, which may or may not align with the `Frequency`.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "previous")]
    fn previous_py(&self, date: NaiveDateTime) -> NaiveDateTime {
        self.previous(&date)
    }

    /// Return the previous unadjusted date under the schedule frequency.
    ///
    /// Parameters
    /// ----------
    /// udate: datetime
    ///     The unadjusted end date of the frequency period. If this is not a valid unadjusted
    ///     date aligned with the Frequency then it will raise.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "uprevious")]
    fn uprevious_py(&self, udate: NaiveDateTime) -> PyResult<NaiveDateTime> {
        self.try_uprevious(&udate)
    }

    /// Return a list of unadjusted regular schedule dates.
    ///
    /// Parameters
    /// ----------
    /// ueffective: datetime
    ///     The unadjusted effective date of the schedule. If this is not a valid unadjusted
    ///     date aligned with the Frequency then it will raise.
    /// utermination: datetime
    ///     The unadjusted termination date of the frequency period. If this is not a valid
    ///     unadjusted date aligned with the Frequency then it will raise.
    ///
    /// Returns
    /// -------
    /// list[datetime]
    #[pyo3(name = "uregular")]
    fn uregular_py(
        &self,
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.try_uregular(&ueffective, &utermination)
    }

    /// Infer an unadjusted stub date from given schedule endpoints.
    ///
    /// Parameters
    /// ----------
    /// ueffective: datetime
    ///     The unadjusted effective date of the schedule.
    /// utermination: datetime
    ///     The unadjusted termination date of the frequency period. If this is not a valid
    ///     unadjusted date aligned with the Frequency then it will raise.
    /// short: bool
    ///     Whether to infer a short or a long stub.
    /// front: bool
    ///     Whether to infer a front or a back stub.
    ///
    /// Returns
    /// -------
    /// datetime or None
    ///
    /// Notes
    /// -----
    /// This function will return `None` if the dates define a regular schedule and no stub is
    /// required.
    #[pyo3(name = "infer_ustub")]
    fn infer_ustub_py(
        &self,
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        short: bool,
        front: bool,
    ) -> PyResult<Option<NaiveDateTime>> {
        if front {
            self.try_infer_ufront_stub(&ueffective, &utermination, short)
        } else {
            self.try_infer_uback_stub(&ueffective, &utermination, short)
        }
    }

    /// Check whether unadjusted dates define a stub period.
    ///
    /// Parameters
    /// ----------
    /// ustart: datetime
    ///     The unadjusted start date of the period.
    /// uend: datetime
    ///     The unadjusted end date of the period.
    /// front: bool
    ///     Test for either a front or a back stub.
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(name = "is_stub")]
    fn is_stub_py(&self, ustart: NaiveDateTime, uend: NaiveDateTime, front: bool) -> bool {
        if front {
            self.is_front_stub(&ustart, &uend)
        } else {
            self.is_back_stub(&ustart, &uend)
        }
    }

    /// Return a string representation of the Frequency.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "string")]
    fn string_py(&self) -> PyResult<String> {
        match self {
            Frequency::Zero {} => Ok("Z".to_string()),
            Frequency::CalDays { number: n } => Ok(format!("{n}D")),
            Frequency::BusDays {
                number: n,
                calendar: _,
            } => Ok(format!("{n}B")),
            Frequency::Months { number: 1, roll: _ } => Ok(format!("M")),
            Frequency::Months { number: 2, roll: _ } => Ok(format!("B")),
            Frequency::Months { number: 3, roll: _ } => Ok(format!("Q")),
            Frequency::Months { number: 4, roll: _ } => Ok(format!("T")),
            Frequency::Months { number: 6, roll: _ } => Ok(format!("S")),
            Frequency::Months {
                number: 12,
                roll: _,
            } => Ok(format!("A")),
            _ => Err(PyValueError::new_err(
                "No recognisable string represenation for Frequency.",
            )),
        }
    }

    fn __str__(&self) -> String {
        match self {
            Frequency::Zero {} => "Z".to_string(),
            Frequency::CalDays { number: n } => format!("{n}D"),
            Frequency::BusDays {
                number: n,
                calendar: _,
            } => format!("{n}B"),
            Frequency::Months { number: n, roll: r } => {
                let x = match r {
                    Some(v) => v.__str__(),
                    None => "none".to_string(),
                };
                format!("{n}M (roll: {x})")
            }
        }
    }
}
