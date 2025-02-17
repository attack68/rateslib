
use crate::calendars::{
    CalType,
    RollDay,
    Modifier
};
use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};

/// A schedule frequency.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone)]
pub enum Frequency {
    /// Periods every month.
    Monthly,
    /// Periods every two months.
    BiMonthly,
    /// Periods every three months.
    Quarterly,
    /// Periods every four months.
    TriAnnually,
    /// Periods every six months.
    SemiAnnually,
    /// Periods every twelve months.
    Annually,
    /// Only every a single period.
    Zero,
}

/// A stub type indicator for date inference.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone)]
pub enum Stub {
    /// Short front stub inference.
    ShortFront
    /// Long front stub inference.
    LongFront
    /// Short back stub inference.
    ShortBack
    /// Long back stub inference.
    LongBack
}

pub struct Schedule {
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    frequency: Frequency,
    front_stub: Option<NaiveDateTime>,
    back_stub: Option<NaiveDateTime>
    roll: RollDay,
    modifier: Modifier,
    calendar: CalType,
    payment_lag: i8,

    // created data objects
    uschedule: Vec<NaiveDateTime>,
    aschedule: Vec<NaiveDateTime>,
    pschedule: Vec<NaiveDateTime>
}

impl Schedule {
    pub fn try_new(
        effective: NaiveDateTime,
        termination: NaiveDateTime,
        frequency: Frequency,
        stub: Stub,
        front_stub: Option<NaiveDateTime>,
        back_stub: Option<NaiveDateTime>,
        roll: Option<RollDay>,
        eom: bool,
        modifier: Modifier,
        calendar: CalType,
        payment_lag: i8,
    ) -> Result<Self, PyErr> {

    }

}