//! Wrapper module to export to Python using pyo3 bindings.

use crate::json::{DeserializedObj, JSON};
use crate::scheduling::{Adjuster, Adjustment, Calendar};
use chrono::NaiveDateTime;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Python wrapper for Adjuster to facilitate complex enum pickling.
#[pyclass(module = "rateslib.rs", name = "Adjuster", eq)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum PyAdjuster {
    #[pyo3(constructor = (_u8=0))]
    Actual { _u8: u8 },
    #[pyo3(constructor = (_u8=1))]
    Following { _u8: u8 },
    #[pyo3(constructor = (_u8=2))]
    ModifiedFollowing { _u8: u8 },
    #[pyo3(constructor = (_u8=3))]
    Previous { _u8: u8 },
    #[pyo3(constructor = (_u8=4))]
    ModifiedPrevious { _u8: u8 },
    #[pyo3(constructor = (_u8=5))]
    FollowingSettle { _u8: u8 },
    #[pyo3(constructor = (_u8=6))]
    ModifiedFollowingSettle { _u8: u8 },
    #[pyo3(constructor = (_u8=7))]
    PreviousSettle { _u8: u8 },
    #[pyo3(constructor = (_u8=8))]
    ModifiedPreviousSettle { _u8: u8 },
    #[pyo3(constructor = (number, _u8=9))]
    BusDaysLagSettle { number: i32, _u8: u8 },
    #[pyo3(constructor = (number, _u8=10))]
    CalDaysLagSettle { number: i32, _u8: u8 },
    #[pyo3(constructor = (_u8=11))]
    FollowingExLast { _u8: u8 },
    #[pyo3(constructor = (_u8=12))]
    FollowingExLastSettle { _u8: u8 },
    #[pyo3(constructor = (number, _u8=13))]
    BusDaysLagSettleInAdvance { number: i32, _u8: u8 },
}

/// Used for providing pickle support for PyAdjuster
enum PyAdjusterNewArgs {
    NoArgs(u8),
    I32(i32, u8),
}

impl<'py> IntoPyObject<'py> for PyAdjusterNewArgs {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            PyAdjusterNewArgs::NoArgs(x) => Ok((x,).into_pyobject(py).unwrap()),
            PyAdjusterNewArgs::I32(x, y) => Ok((x, y).into_pyobject(py).unwrap()),
        }
    }
}

impl<'py> FromPyObject<'py, 'py> for PyAdjusterNewArgs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let ext: PyResult<(u8,)> = obj.extract();
        if ext.is_ok() {
            let (x,) = ext.unwrap();
            return Ok(PyAdjusterNewArgs::NoArgs(x));
        }
        let ext: PyResult<(i32, u8)> = obj.extract();
        if ext.is_ok() {
            let (x, y) = ext.unwrap();
            return Ok(PyAdjusterNewArgs::I32(x, y));
        }
        Err(PyValueError::new_err("Undefined behaviour"))
    }
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
            Adjuster::BusDaysLagSettle(n) => PyAdjuster::BusDaysLagSettle { number: n, _u8: 9 },
            Adjuster::CalDaysLagSettle(n) => PyAdjuster::CalDaysLagSettle { number: n, _u8: 10 },
            Adjuster::FollowingExLast {} => PyAdjuster::FollowingExLast { _u8: 11 },
            Adjuster::FollowingExLastSettle {} => PyAdjuster::FollowingExLastSettle { _u8: 12 },
            Adjuster::BusDaysLagSettleInAdvance(n) => {
                PyAdjuster::BusDaysLagSettleInAdvance { number: n, _u8: 13 }
            }
        }
    }
}

impl From<PyAdjuster> for Adjuster {
    fn from(value: PyAdjuster) -> Self {
        match value {
            PyAdjuster::Actual { _u8: _ } => Adjuster::Actual {},
            PyAdjuster::Following { _u8: _ } => Adjuster::Following {},
            PyAdjuster::ModifiedFollowing { _u8: _ } => Adjuster::ModifiedFollowing {},
            PyAdjuster::Previous { _u8: _ } => Adjuster::Previous {},
            PyAdjuster::ModifiedPrevious { _u8: _ } => Adjuster::ModifiedPrevious {},
            PyAdjuster::FollowingSettle { _u8: _ } => Adjuster::FollowingSettle {},
            PyAdjuster::ModifiedFollowingSettle { _u8: _ } => Adjuster::ModifiedFollowingSettle {},
            PyAdjuster::PreviousSettle { _u8: _ } => Adjuster::PreviousSettle {},
            PyAdjuster::ModifiedPreviousSettle { _u8: _ } => Adjuster::ModifiedPreviousSettle {},
            PyAdjuster::BusDaysLagSettle { number: n, _u8: _ } => Adjuster::BusDaysLagSettle(n),
            PyAdjuster::CalDaysLagSettle { number: n, _u8: _ } => Adjuster::CalDaysLagSettle(n),
            PyAdjuster::FollowingExLast { _u8: _ } => Adjuster::FollowingExLast {},
            PyAdjuster::FollowingExLastSettle { _u8: _ } => Adjuster::FollowingExLastSettle {},
            PyAdjuster::BusDaysLagSettleInAdvance { number: n, _u8: _ } => {
                Adjuster::BusDaysLagSettleInAdvance(n)
            }
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

    /// Return a list of `dates` which result in ``date`` when the adjustment is applied.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     Date to reverse to detect possible unadjusted dates.
    /// calendar: Cal, UnionCal or NamedCal
    ///     The calendar to assist with date adjustment.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "reverse")]
    fn reverse_py(&self, date: NaiveDateTime, calendar: Calendar) -> Vec<NaiveDateTime> {
        let adjuster: Adjuster = (*self).into();
        adjuster.reverse(&date, &calendar)
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
            PyAdjuster::Actual { _u8: _ } => "NONE".to_string(),
            PyAdjuster::Following { _u8: _ } => "F".to_string(),
            PyAdjuster::Previous { _u8: _ } => "P".to_string(),
            PyAdjuster::ModifiedFollowing { _u8: _ } => "MF".to_string(),
            PyAdjuster::ModifiedPrevious { _u8: _ } => "MP".to_string(),
            PyAdjuster::FollowingSettle { _u8: _ } => "FSETTLE".to_string(),
            PyAdjuster::PreviousSettle { _u8: _ } => "PSETTLE".to_string(),
            PyAdjuster::ModifiedFollowingSettle { _u8: _ } => "MFSETTLE".to_string(),
            PyAdjuster::ModifiedPreviousSettle { _u8: _ } => "FSETTLE".to_string(),
            PyAdjuster::BusDaysLagSettle { number: n, _u8: _ } => format!("{n}B"),
            PyAdjuster::CalDaysLagSettle { number: n, _u8: _ } => format!("{n}D"),
            PyAdjuster::FollowingExLast { _u8: _ } => format!("FEX"),
            PyAdjuster::FollowingExLastSettle { _u8: _ } => format!("FEXSETTLE"),
            PyAdjuster::BusDaysLagSettleInAdvance { number: n, _u8: _ } => format!("IA{n}B"),
        }
    }

    fn __getnewargs__(&self) -> PyAdjusterNewArgs {
        match self {
            PyAdjuster::Actual { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::Following { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::Previous { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::ModifiedFollowing { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::ModifiedPrevious { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::FollowingSettle { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::PreviousSettle { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::ModifiedFollowingSettle { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::ModifiedPreviousSettle { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::BusDaysLagSettle { number: n, _u8: u } => PyAdjusterNewArgs::I32(*n, *u),
            PyAdjuster::CalDaysLagSettle { number: n, _u8: u } => PyAdjusterNewArgs::I32(*n, *u),
            PyAdjuster::FollowingExLast { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::FollowingExLastSettle { _u8: u } => PyAdjusterNewArgs::NoArgs(*u),
            PyAdjuster::BusDaysLagSettleInAdvance { number: n, _u8: u } => {
                PyAdjusterNewArgs::I32(*n, *u)
            }
        }
    }

    #[new]
    fn new_py(args: PyAdjusterNewArgs) -> PyAdjuster {
        match args {
            PyAdjusterNewArgs::NoArgs(0) => PyAdjuster::Actual { _u8: 0 },
            PyAdjusterNewArgs::NoArgs(1) => PyAdjuster::Following { _u8: 1 },
            PyAdjusterNewArgs::NoArgs(2) => PyAdjuster::Previous { _u8: 2 },
            PyAdjusterNewArgs::NoArgs(3) => PyAdjuster::ModifiedFollowing { _u8: 3 },
            PyAdjusterNewArgs::NoArgs(4) => PyAdjuster::ModifiedPrevious { _u8: 4 },
            PyAdjusterNewArgs::NoArgs(5) => PyAdjuster::FollowingSettle { _u8: 5 },
            PyAdjusterNewArgs::NoArgs(6) => PyAdjuster::PreviousSettle { _u8: 6 },
            PyAdjusterNewArgs::NoArgs(7) => PyAdjuster::ModifiedFollowingSettle { _u8: 7 },
            PyAdjusterNewArgs::NoArgs(8) => PyAdjuster::ModifiedPreviousSettle { _u8: 8 },
            PyAdjusterNewArgs::I32(n, 9) => PyAdjuster::BusDaysLagSettle { number: n, _u8: 9 },
            PyAdjusterNewArgs::I32(n, 10) => PyAdjuster::CalDaysLagSettle { number: n, _u8: 10 },
            PyAdjusterNewArgs::NoArgs(11) => PyAdjuster::FollowingExLast { _u8: 11 },
            PyAdjusterNewArgs::NoArgs(12) => PyAdjuster::FollowingExLastSettle { _u8: 12 },
            PyAdjusterNewArgs::I32(n, 13) => {
                PyAdjuster::BusDaysLagSettleInAdvance { number: n, _u8: 13 }
            }
            _ => panic!("Undefined behaviour."),
        }
    }

    fn __repr__(&self) -> String {
        let adjuster: Adjuster = (*self).into();
        format!("<rl.Adjuster.{:?} at {:p}>", adjuster, self)
    }

    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::PyAdjuster(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `Adjuster` to JSON.",
            )),
        }
    }
}

/// This function appears to be unused.
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
        (("fex", false), Adjuster::FollowingExLast {}),
        (("fex", true), Adjuster::FollowingExLastSettle {}),
    ]);
    match hmap.get(&input) {
        None => Err(PyValueError::new_err(format!(
            "'{}', with '{}' settlement, is not found in the list of allowed roll adjusters.",
            input.0, input.1
        ))),
        Some(value) => Ok(*value),
    }
}
