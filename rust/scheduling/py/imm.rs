use crate::json::{DeserializedObj, JSON};
use crate::scheduling::frequency::Imm;

use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
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

    // JSON
    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::Imm(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err("Failed to serialize `Imm` to JSON.")),
        }
    }

    // Pickling
    #[new]
    fn new_py(item: usize) -> Imm {
        match item {
            _ if item == Imm::Wed3 as usize => Imm::Wed3,
            _ if item == Imm::Wed3_HMUZ as usize => Imm::Wed3_HMUZ,
            _ if item == Imm::Fri2 as usize => Imm::Fri2,
            _ if item == Imm::Fri2_HMUZ as usize => Imm::Fri2_HMUZ,
            _ if item == Imm::Day20 as usize => Imm::Day20,
            _ if item == Imm::Day20_HU as usize => Imm::Day20_HU,
            _ if item == Imm::Day20_MZ as usize => Imm::Day20_MZ,
            _ if item == Imm::Day20_HMUZ as usize => Imm::Day20_HMUZ,
            _ if item == Imm::Wed1_Post9 as usize => Imm::Wed1_Post9,
            _ if item == Imm::Wed1_Post9_HMUZ as usize => Imm::Wed1_Post9_HMUZ,
            _ if item == Imm::Eom as usize => Imm::Eom,
            _ if item == Imm::Leap as usize => Imm::Leap,
            _ => panic!("Reportable issue: must map this enum variant for serialization."),
        }
    }
    fn __getnewargs__<'py>(&self) -> PyResult<(usize,)> {
        Ok((*self as usize,))
    }

    fn __repr__(&self) -> String {
        format!("<rl.Imm.{:?} at {:p}>", self, self)
    }
}
