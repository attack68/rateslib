use crate::scheduling::frequency::{Frequency, Scheduling};

use chrono::prelude::*;
use pyo3::prelude::*;

#[pymethods]
impl Frequency {
    /// Return the next unadjusted date under the schedule frequency.
    ///
    /// Parameters
    /// ----------
    /// ueffective: datetime
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
    /// ueffective: datetime
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

    /// Infer an unadjusted front stub date from given schedule endpoints.
    ///
    /// Parameters
    /// ----------
    /// ueffective: datetime
    ///     The unadjusted effective date of the schedule.
    /// utermination: datetime
    ///     The unadjusted termination date of the frequency period. If this is not a valid
    ///     unadjusted date aligned with the Frequency then it will raise.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "infer_ufront_stub")]
    fn infer_ufront_stub_py(
        &self,
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        short: bool,
    ) -> PyResult<NaiveDateTime> {
        self.try_infer_ufront_stub(&ueffective, &utermination, short)
    }

    /// Infer an unadjusted back stub date from given schedule endpoints.
    ///
    /// Parameters
    /// ----------
    /// ueffective: datetime
    ///     The unadjusted effective date of the schedule. If this is not a valid
    ///     unadjusted date aligned with the Frequency then it will raise.
    /// utermination: datetime
    ///     The unadjusted termination date of the frequency period.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "infer_uback_stub")]
    fn infer_uback_stub_py(
        &self,
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        short: bool,
    ) -> PyResult<NaiveDateTime> {
        self.try_infer_uback_stub(&ueffective, &utermination, short)
    }
}
