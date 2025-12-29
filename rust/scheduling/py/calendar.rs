//! Wrapper module to export to Python using pyo3 bindings.

use crate::json::json_py::DeserializedObj;
use crate::json::JSON;
use crate::scheduling::py::adjuster::get_roll_adjuster_from_str;
use crate::scheduling::{
    Adjuster, Adjustment, Cal, Calendar, CalendarAdjustment, DateRoll, NamedCal, PyAdjuster,
    RollDay, UnionCal,
};
use chrono::NaiveDateTime;
use indexmap::set::IndexSet;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashSet;

#[pymethods]
impl Cal {
    /// Create a new *Cal* object.
    ///
    /// Parameters
    /// ----------
    /// holidays: list[datetime]
    ///     List of datetimes as the specific holiday days.
    /// week_mask: list[int],
    ///     List of integers defining the weekends, [5, 6] for Saturday and Sunday.
    #[new]
    fn new_py(holidays: Vec<NaiveDateTime>, week_mask: Vec<u8>) -> PyResult<Self> {
        Ok(Cal::new(holidays, week_mask))
    }

    /// Create a new *Cal* object from simple string name.
    /// Parameters
    /// ----------
    /// name: str
    ///     The 3-digit name of the calendar to load. Must be pre-defined in the Rust core code.
    ///
    /// Returns
    /// -------
    /// Cal
    #[classmethod]
    #[pyo3(name = "from_name")]
    fn from_name_py(_cls: &Bound<'_, PyType>, name: String) -> PyResult<Self> {
        Cal::try_from_name(&name)
    }

    /// A list of specifically provided non-business days.
    #[getter]
    fn holidays(&self) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.holidays.clone().into_iter().collect())
    }

    /// A list of days in the week defined as weekends.
    #[getter]
    fn week_mask(&self) -> PyResult<HashSet<u8>> {
        Ok(HashSet::from_iter(
            self.week_mask
                .clone()
                .into_iter()
                .map(|x| x.num_days_from_monday() as u8),
        ))
    }

    // #[getter]
    // fn rules(&self) -> PyResult<String> {
    //     Ok(self.meta.join(",\n"))
    // }

    /// Return whether the `date` is a business day.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     Date to test
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    /// Return whether the `date` is **not** a business day.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     Date to test
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    /// Return whether the `date` is a business day of an associated settlement calendar.
    ///
    /// .. note::
    ///
    ///    *Cal* objects will always return *True*, since they do not contain any
    ///    associated settlement calendars. This method is provided only for API consistency.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     Date to test
    ///
    /// Returns
    /// -------
    /// bool
    #[pyo3(name = "is_settlement")]
    fn is_settlement_py(&self, date: NaiveDateTime) -> bool {
        self.is_settlement(&date)
    }

    /// Return a date separated by calendar days from input date, and rolled with a modifier.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     The original business date. Raise if a non-business date is given.
    /// days: int
    ///     The number of calendar days to add.
    /// adjuster: Adjuster
    ///     The date adjustment rule to use on the unadjusted result.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "add_cal_days")]
    fn add_cal_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        adjuster: PyAdjuster,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_cal_days(&date, days, &adjuster.into()))
    }

    /// Return a business date separated by `days` from an input business `date`.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     The original business date. *Raises* if a non-business date is given.
    /// days: int
    ///     Number of business days to add.
    /// settlement: bool
    ///     Enforce an associated settlement calendar, if *True* and if one exists.
    ///
    /// Returns
    /// -------
    /// datetime
    ///
    /// Notes
    /// -----
    /// If adding negative number of business days a failing
    /// settlement will be rolled **backwards**, whilst adding a
    /// positive number of days will roll a failing settlement day **forwards**,
    /// if ``settlement`` is *True*.
    ///
    /// .. seealso::
    ///
    ///    :meth:`~rateslib.scheduling.Cal.lag_bus_days`: Add business days to inputs which are potentially
    ///    non-business dates.
    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        self.add_bus_days(&date, days, settlement)
    }

    /// Return a date separated by months from an input date, and rolled with a modifier.
    ///
    /// Parameters
    /// ----------
    /// date: datetime
    ///     The original date to adjust.
    /// months: int
    ///     The number of months to add.
    /// adjuster: Adjuster
    ///     The date adjustment rule to apply to the unadjusted result.
    /// roll: RollDay, optional
    ///     The day of the month to adjust to. If not given adopts the calendar day of ``date``.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "add_months")]
    fn add_months_py(
        &self,
        date: NaiveDateTime,
        months: i32,
        adjuster: PyAdjuster,
        roll: Option<RollDay>,
    ) -> NaiveDateTime {
        let roll_ = match roll {
            Some(val) => val,
            None => RollDay::vec_from(&vec![date])[0],
        };
        let adjuster: Adjuster = adjuster.into();
        adjuster.adjust(&roll_.uadd(&date, months), self)
    }

    /// Roll a date under a simplified adjustment rule.
    ///
    /// Parameters
    /// -----------
    /// date: datetime
    ///     The date to adjust.
    /// modifier: str in {"F", "P", "MF", "MP", "Act"}
    ///     The simplified date adjustment rule to apply
    /// settlement: bool
    ///     Whether to adhere to an additional settlement calendar.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "roll")]
    fn roll_py(
        &self,
        date: NaiveDateTime,
        modifier: &str,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        let adjuster = get_roll_adjuster_from_str((&modifier.to_lowercase(), settlement))?;
        Ok(self.adjust(&date, &adjuster))
    }

    /// Adjust a date under a date adjustment rule.
    ///
    /// Parameters
    /// -----------
    /// date: datetime
    ///     The date to adjust.
    /// adjuster: Adjuster
    ///     The date adjustment rule to apply.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "adjust")]
    fn adjust_py(&self, date: NaiveDateTime, adjuster: PyAdjuster) -> PyResult<NaiveDateTime> {
        Ok(self.adjust(&date, &adjuster.into()))
    }

    /// Adjust a list of dates under a date adjustment rule.
    ///
    /// Parameters
    /// -----------
    /// dates: list[datetime]
    ///     The dates to adjust.
    /// adjuster: Adjuster
    ///     The date adjustment rule to apply.
    ///
    /// Returns
    /// -------
    /// list[datetime]
    #[pyo3(name = "adjusts")]
    fn adjusts_py(
        &self,
        dates: Vec<NaiveDateTime>,
        adjuster: PyAdjuster,
    ) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.adjusts(&dates, &adjuster.into()))
    }

    /// Adjust a date by a number of business days, under lag rules.
    ///
    /// Parameters
    /// -----------
    /// date: datetime
    ///     The date to adjust.
    /// days: int
    ///     Number of business days to add.
    /// settlement: bool
    ///     Whether to enforce settlement against an associated settlement calendar.
    ///
    /// Returns
    /// --------
    /// datetime
    ///
    /// Notes
    /// -----
    /// ``lag_bus_days`` and ``add_bus_days`` will return the same value if the input date is a business
    /// date. If not a business date, ``add_bus_days`` will raise, while ``lag_bus_days`` will follow
    /// lag rules. ``lag_bus_days`` should be used when the input date cannot be guaranteed to be a
    /// business date.
    ///
    /// **Lag rules** define the addition of business days to a date that is a non-business date:
    ///
    /// - Adding zero days will roll the date **forwards** to the next available business day.
    /// - Adding one day will roll the date **forwards** to the next available business day.
    /// - Subtracting one day will roll the date **backwards** to the previous available business day.
    ///
    /// Adding (or subtracting) further business days adopts the
    /// :meth:`~rateslib.scheduling.Cal.add_bus_days` approach with a valid result.
    #[pyo3(name = "lag_bus_days")]
    fn lag_bus_days_py(&self, date: NaiveDateTime, days: i32, settlement: bool) -> NaiveDateTime {
        self.lag_bus_days(&date, days, settlement)
    }

    /// Return a list of business dates in a range.
    ///
    /// Parameters
    /// ----------
    /// start: datetime
    ///     The start date of the range, inclusive.
    /// end: datetime
    ///     The end date of the range, inclusive.
    ///
    /// Returns
    /// -------
    /// list[datetime]
    #[pyo3(name = "bus_date_range")]
    fn bus_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.bus_date_range(&start, &end)
    }

    /// Return a list of calendar dates within a range.
    ///
    /// Parameters
    /// -----------
    /// start: datetime
    ///     The start date of the range, inclusive.
    /// end: datetime
    ///     The end date of the range, inclusive,
    ///
    /// Returns
    /// --------
    /// list[datetime]
    #[pyo3(name = "cal_date_range")]
    fn cal_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.cal_date_range(&start, &end)
    }

    // Pickling
    fn __getnewargs__(&self) -> PyResult<(Vec<NaiveDateTime>, Vec<u8>)> {
        Ok((
            self.clone().holidays.into_iter().collect(),
            self.clone()
                .week_mask
                .into_iter()
                .map(|x| x.num_days_from_monday() as u8)
                .collect(),
        ))
    }

    // JSON
    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::Cal(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err("Failed to serialize `Cal` to JSON.")),
        }
    }

    // Equality
    fn __eq__(&self, other: Calendar) -> bool {
        match other {
            Calendar::UnionCal(c) => *self == c,
            Calendar::Cal(c) => *self == c,
            Calendar::NamedCal(c) => *self == c,
        }
    }

    fn __repr__(&self) -> String {
        format!("<rl.Cal at {:p}>", self)
    }
}

#[pymethods]
impl UnionCal {
    #[new]
    #[pyo3(signature = (calendars, settlement_calendars=None))]
    fn new_py(calendars: Vec<Cal>, settlement_calendars: Option<Vec<Cal>>) -> PyResult<Self> {
        Ok(UnionCal::new(calendars, settlement_calendars))
    }

    /// A list of specifically provided non-business days.
    #[getter]
    fn holidays(&self) -> PyResult<Vec<NaiveDateTime>> {
        let mut set = self.calendars.iter().fold(IndexSet::new(), |acc, x| {
            IndexSet::from_iter(acc.union(&x.holidays).cloned())
        });
        set.sort();
        Ok(Vec::from_iter(set))
    }

    /// A list of days in the week defined as weekends.
    #[getter]
    fn week_mask(&self) -> PyResult<HashSet<u8>> {
        let mut s: HashSet<u8> = HashSet::new();
        for cal in &self.calendars {
            let ns = cal.week_mask()?;
            s.extend(&ns);
        }
        Ok(s)
    }

    /// A list of :class:`~rateslib.scheduling.Cal` objects defining **business days**.
    #[getter]
    fn calendars(&self) -> Vec<Cal> {
        self.calendars.clone()
    }

    /// A list of :class:`~rateslib.scheduling.Cal` objects defining **settleable days**.
    #[getter]
    fn settlement_calendars(&self) -> Option<Vec<Cal>> {
        self.settlement_calendars.clone()
    }

    /// Return whether the `date` is a business day.
    ///
    /// See :meth:`Cal.is_bus_day <rateslib.scheduling.Cal.is_bus_day>`.
    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    /// Return whether the `date` is **not** a business day.
    ///
    /// See :meth:`Cal.is_non_bus_day <rateslib.scheduling.Cal.is_non_bus_day>`.
    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    /// Return whether the `date` is a business day in an associated settlement calendar.
    ///
    /// If no such associated settlement calendar exists this will return *True*.
    ///
    /// See :meth:`Cal.is_settlement <rateslib.scheduling.Cal.is_settlement>`.
    #[pyo3(name = "is_settlement")]
    fn is_settlement_py(&self, date: NaiveDateTime) -> bool {
        self.is_settlement(&date)
    }

    /// Return a date separated by calendar days from input date, and rolled with a modifier.
    ///
    /// See :meth:`Cal.add_cal_days <rateslib.scheduling.Cal.add_cal_days>`.
    #[pyo3(name = "add_cal_days")]
    fn add_cal_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        adjuster: PyAdjuster,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_cal_days(&date, days, &adjuster.into()))
    }

    /// Return a business date separated by `days` from an input business `date`.
    ///
    /// See :meth:`Cal.add_bus_days <rateslib.scheduling.Cal.add_bus_days>`.
    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        self.add_bus_days(&date, days, settlement)
    }

    /// Return a date separated by months from an input date, and rolled with a modifier.
    ///
    /// See :meth:`Cal.add_months <rateslib.scheduling.Cal.add_months>`.
    #[pyo3(name = "add_months")]
    fn add_months_py(
        &self,
        date: NaiveDateTime,
        months: i32,
        adjuster: PyAdjuster,
        roll: Option<RollDay>,
    ) -> NaiveDateTime {
        let roll_ = match roll {
            Some(val) => val,
            None => RollDay::vec_from(&vec![date])[0],
        };
        let adjuster: Adjuster = adjuster.into();
        adjuster.adjust(&roll_.uadd(&date, months), self)
    }

    /// Adjust a non-business date to a business date under a specific modification rule.
    ///
    /// See :meth:`Cal.adjust <rateslib.scheduling.Cal.adjust>`.
    #[pyo3(name = "adjust")]
    fn adjust_py(&self, date: NaiveDateTime, adjuster: PyAdjuster) -> PyResult<NaiveDateTime> {
        Ok(self.adjust(&date, &adjuster.into()))
    }

    /// Adjust a list of dates under a date adjustment rule.
    ///
    /// See :meth:`Cal.adjusts <rateslib.scheduling.Cal.adjusts>`.
    #[pyo3(name = "adjusts")]
    fn adjusts_py(
        &self,
        dates: Vec<NaiveDateTime>,
        adjuster: PyAdjuster,
    ) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.adjusts(&dates, &adjuster.into()))
    }

    /// Roll a date under a simplified adjustment rule.
    ///
    /// See :meth:`Cal.roll <rateslib.scheduling.Cal.roll>`.
    #[pyo3(name = "roll")]
    fn roll_py(
        &self,
        date: NaiveDateTime,
        modifier: &str,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        let adjuster = get_roll_adjuster_from_str((&modifier.to_lowercase(), settlement))?;
        Ok(self.adjust(&date, &adjuster))
    }

    /// Adjust a date by a number of business days, under lag rules.
    ///
    /// See :meth:`Cal.lag_bus_days <rateslib.scheduling.Cal.lag_bus_days>`.
    #[pyo3(name = "lag_bus_days")]
    fn lag_bus_days_py(&self, date: NaiveDateTime, days: i32, settlement: bool) -> NaiveDateTime {
        self.lag_bus_days(&date, days, settlement)
    }

    /// Return a list of business dates in a range.
    ///
    /// See :meth:`Cal.bus_date_range <rateslib.scheduling.Cal.bus_date_range>`.
    #[pyo3(name = "bus_date_range")]
    fn bus_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.bus_date_range(&start, &end)
    }

    /// Return a list of calendar dates in a range.
    ///
    /// See :meth:`Cal.cal_date_range <rateslib.scheduling.Cal.cal_date_range>`.
    #[pyo3(name = "cal_date_range")]
    fn cal_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.cal_date_range(&start, &end)
    }

    // Pickling
    fn __getnewargs__(&self) -> PyResult<(Vec<Cal>, Option<Vec<Cal>>)> {
        Ok((self.calendars.clone(), self.settlement_calendars.clone()))
    }

    // JSON
    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::UnionCal(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `UnionCal` to JSON.",
            )),
        }
    }

    // Equality
    fn __eq__(&self, other: Calendar) -> bool {
        match other {
            Calendar::UnionCal(c) => *self == c,
            Calendar::Cal(c) => *self == c,
            Calendar::NamedCal(c) => *self == c,
        }
    }

    fn __repr__(&self) -> String {
        format!("<rl.UnionCal at {:p}>", self)
    }
}

#[pymethods]
impl NamedCal {
    #[new]
    fn new_py(name: String) -> PyResult<Self> {
        NamedCal::try_new(&name)
    }

    /// A list of specifically provided non-business days.
    #[getter]
    fn holidays(&self) -> PyResult<Vec<NaiveDateTime>> {
        self.union_cal.holidays()
    }

    /// A list of days in the week defined as weekends.
    #[getter]
    fn week_mask(&self) -> PyResult<HashSet<u8>> {
        self.union_cal.week_mask()
    }

    /// The string identifier for this constructed calendar.
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// The wrapped :class:`~rateslib.scheduling.UnionCal` object.
    #[getter]
    fn union_cal(&self) -> UnionCal {
        self.union_cal.clone()
    }

    /// Return whether the `date` is a business day.
    ///
    /// See :meth:`Cal.is_bus_day <rateslib.scheduling.Cal.is_bus_day>`.
    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    /// Return whether the `date` is **not** a business day.
    ///
    /// See :meth:`Cal.is_non_bus_day <rateslib.scheduling.Cal.is_non_bus_day>`.
    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    /// Return whether the `date` is a business day in an associated settlement calendar.
    ///
    /// If no such associated settlement calendar exists this will return *True*.
    ///
    /// See :meth:`Cal.is_settlement <rateslib.scheduling.Cal.is_settlement>`.
    #[pyo3(name = "is_settlement")]
    fn is_settlement_py(&self, date: NaiveDateTime) -> bool {
        self.is_settlement(&date)
    }

    /// Return a date separated by calendar days from input date, and rolled with a modifier.
    ///
    /// See :meth:`Cal.add_cal_days <rateslib.scheduling.Cal.add_cal_days>`.
    #[pyo3(name = "add_cal_days")]
    fn add_cal_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        adjuster: PyAdjuster,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_cal_days(&date, days, &adjuster.into()))
    }

    /// Return a business date separated by `days` from an input business `date`.
    ///
    /// See :meth:`Cal.add_bus_days <rateslib.scheduling.Cal.add_bus_days>`.
    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        self.add_bus_days(&date, days, settlement)
    }

    /// Return a date separated by months from an input date, and rolled with a modifier.
    ///
    /// See :meth:`Cal.add_months <rateslib.scheduling.Cal.add_months>`.
    #[pyo3(name = "add_months")]
    fn add_months_py(
        &self,
        date: NaiveDateTime,
        months: i32,
        adjuster: PyAdjuster,
        roll: Option<RollDay>,
    ) -> NaiveDateTime {
        let roll_ = match roll {
            Some(val) => val,
            None => RollDay::vec_from(&vec![date])[0],
        };
        let adjuster: Adjuster = adjuster.into();
        adjuster.adjust(&roll_.uadd(&date, months), self)
    }

    /// Adjust a non-business date to a business date under a specific modification rule.
    ///
    /// See :meth:`Cal.adjust <rateslib.scheduling.Cal.adjust>`.
    #[pyo3(name = "adjust")]
    fn adjust_py(&self, date: NaiveDateTime, adjuster: PyAdjuster) -> PyResult<NaiveDateTime> {
        Ok(self.adjust(&date, &adjuster.into()))
    }

    /// Adjust a list of dates under a date adjustment rule.
    ///
    /// See :meth:`Cal.adjusts <rateslib.scheduling.Cal.adjusts>`.
    #[pyo3(name = "adjusts")]
    fn adjusts_py(
        &self,
        dates: Vec<NaiveDateTime>,
        adjuster: PyAdjuster,
    ) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.adjusts(&dates, &adjuster.into()))
    }

    /// Roll a date under a simplified adjustment rule.
    ///
    /// See :meth:`Cal.roll <rateslib.scheduling.Cal.roll>`.
    #[pyo3(name = "roll")]
    fn roll_py(
        &self,
        date: NaiveDateTime,
        modifier: &str,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        let adjuster = get_roll_adjuster_from_str((&modifier.to_lowercase(), settlement))?;
        Ok(self.adjust(&date, &adjuster))
    }

    /// Adjust a date by a number of business days, under lag rules.
    ///
    /// See :meth:`Cal.lag_bus_days <rateslib.scheduling.Cal.lag_bus_days>`.
    #[pyo3(name = "lag_bus_days")]
    fn lag_bus_days_py(&self, date: NaiveDateTime, days: i32, settlement: bool) -> NaiveDateTime {
        self.lag_bus_days(&date, days, settlement)
    }

    /// Return a list of business dates in a range.
    ///
    /// See :meth:`Cal.bus_date_range <rateslib.scheduling.Cal.bus_date_range>`.
    #[pyo3(name = "bus_date_range")]
    fn bus_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.bus_date_range(&start, &end)
    }

    /// Return a list of calendar dates in a range.
    ///
    /// See :meth:`Cal.cal_date_range <rateslib.scheduling.Cal.cal_date_range>`.
    #[pyo3(name = "cal_date_range")]
    fn cal_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.cal_date_range(&start, &end)
    }

    // Pickling
    fn __getnewargs__(&self) -> PyResult<(String,)> {
        Ok((self.name.clone(),))
    }

    // JSON
    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::NamedCal(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `NamedCal` to JSON.",
            )),
        }
    }

    // Equality
    fn __eq__(&self, other: Calendar) -> bool {
        match other {
            Calendar::UnionCal(c) => *self == c,
            Calendar::Cal(c) => *self == c,
            Calendar::NamedCal(c) => *self == c,
        }
    }

    fn __repr__(&self) -> String {
        format!("<rl.NamedCal:'{}' at {:p}>", self.name, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::ndt;

    #[test]
    fn test_add_37_months() {
        let cal = Cal::try_from_name("all").unwrap();

        let dates = vec![
            (ndt(2000, 1, 1), ndt(2003, 2, 1)),
            (ndt(2000, 2, 1), ndt(2003, 3, 1)),
            (ndt(2000, 3, 1), ndt(2003, 4, 1)),
            (ndt(2000, 4, 1), ndt(2003, 5, 1)),
            (ndt(2000, 5, 1), ndt(2003, 6, 1)),
            (ndt(2000, 6, 1), ndt(2003, 7, 1)),
            (ndt(2000, 7, 1), ndt(2003, 8, 1)),
            (ndt(2000, 8, 1), ndt(2003, 9, 1)),
            (ndt(2000, 9, 1), ndt(2003, 10, 1)),
            (ndt(2000, 10, 1), ndt(2003, 11, 1)),
            (ndt(2000, 11, 1), ndt(2003, 12, 1)),
            (ndt(2000, 12, 1), ndt(2004, 1, 1)),
        ];
        for i in 0..12 {
            assert_eq!(
                cal.add_months_py(
                    dates[i].0,
                    37,
                    Adjuster::FollowingSettle {}.into(),
                    Some(RollDay::Day(1)),
                ),
                dates[i].1
            )
        }
    }

    #[test]
    fn test_sub_37_months() {
        let cal = Cal::try_from_name("all").unwrap();

        let dates = vec![
            (ndt(2000, 1, 1), ndt(1996, 12, 1)),
            (ndt(2000, 2, 1), ndt(1997, 1, 1)),
            (ndt(2000, 3, 1), ndt(1997, 2, 1)),
            (ndt(2000, 4, 1), ndt(1997, 3, 1)),
            (ndt(2000, 5, 1), ndt(1997, 4, 1)),
            (ndt(2000, 6, 1), ndt(1997, 5, 1)),
            (ndt(2000, 7, 1), ndt(1997, 6, 1)),
            (ndt(2000, 8, 1), ndt(1997, 7, 1)),
            (ndt(2000, 9, 1), ndt(1997, 8, 1)),
            (ndt(2000, 10, 1), ndt(1997, 9, 1)),
            (ndt(2000, 11, 1), ndt(1997, 10, 1)),
            (ndt(2000, 12, 1), ndt(1997, 11, 1)),
        ];
        for i in 0..12 {
            assert_eq!(
                cal.add_months_py(
                    dates[i].0,
                    -37,
                    Adjuster::FollowingSettle {}.into(),
                    Some(RollDay::Day(1)),
                ),
                dates[i].1
            )
        }
    }

    #[test]
    fn test_add_months_py_roll() {
        let cal = Cal::try_from_name("all").unwrap();
        let roll = vec![
            (RollDay::Day(7), ndt(1998, 3, 7), ndt(1996, 12, 7)),
            (RollDay::Day(21), ndt(1998, 3, 21), ndt(1996, 12, 21)),
            (RollDay::Day(31), ndt(1998, 3, 31), ndt(1996, 12, 31)),
            (RollDay::Day(1), ndt(1998, 3, 1), ndt(1996, 12, 1)),
            (RollDay::IMM(), ndt(1998, 3, 18), ndt(1996, 12, 18)),
        ];
        for i in 0..5 {
            assert_eq!(
                cal.add_months_py(
                    roll[i].1,
                    -15,
                    Adjuster::FollowingSettle {}.into(),
                    Some(roll[i].0)
                ),
                roll[i].2
            );
        }
    }

    #[test]
    fn test_add_months_roll_invalid_days() {
        let cal = Cal::try_from_name("all").unwrap();
        let roll = vec![
            (RollDay::Day(21), ndt(1996, 12, 21)),
            (RollDay::Day(31), ndt(1996, 12, 31)),
            (RollDay::Day(1), ndt(1996, 12, 1)),
            (RollDay::IMM(), ndt(1996, 12, 18)),
        ];
        for i in 0..4 {
            assert_eq!(
                roll[i].1,
                cal.add_months_py(
                    ndt(1998, 3, 7),
                    -15,
                    Adjuster::FollowingSettle {}.into(),
                    Some(roll[i].0),
                ),
            );
        }
    }

    #[test]
    fn test_add_months_modifier() {
        let cal = Cal::try_from_name("bus").unwrap();
        let modi = vec![
            (Adjuster::Actual {}, ndt(2023, 9, 30)),          // Saturday
            (Adjuster::FollowingSettle {}, ndt(2023, 10, 2)), // Monday
            (Adjuster::ModifiedFollowingSettle {}, ndt(2023, 9, 29)), // Friday
            (Adjuster::PreviousSettle {}, ndt(2023, 9, 29)),  // Friday
            (Adjuster::ModifiedPreviousSettle {}, ndt(2023, 9, 29)), // Friday
        ];
        for i in 0..4 {
            assert_eq!(
                cal.add_months_py(
                    ndt(2023, 8, 31),
                    1,
                    modi[i].0.into(),
                    Some(RollDay::Day(31))
                ),
                modi[i].1
            );
        }
    }

    #[test]
    fn test_add_months_modifier_p() {
        let cal = Cal::try_from_name("bus").unwrap();
        let modi = vec![
            (Adjuster::Actual {}, ndt(2023, 7, 1)),          // Saturday
            (Adjuster::FollowingSettle {}, ndt(2023, 7, 3)), // Monday
            (Adjuster::ModifiedFollowingSettle {}, ndt(2023, 7, 3)), // Monday
            (Adjuster::PreviousSettle {}, ndt(2023, 6, 30)), // Friday
            (Adjuster::ModifiedPreviousSettle {}, ndt(2023, 7, 3)), // Monday
        ];
        for i in 0..4 {
            assert_eq!(
                cal.add_months_py(ndt(2023, 8, 1), -1, modi[i].0.into(), Some(RollDay::Day(1))),
                modi[i].1
            );
        }
    }
}
