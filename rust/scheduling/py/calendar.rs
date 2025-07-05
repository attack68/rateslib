//! Wrapper module to export to Python using pyo3 bindings.

use crate::json::json_py::DeserializedObj;
use crate::json::JSON;
use crate::scheduling::py::adjuster::get_roll_adjuster_from_str;
use crate::scheduling::{
    get_calendar_by_name, Adjuster, Adjustment, Cal, Calendar, CalendarAdjustment, Convention,
    DateRoll, NamedCal, RollDay, UnionCal,
};
use bincode::config::legacy;
use bincode::serde::{decode_from_slice, encode_to_vec};
use chrono::NaiveDateTime;
use indexmap::set::IndexSet;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashSet;

// // removed when upgrading to py03 0.23, see https://pyo3.rs/v0.23.0/migration#intopyobject-and-intopyobjectref-derive-macros
// impl IntoPy<PyObject> for Calendar {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         macro_rules! into_py {
//             ($obj: ident) => {
//                 Py::new(py, $obj).unwrap().to_object(py)
//             };
//         }
//
//         match self {
//             Calendar::Cal(i) => into_py!(i),
//             Calendar::UnionCal(i) => into_py!(i),
//             Calendar::NamedCal(i) => into_py!(i),
//         }
//     }
// }

#[pymethods]
impl Convention {
    // Pickling
    #[new]
    fn new_py(ad: u8) -> PyResult<Convention> {
        match ad {
            0_u8 => Ok(Convention::One),
            1_u8 => Ok(Convention::OnePlus),
            2_u8 => Ok(Convention::Act365F),
            3_u8 => Ok(Convention::Act365FPlus),
            4_u8 => Ok(Convention::Act360),
            5_u8 => Ok(Convention::ThirtyE360),
            6_u8 => Ok(Convention::Thirty360),
            7_u8 => Ok(Convention::Thirty360ISDA),
            8_u8 => Ok(Convention::ActActISDA),
            9_u8 => Ok(Convention::ActActICMA),
            10_u8 => Ok(Convention::Bus252),
            _ => Err(PyValueError::new_err(
                "unreachable code on Convention pickle.",
            )),
        }
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = decode_from_slice(state.as_bytes(), legacy()).unwrap().0;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &encode_to_vec(&self, legacy()).unwrap()))
    }
    pub fn __getnewargs__<'py>(&self) -> PyResult<(u8,)> {
        match self {
            Convention::One => Ok((0_u8,)),
            Convention::OnePlus => Ok((1_u8,)),
            Convention::Act365F => Ok((2_u8,)),
            Convention::Act365FPlus => Ok((3_u8,)),
            Convention::Act360 => Ok((4_u8,)),
            Convention::ThirtyE360 => Ok((5_u8,)),
            Convention::Thirty360 => Ok((6_u8,)),
            Convention::Thirty360ISDA => Ok((7_u8,)),
            Convention::ActActISDA => Ok((8_u8,)),
            Convention::ActActICMA => Ok((9_u8,)),
            Convention::Bus252 => Ok((10_u8,)),
        }
    }
}

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

    #[getter]
    fn holidays(&self) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.holidays.clone().into_iter().collect())
    }

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
        adjuster: Adjuster,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_cal_days(&date, days, &adjuster))
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
    ///    :meth:`~rateslib.calendars.Cal.lag_bus_days`: Add business days to inputs which are potentially
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
    /// roll: RollDay
    ///     The day of the month to adjust to.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "add_months")]
    fn add_months_py(
        &self,
        date: NaiveDateTime,
        months: i32,
        adjuster: Adjuster,
        roll: RollDay,
    ) -> NaiveDateTime {
        adjuster.adjust(&roll.uadd(&date, months), self)
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
    fn adjust_py(&self, date: NaiveDateTime, adjuster: Adjuster) -> PyResult<NaiveDateTime> {
        Ok(self.adjust(&date, &adjuster))
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
        adjuster: Adjuster,
    ) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.adjusts(&dates, &adjuster))
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
    /// :meth:`~rateslib.calendars.Cal.add_bus_days` approach with a valid result.
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
    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = decode_from_slice(state.as_bytes(), legacy()).unwrap().0;
        Ok(())
    }
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &encode_to_vec(&self, legacy()).unwrap()))
    }
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
}

#[pymethods]
impl UnionCal {
    #[new]
    #[pyo3(signature = (calendars, settlement_calendars=None))]
    fn new_py(calendars: Vec<Cal>, settlement_calendars: Option<Vec<Cal>>) -> PyResult<Self> {
        Ok(UnionCal::new(calendars, settlement_calendars))
    }

    #[getter]
    fn holidays(&self) -> PyResult<Vec<NaiveDateTime>> {
        let mut set = self.calendars.iter().fold(IndexSet::new(), |acc, x| {
            IndexSet::from_iter(acc.union(&x.holidays).cloned())
        });
        set.sort();
        Ok(Vec::from_iter(set))
    }

    #[getter]
    fn week_mask(&self) -> PyResult<HashSet<u8>> {
        let mut s: HashSet<u8> = HashSet::new();
        for cal in &self.calendars {
            let ns = cal.week_mask()?;
            s.extend(&ns);
        }
        Ok(s)
    }

    #[getter]
    fn calendars(&self) -> Vec<Cal> {
        self.calendars.clone()
    }

    #[getter]
    fn settlement_calendars(&self) -> Option<Vec<Cal>> {
        self.settlement_calendars.clone()
    }

    /// Return whether the `date` is a business day.
    ///
    /// See :meth:`Cal.is_bus_day <rateslib.calendars.Cal.is_bus_day>`.
    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    /// Return whether the `date` is **not** a business day.
    ///
    /// See :meth:`Cal.is_non_bus_day <rateslib.calendars.Cal.is_non_bus_day>`.
    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    /// Return whether the `date` is a business day in an associated settlement calendar.
    ///
    /// If no such associated settlement calendar exists this will return *True*.
    ///
    /// See :meth:`Cal.is_settlement <rateslib.calendars.Cal.is_settlement>`.
    #[pyo3(name = "is_settlement")]
    fn is_settlement_py(&self, date: NaiveDateTime) -> bool {
        self.is_settlement(&date)
    }

    /// Return a date separated by calendar days from input date, and rolled with a modifier.
    ///
    /// See :meth:`Cal.add_days <rateslib.calendars.Cal.add_days>`.
    #[pyo3(name = "add_cal_days")]
    fn add_cal_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        adjuster: Adjuster,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_cal_days(&date, days, &adjuster))
    }

    /// Return a business date separated by `days` from an input business `date`.
    ///
    /// See :meth:`Cal.add_bus_days <rateslib.calendars.Cal.add_bus_days>`.
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
    /// See :meth:`Cal.add_months <rateslib.calendars.Cal.add_months>`.
    #[pyo3(name = "add_months")]
    fn add_months_py(
        &self,
        date: NaiveDateTime,
        months: i32,
        adjuster: Adjuster,
        roll: RollDay,
    ) -> NaiveDateTime {
        adjuster.adjust(&roll.uadd(&date, months), self)
    }

    /// Adjust a non-business date to a business date under a specific modification rule.
    ///
    /// See :meth:`Cal.adjust <rateslib.calendars.Cal.adjust>`.
    #[pyo3(name = "adjust")]
    fn adjust_py(&self, date: NaiveDateTime, adjuster: Adjuster) -> PyResult<NaiveDateTime> {
        Ok(self.adjust(&date, &adjuster))
    }

    /// Adjust a list of dates under a date adjustment rule.
    ///
    /// See :meth:`Cal.adjusts <rateslib.calendars.Cal.adjusts>`.
    #[pyo3(name = "adjusts")]
    fn adjusts_py(
        &self,
        dates: Vec<NaiveDateTime>,
        adjuster: Adjuster,
    ) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.adjusts(&dates, &adjuster))
    }

    /// Roll a date under a simplified adjustment rule.
    ///
    /// See :meth:`Cal.roll <rateslib.calendars.Cal.roll>`.
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
    /// See :meth:`Cal.lag_bus_days <rateslib.calendars.Cal.lag_bus_days>`.
    #[pyo3(name = "lag_bus_days")]
    fn lag_bus_days_py(&self, date: NaiveDateTime, days: i32, settlement: bool) -> NaiveDateTime {
        self.lag_bus_days(&date, days, settlement)
    }

    /// Return a list of business dates in a range.
    ///
    /// See :meth:`Cal.bus_date_range <rateslib.calendars.Cal.bus_date_range>`.
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
    /// See :meth:`Cal.cal_date_range <rateslib.calendars.Cal.cal_date_range>`.
    #[pyo3(name = "cal_date_range")]
    fn cal_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.cal_date_range(&start, &end)
    }

    // Pickling
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = decode_from_slice(state.as_bytes(), legacy()).unwrap().0;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &encode_to_vec(&self, legacy()).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(Vec<Cal>, Option<Vec<Cal>>)> {
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
}

#[pymethods]
impl NamedCal {
    #[new]
    fn new_py(name: String) -> PyResult<Self> {
        NamedCal::try_new(&name)
    }

    #[getter]
    fn holidays(&self) -> PyResult<Vec<NaiveDateTime>> {
        self.union_cal.holidays()
    }

    #[getter]
    fn week_mask(&self) -> PyResult<HashSet<u8>> {
        self.union_cal.week_mask()
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn union_cal(&self) -> UnionCal {
        self.union_cal.clone()
    }

    /// Return whether the `date` is a business day.
    ///
    /// See :meth:`Cal.is_bus_day <rateslib.calendars.Cal.is_bus_day>`.
    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    /// Return whether the `date` is **not** a business day.
    ///
    /// See :meth:`Cal.is_non_bus_day <rateslib.calendars.Cal.is_non_bus_day>`.
    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    /// Return whether the `date` is a business day in an associated settlement calendar.
    ///
    /// If no such associated settlement calendar exists this will return *True*.
    ///
    /// See :meth:`Cal.is_settlement <rateslib.calendars.Cal.is_settlement>`.
    #[pyo3(name = "is_settlement")]
    fn is_settlement_py(&self, date: NaiveDateTime) -> bool {
        self.is_settlement(&date)
    }

    /// Return a date separated by calendar days from input date, and rolled with a modifier.
    ///
    /// See :meth:`Cal.add_days <rateslib.calendars.Cal.add_days>`.
    #[pyo3(name = "add_cal_days")]
    fn add_cal_days_py(
        &self,
        date: NaiveDateTime,
        days: i32,
        adjuster: Adjuster,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_cal_days(&date, days, &adjuster))
    }

    /// Return a business date separated by `days` from an input business `date`.
    ///
    /// See :meth:`Cal.add_bus_days <rateslib.calendars.Cal.add_bus_days>`.
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
    /// See :meth:`Cal.add_months <rateslib.calendars.Cal.add_months>`.
    #[pyo3(name = "add_months")]
    fn add_months_py(
        &self,
        date: NaiveDateTime,
        months: i32,
        adjuster: Adjuster,
        roll: RollDay,
    ) -> NaiveDateTime {
        adjuster.adjust(&roll.uadd(&date, months), self)
    }

    /// Adjust a non-business date to a business date under a specific modification rule.
    ///
    /// See :meth:`Cal.adjust <rateslib.calendars.Cal.adjust>`.
    #[pyo3(name = "adjust")]
    fn adjust_py(&self, date: NaiveDateTime, adjuster: Adjuster) -> PyResult<NaiveDateTime> {
        Ok(self.adjust(&date, &adjuster))
    }

    /// Adjust a list of dates under a date adjustment rule.
    ///
    /// See :meth:`Cal.adjusts <rateslib.calendars.Cal.adjusts>`.
    #[pyo3(name = "adjusts")]
    fn adjusts_py(
        &self,
        dates: Vec<NaiveDateTime>,
        adjuster: Adjuster,
    ) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.adjusts(&dates, &adjuster))
    }

    /// Roll a date under a simplified adjustment rule.
    ///
    /// See :meth:`Cal.roll <rateslib.calendars.Cal.roll>`.
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
    /// See :meth:`Cal.lag_bus_days <rateslib.calendars.Cal.lag_bus_days>`.
    #[pyo3(name = "lag_bus_days")]
    fn lag_bus_days_py(&self, date: NaiveDateTime, days: i32, settlement: bool) -> NaiveDateTime {
        self.lag_bus_days(&date, days, settlement)
    }

    /// Return a list of business dates in a range.
    ///
    /// See :meth:`Cal.bus_date_range <rateslib.calendars.Cal.bus_date_range>`.
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
    /// See :meth:`Cal.cal_date_range <rateslib.calendars.Cal.cal_date_range>`.
    #[pyo3(name = "cal_date_range")]
    fn cal_date_range_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
    ) -> PyResult<Vec<NaiveDateTime>> {
        self.cal_date_range(&start, &end)
    }

    // Pickling
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = decode_from_slice(state.as_bytes(), legacy()).unwrap().0;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &encode_to_vec(&self, legacy()).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(String,)> {
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
}

/// Return a calendar container from named identifier.
#[pyfunction]
#[pyo3(name = "get_named_calendar")]
pub fn get_calendar_by_name_py(name: &str) -> PyResult<Cal> {
    get_calendar_by_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt};

    #[test]
    fn test_add_37_months() {
        let cal = get_calendar_by_name("all").unwrap();

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
                    Adjuster::FollowingSettle {},
                    RollDay::Unspecified {},
                ),
                dates[i].1
            )
        }
    }

    #[test]
    fn test_sub_37_months() {
        let cal = get_calendar_by_name("all").unwrap();

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
                    Adjuster::FollowingSettle {},
                    RollDay::Unspecified {},
                ),
                dates[i].1
            )
        }
    }

    #[test]
    fn test_add_months_py_roll() {
        let cal = get_calendar_by_name("all").unwrap();
        let roll = vec![
            (RollDay::Unspecified {}, ndt(1998, 3, 7), ndt(1996, 12, 7)),
            (
                RollDay::Int { day: 21 },
                ndt(1998, 3, 21),
                ndt(1996, 12, 21),
            ),
            (RollDay::EoM {}, ndt(1998, 3, 31), ndt(1996, 12, 31)),
            (RollDay::SoM {}, ndt(1998, 3, 1), ndt(1996, 12, 1)),
            (RollDay::IMM {}, ndt(1998, 3, 18), ndt(1996, 12, 18)),
        ];
        for i in 0..5 {
            assert_eq!(
                cal.add_months_py(roll[i].1, -15, Adjuster::FollowingSettle {}, roll[i].0),
                roll[i].2
            );
        }
    }

    #[test]
    fn test_add_months_roll_invalid_days() {
        let cal = get_calendar_by_name("all").unwrap();
        let roll = vec![
            (RollDay::Int { day: 21 }, ndt(1996, 12, 21)),
            (RollDay::EoM {}, ndt(1996, 12, 31)),
            (RollDay::SoM {}, ndt(1996, 12, 1)),
            (RollDay::IMM {}, ndt(1996, 12, 18)),
        ];
        for i in 0..4 {
            assert_eq!(
                roll[i].1,
                cal.add_months_py(
                    ndt(1998, 3, 7),
                    -15,
                    Adjuster::FollowingSettle {},
                    roll[i].0
                ),
            );
        }
    }

    #[test]
    fn test_add_months_modifier() {
        let cal = get_calendar_by_name("bus").unwrap();
        let modi = vec![
            (Adjuster::Actual {}, ndt(2023, 9, 30)),          // Saturday
            (Adjuster::FollowingSettle {}, ndt(2023, 10, 2)), // Monday
            (Adjuster::ModifiedFollowingSettle {}, ndt(2023, 9, 29)), // Friday
            (Adjuster::PreviousSettle {}, ndt(2023, 9, 29)),  // Friday
            (Adjuster::ModifiedPreviousSettle {}, ndt(2023, 9, 29)), // Friday
        ];
        for i in 0..4 {
            assert_eq!(
                cal.add_months_py(ndt(2023, 8, 31), 1, modi[i].0, RollDay::Unspecified {},),
                modi[i].1
            );
        }
    }

    #[test]
    fn test_add_months_modifier_p() {
        let cal = get_calendar_by_name("bus").unwrap();
        let modi = vec![
            (Adjuster::Actual {}, ndt(2023, 7, 1)),          // Saturday
            (Adjuster::FollowingSettle {}, ndt(2023, 7, 3)), // Monday
            (Adjuster::ModifiedFollowingSettle {}, ndt(2023, 7, 3)), // Monday
            (Adjuster::PreviousSettle {}, ndt(2023, 6, 30)), // Friday
            (Adjuster::ModifiedPreviousSettle {}, ndt(2023, 7, 3)), // Monday
        ];
        for i in 0..4 {
            assert_eq!(
                cal.add_months_py(ndt(2023, 8, 1), -1, modi[i].0, RollDay::Unspecified {}),
                modi[i].1
            );
        }
    }
}
