//! Wrapper module to export to Python using pyo3 bindings.

use crate::calendars::named::get_calendar_by_name;
use crate::calendars::{Cal, CalType, Convention, DateRoll, Modifier, NamedCal, RollDay, UnionCal};
use crate::json::json_py::DeserializedObj;
use crate::json::JSON;
use bincode::{deserialize, serialize};
use chrono::NaiveDateTime;
use indexmap::set::IndexSet;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashSet;

impl IntoPy<PyObject> for CalType {
    fn into_py(self, py: Python<'_>) -> PyObject {
        macro_rules! into_py {
            ($obj: ident) => {
                Py::new(py, $obj).unwrap().to_object(py)
            };
        }

        match self {
            CalType::Cal(i) => into_py!(i),
            CalType::UnionCal(i) => into_py!(i),
            CalType::NamedCal(i) => into_py!(i),
        }
    }
}

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
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
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
            Convention::Thirty360ISDA => Ok((6_u8,)),
            Convention::ActActISDA => Ok((8_u8,)),
            Convention::ActActICMA => Ok((9_u8,)),
            Convention::Bus252 => Ok((10_u8,)),
        }
    }
}

#[pymethods]
impl Modifier {
    // Pickling
    #[new]
    fn new_py(ad: u8) -> PyResult<Modifier> {
        match ad {
            0_u8 => Ok(Modifier::Act),
            1_u8 => Ok(Modifier::F),
            2_u8 => Ok(Modifier::ModF),
            3_u8 => Ok(Modifier::P),
            4_u8 => Ok(Modifier::ModP),
            _ => Err(PyValueError::new_err(
                "unreachable code on Convention pickle.",
            )),
        }
    }
    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__<'py>(&self) -> PyResult<(u8,)> {
        match self {
            Modifier::Act => Ok((0_u8,)),
            Modifier::F => Ok((1_u8,)),
            Modifier::ModF => Ok((2_u8,)),
            Modifier::P => Ok((3_u8,)),
            Modifier::ModP => Ok((4_u8,)),
        }
    }
}

#[pyfunction]
pub(crate) fn _get_modifier_str(modifier: Modifier) -> String {
    match modifier {
        Modifier::F => "F".to_string(),
        Modifier::ModF => "MF".to_string(),
        Modifier::P => "P".to_string(),
        Modifier::ModP => "MP".to_string(),
        Modifier::Act => "NONE".to_string(),
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
    /// modifier: Modifier
    ///     The rule to use to roll resultant non-business days.
    /// settlement: bool
    ///     Enforce an associated settlement calendar, if *True* and if one exists.
    ///
    /// Returns
    /// -------
    /// datetime
    #[pyo3(name = "add_days")]
    fn add_days_py(
        &self,
        date: NaiveDateTime,
        days: i8,
        modifier: Modifier,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_days(&date, days, &modifier, settlement))
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
    ///    :meth:`~rateslib.calendars.Cal.lag`: Add business days to inputs which are potentially
    ///    non-business dates.
    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(
        &self,
        date: NaiveDateTime,
        days: i8,
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
    /// modifier: Modifier
    ///     The rule to use to roll a resultant non-business day.
    /// roll: RollDay
    ///     The day of the month to adjust to.
    /// settlement: bool
    ///     Enforce an associated settlement calendar, if *True* and if one exists.
    #[pyo3(name = "add_months")]
    fn add_months_py(
        &self,
        date: NaiveDateTime,
        months: i32,
        modifier: Modifier,
        roll: RollDay,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_months(&date, months, &modifier, &roll, settlement))
    }

    /// Adjust a non-business date to a business date under a specific modification rule.
    ///
    /// Parameters
    /// -----------
    /// date: datetime
    ///     The date to adjust.
    /// modifier: Modifier
    ///     The modification rule
    /// settlement: bool
    ///     Whether to enforce settlement against an associated settlement calendar.
    ///
    /// Returns
    /// -------
    /// datetime
    ///
    /// Notes
    /// -----
    /// An input date which is already a settleable, business date will be returned unchanged.
    #[pyo3(name = "roll")]
    fn roll_py(
        &self,
        date: NaiveDateTime,
        modifier: Modifier,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.roll(&date, &modifier, settlement))
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
    /// ``lag`` and ``add_bus_days`` will return the same value if the input date is a business
    /// date. If not a business date, ``add_bus_days`` will raise, while ``lag`` will follow
    /// lag rules. ``lag`` should be used when the input date cannot be guaranteed to be a
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
    #[pyo3(name = "lag")]
    fn lag_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> NaiveDateTime {
        self.lag(&date, days, settlement)
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
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
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
    fn __eq__(&self, other: CalType) -> bool {
        match other {
            CalType::UnionCal(c) => *self == c,
            CalType::Cal(c) => *self == c,
            CalType::NamedCal(c) => *self == c,
        }
    }
}

#[pymethods]
impl UnionCal {
    #[new]
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
    #[pyo3(name = "add_days")]
    fn add_days_py(
        &self,
        date: NaiveDateTime,
        days: i8,
        modifier: Modifier,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_days(&date, days, &modifier, settlement))
    }

    /// Return a business date separated by `days` from an input business `date`.
    ///
    /// See :meth:`Cal.add_bus_days <rateslib.calendars.Cal.add_bus_days>`.
    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(
        &self,
        date: NaiveDateTime,
        days: i8,
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
        modifier: Modifier,
        roll: RollDay,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_months(&date, months, &modifier, &roll, settlement))
    }

    /// Adjust a non-business date to a business date under a specific modification rule.
    ///
    /// See :meth:`Cal.roll <rateslib.calendars.Cal.roll>`.
    #[pyo3(name = "roll")]
    fn roll_py(
        &self,
        date: NaiveDateTime,
        modifier: Modifier,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.roll(&date, &modifier, settlement))
    }

    /// Adjust a date by a number of business days, under lag rules.
    ///
    /// See :meth:`Cal.lag <rateslib.calendars.Cal.lag>`.
    #[pyo3(name = "lag")]
    fn lag_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> NaiveDateTime {
        self.lag(&date, days, settlement)
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
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
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
    fn __eq__(&self, other: CalType) -> bool {
        match other {
            CalType::UnionCal(c) => *self == c,
            CalType::Cal(c) => *self == c,
            CalType::NamedCal(c) => *self == c,
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
    #[pyo3(name = "add_days")]
    fn add_days_py(
        &self,
        date: NaiveDateTime,
        days: i8,
        modifier: Modifier,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_days(&date, days, &modifier, settlement))
    }

    /// Return a business date separated by `days` from an input business `date`.
    ///
    /// See :meth:`Cal.add_bus_days <rateslib.calendars.Cal.add_bus_days>`.
    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(
        &self,
        date: NaiveDateTime,
        days: i8,
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
        modifier: Modifier,
        roll: RollDay,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.add_months(&date, months, &modifier, &roll, settlement))
    }

    /// Adjust a non-business date to a business date under a specific modification rule.
    ///
    /// See :meth:`Cal.roll <rateslib.calendars.Cal.roll>`.
    #[pyo3(name = "roll")]
    fn roll_py(
        &self,
        date: NaiveDateTime,
        modifier: Modifier,
        settlement: bool,
    ) -> PyResult<NaiveDateTime> {
        Ok(self.roll(&date, &modifier, settlement))
    }

    /// Adjust a date by a number of business days, under lag rules.
    ///
    /// See :meth:`Cal.lag <rateslib.calendars.Cal.lag>`.
    #[pyo3(name = "lag")]
    fn lag_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> NaiveDateTime {
        self.lag(&date, days, settlement)
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
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
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
    fn __eq__(&self, other: CalType) -> bool {
        match other {
            CalType::UnionCal(c) => *self == c,
            CalType::Cal(c) => *self == c,
            CalType::NamedCal(c) => *self == c,
        }
    }
}

/// Return a calendar container from named identifier.
#[pyfunction]
#[pyo3(name = "get_named_calendar")]
pub fn get_calendar_by_name_py(name: &str) -> PyResult<Cal> {
    get_calendar_by_name(name)
}
