use crate::calendars::calendar::{Cal, UnionCal, DateRoll, Modifier, RollDay};
use crate::calendars::named::get_calendar_by_name;
use indexmap::set::IndexSet;
use chrono::{NaiveDateTime};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use bincode::{serialize, deserialize};

#[pymethods]
impl Cal {
    #[new]
    fn new_py(holidays: Vec<NaiveDateTime>, week_mask: Vec<u8>) -> PyResult<Self> {
        Ok(Cal::new(holidays, week_mask))
    }

    #[getter]
    fn holidays(&self) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.holidays.clone().into_iter().collect())
    }

    #[getter]
    fn week_mask(&self) -> PyResult<Vec<u8>> {
        Ok(self.week_mask.clone().into_iter().map(|x| x.num_days_from_monday() as u8).collect())
    }

    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    #[pyo3(name = "add_days")]
    fn add_days_py(&self, date: NaiveDateTime, days: i8, modifier: Modifier, settlement: bool) -> PyResult<NaiveDateTime> {
        Ok(self.add_days(&date, days, &modifier, settlement))
    }

    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> PyResult<NaiveDateTime> {
        self.add_bus_days(&date, days, settlement)
    }

    #[pyo3(name = "add_months")]
    fn add_months_py(&self, date: NaiveDateTime, months: i32, modifier: Modifier, roll: RollDay, settlement: bool) -> PyResult<NaiveDateTime> {
        Ok(self.add_months(&date, months, &modifier, &roll, settlement))
    }

    #[pyo3(name = "roll")]
    fn roll_py(&self, date: NaiveDateTime, modifier: Modifier, settlement: bool) -> PyResult<NaiveDateTime> {
        Ok(self.roll(&date, &modifier, settlement))
    }

    #[pyo3(name = "lag")]
    fn lag_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> NaiveDateTime {
        self.lag(&date, days, settlement)
    }

    #[pyo3(name = "bus_date_range")]
    fn bus_date_range_py(&self, start: NaiveDateTime, end: NaiveDateTime) -> PyResult<Vec<NaiveDateTime>> {
        self.bus_date_range(&start, &end)
    }

    pub fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(Vec<NaiveDateTime>, Vec<u8>)> {
        Ok((
            self.clone().holidays.into_iter().collect(),
            self.clone().week_mask.into_iter().map(|x| x.num_days_from_monday() as u8).collect()
        ))
    }
    pub fn __eq__(&self, other: Cal) -> bool {
        *self == other
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
        let mut set = self.calendars.iter().fold(IndexSet::new(), |acc, x| IndexSet::from_iter(acc.union(&x.holidays).cloned()));
        set.sort();
        Ok(Vec::from_iter(set.into_iter()))
    }

    #[getter]
    fn week_mask(&self) -> PyResult<Vec<u8>> {
        panic!("not implemented")
    }

    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    #[pyo3(name = "add_days")]
    fn add_days_py(&self, date: NaiveDateTime, days: i8, modifier: Modifier, settlement: bool) -> PyResult<NaiveDateTime> {
        Ok(self.add_days(&date, days, &modifier, settlement))
    }

    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> PyResult<NaiveDateTime> {
        self.add_bus_days(&date, days, settlement)
    }

    #[pyo3(name = "add_months")]
    fn add_months_py(&self, date: NaiveDateTime, months: i32, modifier: Modifier, roll: RollDay, settlement: bool) -> PyResult<NaiveDateTime> {
        Ok(self.add_months(&date, months, &modifier, &roll, settlement))
    }

    #[pyo3(name = "roll")]
    fn roll_py(&self, date: NaiveDateTime, modifier: Modifier, settlement: bool) -> PyResult<NaiveDateTime> {
        Ok(self.roll(&date, &modifier, settlement))
    }

    #[pyo3(name = "lag")]
    fn lag_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> NaiveDateTime {
        self.lag(&date, days, settlement)
    }

    #[pyo3(name = "bus_date_range")]
    fn bus_date_range_py(&self, start: NaiveDateTime, end: NaiveDateTime) -> PyResult<Vec<NaiveDateTime>> {
        self.bus_date_range(&start, &end)
    }

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
    pub fn __eq__(&self, other: UnionCal) -> bool {
        *self == other
    }
}

#[pyfunction]
#[pyo3(name = "get_named_calendar")]
pub fn get_calendar_by_name_py(name: &str) -> PyResult<Cal> {
    get_calendar_by_name(name)
}
