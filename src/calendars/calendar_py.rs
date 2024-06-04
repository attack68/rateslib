use pyo3::exceptions::PyValueError;
use crate::calendars::calendar::{Cal, UnionCal, DateRoll, Modifier};
use crate::calendars::named::get_calendar_by_name;
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

    #[pyo3(name = "is_bus_day")]
    fn is_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_bus_day(&date)
    }

    #[pyo3(name = "is_non_bus_day")]
    fn is_non_bus_day_py(&self, date: NaiveDateTime) -> bool {
        self.is_non_bus_day(&date)
    }

    #[pyo3(name = "add_days")]
    fn add_days_py(&self, date: NaiveDateTime, days: i8, modifier: &str, settlement: bool) -> PyResult<NaiveDateTime> {
        match modifier {
            "F" | "f" => Ok(self.add_days(&date, days, &Modifier::F, settlement)),
            "MF" | "mf" => Ok(self.add_days(&date, days, &Modifier::ModF, settlement)),
            "P" | "p" => Ok(self.add_days(&date, days, &Modifier::P, settlement)),
            "MP" | "mp" => Ok(self.add_days(&date, days, &Modifier::ModP, settlement)),
            "NONE" | "none" => Ok(self.add_days(&date, days, &Modifier::Act, settlement)),
            _ => Err(PyValueError::new_err("`modifier` must be in {'F', 'MF', 'P', 'MP', 'NONE'}."))
        }
    }

    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> PyResult<NaiveDateTime> {
        self.add_bus_days(&date, days, settlement)
    }

    #[pyo3(name = "roll")]
    fn roll_py(&self, date: NaiveDateTime, modifier: &str, settlement: bool) -> PyResult<NaiveDateTime> {
        match modifier {
            "F" | "f" => Ok(self.roll(&date, &Modifier::F, settlement)),
            "MF" | "mf" => Ok(self.roll(&date, &Modifier::ModF, settlement)),
            "P" | "p" => Ok(self.roll(&date, &Modifier::P, settlement)),
            "MP" | "mp" => Ok(self.roll(&date, &Modifier::ModP, settlement)),
            "NONE" | "none" => Ok(self.roll(&date, &Modifier::Act, settlement)),
            _ => Err(PyValueError::new_err("`modifier` must be in {'F', 'MF', 'P', 'MP', 'NONE'}."))
        }
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

}

#[pymethods]
impl UnionCal {
    #[new]
    fn new_py(calendars: Vec<Cal>, settlement_calendars: Option<Vec<Cal>>) -> PyResult<Self> {
        Ok(UnionCal::new(calendars, settlement_calendars))
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
    fn add_days_py(&self, date: NaiveDateTime, days: i8, modifier: &str, settlement: bool) -> PyResult<NaiveDateTime> {
        match modifier {
            "F" | "f" => Ok(self.add_days(&date, days, &Modifier::F, settlement)),
            "MF" | "mf" => Ok(self.add_days(&date, days, &Modifier::ModF, settlement)),
            "P" | "p" => Ok(self.add_days(&date, days, &Modifier::P, settlement)),
            "MP" | "mp" => Ok(self.add_days(&date, days, &Modifier::ModP, settlement)),
            "NONE" | "none" => Ok(self.add_days(&date, days, &Modifier::Act, settlement)),
            _ => Err(PyValueError::new_err("`modifier` must be in {'F', 'MF', 'P', 'MP', 'NONE'}."))
        }
    }

    #[pyo3(name = "add_bus_days")]
    fn add_bus_days_py(&self, date: NaiveDateTime, days: i8, settlement: bool) -> PyResult<NaiveDateTime> {
        self.add_bus_days(&date, days, settlement)
    }

    #[pyo3(name = "roll")]
    fn roll_py(&self, date: NaiveDateTime, modifier: &str, settlement: bool) -> PyResult<NaiveDateTime> {
        match modifier {
            "F" | "f" => Ok(self.roll(&date, &Modifier::F, settlement)),
            "MF" | "mf" => Ok(self.roll(&date, &Modifier::ModF, settlement)),
            "P" | "p" => Ok(self.roll(&date, &Modifier::P, settlement)),
            "MP" | "mp" => Ok(self.roll(&date, &Modifier::ModP, settlement)),
            "NONE" | "none" => Ok(self.roll(&date, &Modifier::Act, settlement)),
            _ => Err(PyValueError::new_err("`modifier` must be in {'F', 'MF', 'P', 'MP', 'NONE'}."))
        }
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
}

#[pyfunction]
#[pyo3(name = "get_named_calendar")]
pub fn get_calendar_by_name_py(name: &str) -> PyResult<Cal> {
    get_calendar_by_name(name)
}
