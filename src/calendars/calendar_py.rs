use pyo3::exceptions::PyValueError;
use crate::calendars::calendar::{Cal, UnionCal, DateRoll, Modifier};
use chrono::prelude::*;
use pyo3::prelude::*;

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
}
