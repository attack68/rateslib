use pyo3::prelude::*;
use chrono::prelude::*;
use crate::custom::obj::DataProvider;


#[pymethods]
impl DataProvider {
    #[new]
    fn new_py() -> PyResult<Self> {
        Ok(DataProvider::new())
    }

    #[getter]
    #[pyo3(name = "real")]
    fn get_dates_py(&self) -> PyResult<Vec<NaiveDateTime>> {
        Ok(self.dates.clone())
    }
}