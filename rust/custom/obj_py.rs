use pyo3::prelude::*;
use chrono::prelude::*;
use crate::custom::obj::DataProvider;

#[pymethods]
impl DataProvider {
    #[new]
    fn new_py() -> PyResult<Self> {
        Ok(DataProvider::new())
    }

//     #[getter]
//     fn get_dates(&self) -> PyResult<Vec<NaiveDateTime>> {
//         Ok(self.dates.clone())
//     }
//
//     #[getter]
//     fn get_val(&self) -> PyResult<f64> {
//         Ok(self.val)
//     }
//
//     #[getter]
//     fn get_vals(&self) -> PyResult<Vec<f64>> {
//         Ok(self.vals.clone())
//     }
}