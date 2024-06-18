use pyo3::prelude::*;
use crate::dual::dual_py::DualsOrF64;
use crate::fx::rates::FXRate;
use chrono::prelude::*;

#[pymethods]
impl FXRate {
    #[new]
    fn new_py(lhs: &str, rhs: &str, rate: DualsOrF64, settlement: Option<NaiveDateTime>) -> PyResult<Self> {
        Ok(FXRate::new(lhs, rhs, rate, settlement))
    }

    #[getter]
    #[pyo3(name = "rate")]
    fn rate_py(&self) -> PyResult<DualsOrF64> {
        Ok(self.rate.clone())
    }

    #[getter]
    #[pyo3(name = "ad")]
    fn ad_py(&self) -> PyResult<u8> {
        Ok(self.ad)
    }

    #[getter]
    #[pyo3(name = "settlement")]
    fn settlement_py(&self) -> PyResult<Option<NaiveDateTime>> {
        Ok(self.settlement)
    }

    #[getter]
    #[pyo3(name = "pair")]
    fn pair_py(&self) -> PyResult<String> {
        Ok(format!("{}", self.pair))
    }
}