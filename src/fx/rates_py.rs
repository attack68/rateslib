use crate::dual::dual_py::DualsOrF64;
use crate::fx::rates::{FXRate, FXRates, Ccy};
use std::collections::HashMap;
use chrono::prelude::*;
use pyo3::prelude::*;

#[pymethods]
impl FXRate {
    #[new]
    fn new_py(
        lhs: &str,
        rhs: &str,
        rate: DualsOrF64,
        settlement: Option<NaiveDateTime>,
    ) -> PyResult<Self> {
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

#[pymethods]
impl FXRates {
    #[new]
    fn new_py(
        fx_rates: HashMap<String, DualsOrF64>,
        settlement: NaiveDateTime,
        base: Option<String>,
    ) -> PyResult<Self> {
        let base_ = base.map_or(None,|v| Some(Ccy::new(&v)));
        let fx_rates_: Vec<FXRate> = fx_rates.into_iter().map(|(k,v)| FXRate::new(&k[..3], &k[3..], v, Some(settlement))).collect();
        Ok(FXRates::new(fx_rates_, settlement, base_))
    }
//
//     #[getter]
//     #[pyo3(name = "rate")]
//     fn rate_py(&self) -> PyResult<DualsOrF64> {
//         Ok(self.rate.clone())
//     }
//
//     #[getter]
//     #[pyo3(name = "ad")]
//     fn ad_py(&self) -> PyResult<u8> {
//         Ok(self.ad)
//     }
//
//     #[getter]
//     #[pyo3(name = "settlement")]
//     fn settlement_py(&self) -> PyResult<Option<NaiveDateTime>> {
//         Ok(self.settlement)
//     }
//
//     #[getter]
//     #[pyo3(name = "pair")]
//     fn pair_py(&self) -> PyResult<String> {
//         Ok(format!("{}", self.pair))
//     }
}
