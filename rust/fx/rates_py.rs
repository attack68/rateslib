//! Wrapper module to export Rust FX rate data types to Python using pyo3 bindings.

use crate::dual::{ADOrder, Number, NumberArray2};
use crate::fx::rates::{Ccy, FXRate, FXRates};
use chrono::prelude::*;
use ndarray::Axis;
use pyo3::prelude::*;
// use std::collections::HashMap;
use pyo3::exceptions::PyValueError;
// use pyo3::exceptions::PyValueError;
// use pyo3::types::PyFloat;
use crate::json::json_py::DeserializedObj;
use crate::json::JSON;

#[pymethods]
impl Ccy {
    #[new]
    fn new_py(name: &str) -> PyResult<Self> {
        Ccy::try_new(name)
    }

    #[getter]
    #[pyo3(name = "name")]
    fn name_py(&self) -> PyResult<String> {
        Ok(self.name.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("<Ccy: '{}'>", self.name))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

#[pymethods]
impl FXRate {
    #[new]
    fn new_py(
        lhs: &str,
        rhs: &str,
        rate: Number,
        settlement: Option<NaiveDateTime>,
    ) -> PyResult<Self> {
        FXRate::try_new(lhs, rhs, rate, settlement)
    }

    #[getter]
    #[pyo3(name = "rate")]
    fn rate_py(&self) -> PyResult<Number> {
        Ok(self.rate.clone())
    }

    #[getter]
    #[pyo3(name = "ad")]
    fn ad_py(&self) -> u8 {
        match self.rate {
            Number::F64(_) => 0,
            Number::Dual(_) => 1,
            Number::Dual2(_) => 2,
        }
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

    fn __repr__(&self) -> PyResult<String> {
        match &self.rate {
            Number::F64(f) => Ok(format!("<FXRate: '{}' {}>", self.pair, f)),
            Number::Dual(d) => Ok(format!(
                "<FXRate: '{}' <Dual: {}, ..>>",
                self.pair,
                d.real()
            )),
            Number::Dual2(d) => Ok(format!(
                "<FXRate: '{}' <Dual2: {}, ..>>",
                self.pair,
                d.real()
            )),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

#[pymethods]
impl FXRates {
    // #[new]
    // fn new_py(
    //     fx_rates: HashMap<String, Number>,
    //     settlement: NaiveDateTime,
    //     base: Option<String>,
    // ) -> PyResult<Self> {
    //     let base_ = match base {
    //         None => None,
    //         Some(v) => Some(Ccy::try_new(&v)?),
    //     };
    //     let fx_rates_ = fx_rates
    //         .into_iter()
    //         .map(|(k, v)| FXRate::try_new(&k[..3], &k[3..], v, Some(settlement)))
    //         .collect::<Result<Vec<_>, _>>()?;
    //     FXRates::try_new(fx_rates_, settlement, base_)
    // }
    #[new]
    fn new_py(fx_rates: Vec<FXRate>, base: Option<Ccy>) -> PyResult<Self> {
        FXRates::try_new(fx_rates, base)
    }

    #[getter]
    #[pyo3(name = "fx_rates")]
    fn fx_rates_py(&self) -> PyResult<Vec<FXRate>> {
        Ok(self.fx_rates.clone())
    }

    #[getter]
    #[pyo3(name = "currencies")]
    fn currencies_py(&self) -> PyResult<Vec<Ccy>> {
        Ok(Vec::from_iter(self.currencies.iter().cloned()))
    }

    #[getter]
    #[pyo3(name = "ad")]
    fn ad_py(&self) -> PyResult<u8> {
        match &self.fx_array {
            NumberArray2::F64(_) => Ok(0),
            NumberArray2::Dual(_) => Ok(1),
            NumberArray2::Dual2(_) => Ok(2),
        }
    }

    #[getter]
    #[pyo3(name = "base")]
    fn base_py(&self) -> PyResult<Ccy> {
        Ok(self.currencies[0])
    }

    #[getter]
    #[pyo3(name = "fx_vector")]
    fn fx_vector_py(&self) -> PyResult<Vec<Number>> {
        match &self.fx_array {
            NumberArray2::F64(arr) => Ok(arr.row(0).iter().map(|x| Number::F64(*x)).collect()),
            NumberArray2::Dual(arr) => {
                Ok(arr.row(0).iter().map(|x| Number::Dual(x.clone())).collect())
            }
            NumberArray2::Dual2(arr) => Ok(arr
                .row(0)
                .iter()
                .map(|x| Number::Dual2(x.clone()))
                .collect()),
        }
    }

    #[getter]
    #[pyo3(name = "fx_array")]
    fn fx_array_py(&self) -> PyResult<Vec<Vec<Number>>> {
        match &self.fx_array {
            NumberArray2::F64(arr) => Ok(arr
                .lanes(Axis(1))
                .into_iter()
                .map(|row| row.iter().map(|d| Number::F64(*d)).collect())
                .collect()),
            NumberArray2::Dual(arr) => Ok(arr
                .lanes(Axis(1))
                .into_iter()
                .map(|row| row.iter().map(|d| Number::Dual(d.clone())).collect())
                .collect()),
            NumberArray2::Dual2(arr) => Ok(arr
                .lanes(Axis(1))
                .into_iter()
                .map(|row| row.iter().map(|d| Number::Dual2(d.clone())).collect())
                .collect()),
        }
    }

    #[pyo3(name = "get_ccy_index")]
    fn get_ccy_index_py(&self, currency: Ccy) -> Option<usize> {
        self.get_ccy_index(&currency)
    }

    #[pyo3(name = "rate")]
    fn rate_py(&self, lhs: &Ccy, rhs: &Ccy) -> PyResult<Option<Number>> {
        Ok(self.rate(lhs, rhs))
    }

    #[pyo3(name = "update")]
    fn update_py(&mut self, fx_rates: Vec<FXRate>) -> PyResult<()> {
        self.update(fx_rates)
    }

    #[pyo3(name = "set_ad_order")]
    fn set_ad_order_py(&mut self, ad: ADOrder) -> PyResult<()> {
        self.set_ad_order(ad)?;
        Ok(())
    }

    // JSON
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::FXRates(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `UnionCal` to JSON.",
            )),
        }
    }

    // Equality
    fn __eq__(&self, other: FXRates) -> bool {
        println!("{:?}", self);
        println!("{:?}", other);
        self.eq(&other)
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;

    #[test]
    fn fxrates_eq() {
        let fxr = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", Number::F64(1.08), Some(ndt(2004, 1, 1))).unwrap(),
                FXRate::try_new("usd", "jpy", Number::F64(110.0), Some(ndt(2004, 1, 1))).unwrap(),
            ],
            None,
        )
        .unwrap();

        let fxr2 = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", Number::F64(1.08), Some(ndt(2004, 1, 1))).unwrap(),
                FXRate::try_new("usd", "jpy", Number::F64(110.0), Some(ndt(2004, 1, 1))).unwrap(),
            ],
            None,
        )
        .unwrap();

        assert!(fxr.__eq__(fxr2))
    }
}
