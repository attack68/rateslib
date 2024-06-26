//! Wrapper module to export Rust FX rate data types to Python using pyo3 bindings.

use crate::dual::dual_py::DualsOrF64;
use crate::fx::rates::{Ccy, FXRate, FXRates, FXArray};
use chrono::prelude::*;
use pyo3::prelude::*;
// use std::collections::HashMap;
use ndarray::Axis;
// use pyo3::exceptions::PyValueError;
// use pyo3::types::PyFloat;

#[pymethods]
impl Ccy {
    #[new]
    fn new_py(name: &str) -> PyResult<Self> {
        Ok(Ccy::try_new(name)?)
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


// impl IntoPy<PyObject> for FXArray {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         match self {
//             FXArray::F64(arr) => {
//                 let v: Vec<Vec<f64>> = arr
//                     .lanes(Axis(0))
//                     .into_iter()
//                     .map(|row| row.iter().map(|d| DualsOrF64::F64(d.clone())).collect())
//                     .collect();
//                 Py::new(py, v).unwrap().to_object(py)
//             },
//             FXArray::Dual(arr) => {
//                 let v: Vec<Vec<DualsOrF64>> = arr
//                     .lanes(Axis(0))
//                     .into_iter()
//                     .map(|row| row.iter().map(|d| DualsOrF64::Dual(d.clone())).collect())
//                     .collect();
//                 Py::new(py, v).unwrap().to_object(py)
//             },
//             FXArray::Dual2(arr) => {
//                 let v: Vec<Vec<DualsOrF64>> = arr
//                     .lanes(Axis(0))
//                     .into_iter()
//                     .map(|row| row.iter().map(|d| DualsOrF64::Dual2(d.clone())).collect())
//                     .collect();
//                 Py::new(py, v).unwrap().to_object(py)
//             },
//         }
//     }
// }

#[pymethods]
impl FXRate {
    #[new]
    fn new_py(
        lhs: &str,
        rhs: &str,
        rate: DualsOrF64,
        settlement: Option<NaiveDateTime>,
    ) -> PyResult<Self> {
        Ok(FXRate::try_new(lhs, rhs, rate, settlement)?)
    }

    #[getter]
    #[pyo3(name = "rate")]
    fn rate_py(&self) -> PyResult<DualsOrF64> {
        Ok(self.rate.clone())
    }

    #[getter]
    #[pyo3(name = "ad")]
    fn ad_py(&self) -> u8 {
        match self.rate {
            DualsOrF64::F64(_) => 0,
            DualsOrF64::Dual(_) => 1,
            DualsOrF64::Dual2(_) => 2,
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
            DualsOrF64::F64(f) => Ok(format!("<FXRate: '{}' {}>", self.pair, f)),
            DualsOrF64::Dual(d) => Ok(format!(
                "<FXRate: '{}' <Dual: {}, ..>>",
                self.pair,
                d.real()
            )),
            DualsOrF64::Dual2(d) => Ok(format!(
                "<FXRate: '{}' <Dual2: {}, ..>>",
                self.pair,
                d.real()
            )),
        }
    }
}

#[pymethods]
impl FXRates {
    // #[new]
    // fn new_py(
    //     fx_rates: HashMap<String, DualsOrF64>,
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
    fn new_py(
        fx_rates: Vec<FXRate>,
        settlement: NaiveDateTime,
        base: Option<Ccy>,
    ) -> PyResult<Self> {
        FXRates::try_new(fx_rates, settlement, base)
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
            FXArray::F64(_) => Ok(0),
            FXArray::Dual(_) => Ok(1),
            FXArray::Dual2(_) => Ok(2),
        }
    }

    #[getter]
    #[pyo3(name = "base")]
    fn base_py(&self) -> PyResult<Ccy> {
        Ok(self.currencies[0])
    }

    #[getter]
    #[pyo3(name = "fx_vector")]
    fn fx_vector_py(&self) -> PyResult<Vec<DualsOrF64>> {
        match &self.fx_array {
            FXArray::F64(arr) => Ok(arr.row(0).iter().map(|x| DualsOrF64::F64(x.clone())).collect()),
            FXArray::Dual(arr) => Ok(arr.row(0).iter().map(|x| DualsOrF64::Dual(x.clone())).collect()),
            FXArray::Dual2(arr) => Ok(arr.row(0).iter().map(|x| DualsOrF64::Dual2(x.clone())).collect()),
        }
    }

    #[getter]
    #[pyo3(name = "fx_array")]
    fn fx_array_py(&self) -> PyResult<Vec<Vec<DualsOrF64>>> {
        match &self.fx_array {
            FXArray::F64(arr) => {
                Ok(arr
                    .lanes(Axis(1))
                    .into_iter()
                    .map(|row| row.iter().map(|d| DualsOrF64::F64(d.clone())).collect())
                    .collect())
            },
            FXArray::Dual(arr) => {
                Ok(arr
                    .lanes(Axis(1))
                    .into_iter()
                    .map(|row| row.iter().map(|d| DualsOrF64::Dual(d.clone())).collect())
                    .collect())
            },
            FXArray::Dual2(arr) => {
                Ok(arr
                    .lanes(Axis(1))
                    .into_iter()
                    .map(|row| row.iter().map(|d| DualsOrF64::Dual2(d.clone())).collect())
                    .collect())
            },
        }
    }

    #[pyo3(name = "get_ccy_index")]
    fn get_ccy_index_py(&self, currency: Ccy) -> Option<usize> {
        self.get_ccy_index(&currency)
    }
    #[pyo3(name = "rate")]
    fn rate_py(&self, lhs: &Ccy, rhs: &Ccy) -> PyResult<Option<DualsOrF64>> {
        Ok(self.rate(lhs, rhs))
    }

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
