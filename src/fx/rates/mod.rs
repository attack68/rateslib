
//! Create objects related to the management and valuation of monetary amounts in different
//! currencies, measured at different settlement dates in time.

use crate::dual::dual::{Dual, Dual2, ADOrder};
use crate::dual::dual_py::DualsOrF64;
use crate::dual::linalg::argabsmax;
use crate::json::JSON;
use chrono::prelude::*;
use indexmap::set::IndexSet;
use itertools::Itertools;
use ndarray::{Array2, ArrayViewMut2, Axis};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub(crate) mod ccy;
pub use crate::fx::rates::ccy::Ccy;

pub(crate) mod fxpair;
pub use crate::fx::rates::fxpair::FXPair;

pub(crate) mod fxrate;
pub use crate::fx::rates::fxrate::FXRate;

/// Two-dimensional data contain FX rate crosses with appropriate AD order.
///
/// The structure of these matrices enforce FX rate inversion: i.e. *L = 1 / U*.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FXArray {
    F64(Array2<f64>),
    Dual(Array2<Dual>),
    Dual2(Array2<Dual2>),
}

/// A multi-currency FX market deriving all crosses from a vector of `FXRate`s.
#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FXRates {
    pub(crate) fx_rates: Vec<FXRate>,
    pub(crate) currencies: IndexSet<Ccy>,
    pub(crate) fx_array: FXArray,
}

impl FXRates {
    pub fn try_new(fx_rates: Vec<FXRate>, base: Option<Ccy>) -> Result<Self, PyErr> {
        // Validations:
        // 1. fx_rates is non-zero length
        // 2. currencies are not under or over overspecified
        // 3. settlement dates are all consistent.
        // 4. No Dual2 data types are provided as input

        // 1.
        if fx_rates.is_empty() {
            return Err(PyValueError::new_err(
                "`fx_rates` must contain at least on fx rate.",
            ));
        }

        let mut currencies: IndexSet<Ccy> = IndexSet::with_capacity(fx_rates.len() + 1_usize);
        if let Some(ccy) = base {
            currencies.insert(ccy);
        }
        for fxr in fx_rates.iter() {
            currencies.insert(fxr.pair.0);
            currencies.insert(fxr.pair.1);
        }
        let q = currencies.len();

        // 2.
        if q > (fx_rates.len() + 1) {
            return Err(PyValueError::new_err(
                "FX Array cannot be solved. `fx_rates` is underspecified.",
            ));
        } else if q < (fx_rates.len() + 1) {
            return Err(PyValueError::new_err(
                "FX Array cannot be solved. `fx_rates` is overspecified.",
            ));
        }

        // 3.
        let settlement: Option<NaiveDateTime> = fx_rates[0].settlement;
        match settlement {
            Some(date) => {
                if !(&fx_rates
                    .iter()
                    .all(|d| d.settlement.map_or(false, |v| v == date)))
                {
                    return Err(PyValueError::new_err(
                        "`fx_rates` must have consistent `settlement` dates across all rates.",
                    ));
                }
            }
            None => {
                if !(&fx_rates
                    .iter()
                    .all(|d| d.settlement.map_or(true, |_v| false)))
                {
                    return Err(PyValueError::new_err(
                        "`fx_rates` must have consistent `settlement` dates across all rates.",
                    ));
                }
            }
        }

        let (mut fx_array, mut edges) = FXRates::_create_initial_arrays(&currencies, &fx_rates);
        let _ = FXRates::_mut_arrays_remaining_elements(
            fx_array.view_mut(),
            edges.view_mut(),
            HashSet::new(),
        )?;

        Ok(FXRates {
            fx_rates,
            fx_array: FXArray::Dual(fx_array),
            currencies,
        })
    }

    fn _create_initial_arrays(
        currencies: &IndexSet<Ccy>,
        fx_rates: &[FXRate],
        ad: ADOrder,
    ) -> (FXArray, Array2<i16>) {
        let mut fx_array = match ad {
            ADOrder::Zero => { FXArray::F64(Array2::<f64>::eye(currencies.len())) }
            ADOrder::One => { FXArray::Dual(Array2::<Dual>::eye(currencies.len())) }
            ADOrder::Two => { FXArray::Dual2(Array2::<Dual2>::eye(currencies.len())) }
        };
        let mut edges: Array2<i16> = Array2::eye(currencies.len());
        for fxr in fx_rates.iter() {
            let row = currencies.get_index_of(&fxr.pair.0).unwrap();
            let col = currencies.get_index_of(&fxr.pair.1).unwrap();
            edges[[row, col]] = 1_i16;
            edges[[col, row]] = 1_i16;
            match &fxr.rate {
                DualsOrF64::F64(f) => {
                    *fx_array[[row, col]] =
                        Dual::new(*f, vec!["fx_".to_string() + &format!("{}", fxr.pair)]);
                    *fx_array[[col, row]] = 1_f64 / &fx_array[[row, col]];
                }
                DualsOrF64::Dual(d) => {
                    fx_array[[row, col]] = d.clone();
                    fx_array[[col, row]] = 1_f64 / &fx_array[[row, col]];
                }
                DualsOrF64::Dual2(_) => panic!("cannot construct from dual2 rates"),
            }
        }
        (fx_array, edges)
    }

    fn _mut_arrays_remaining_elements(
        mut fx_array: ArrayViewMut2<Dual>,
        mut edges: ArrayViewMut2<i16>,
        mut prev_value: HashSet<usize>,
    ) -> Result<bool, PyErr> {
        if prev_value.len() == edges.len_of(Axis(0)) {
            return Err(PyValueError::new_err(
                "FX Array cannot be solved. There are degenerate FX rate pairs.\n\
                For example ('eurusd' + 'usdeur') or ('usdeur', 'eurjpy', 'usdjpy').",
            ));
        }
        if edges.sum() == ((edges.len_of(Axis(0)) * edges.len_of(Axis(1))) as i16) {
            return Ok(true); // all edges and values have already been populated.
        }
        let mut row_edges = edges.sum_axis(Axis(1));

        let mut node: usize = edges.len_of(Axis(1)) + 1_usize;
        let mut combinations_: Vec<Vec<usize>> = Vec::new();
        let mut start_flag = true;
        while start_flag || prev_value.contains(&node) {
            start_flag = false;

            // find node with most outgoing edges
            node = argabsmax(row_edges.view());
            row_edges[node] = 0_i16;

            // filter by combinations that are not already populated
            combinations_ = edges
                .row(node)
                .iter()
                .zip(0_usize..)
                .filter(|(v, i)| **v == 1_i16 && *i != node)
                .map(|(_v, i)| i)
                .combinations(2)
                .filter(|v| edges[[v[0], v[1]]] == 0_i16)
                .collect();
        }

        let mut counter: i16 = 0;
        for c in combinations_ {
            counter += 1_i16;
            edges[[c[0], c[1]]] = 1_i16;
            edges[[c[1], c[0]]] = 1_i16;
            fx_array[[c[0], c[1]]] = &fx_array[[c[0], node]] * &fx_array[[node, c[1]]];
            fx_array[[c[1], c[0]]] = 1.0_f64 / &fx_array[[c[0], c[1]]];
        }

        if counter == 0 {
            prev_value.insert(node);
            return FXRates::_mut_arrays_remaining_elements(
                fx_array.view_mut(),
                edges.view_mut(),
                prev_value,
            );
        } else {
            return FXRates::_mut_arrays_remaining_elements(
                fx_array.view_mut(),
                edges.view_mut(),
                HashSet::from([node]),
            );
        }
    }

    pub fn get_ccy_index(&self, currency: &Ccy) -> Option<usize> {
        self.currencies.get_index_of(currency)
    }

    pub fn rate(&self, lhs: &Ccy, rhs: &Ccy) -> Option<DualsOrF64> {
        let dom_idx = self.currencies.get_index_of(lhs)?;
        let for_idx = self.currencies.get_index_of(rhs)?;
        match &self.fx_array {
            FXArray::F64(arr) => Some(DualsOrF64::F64(arr[[dom_idx, for_idx]])),
            FXArray::Dual(arr) => Some(DualsOrF64::Dual(arr[[dom_idx, for_idx]].clone())),
            FXArray::Dual2(arr) => Some(DualsOrF64::Dual2(arr[[dom_idx, for_idx]].clone())),
        }
    }

    pub fn update(&mut self, fx_rates: Vec<FXRate>) -> Result<(), PyErr> {
        // validate that the input vector contains FX pairs that are already associated with the instance
        if !(fx_rates
            .iter()
            .all(|v| self.fx_rates.iter().any(|x| x.pair == v.pair)))
        {
            return Err(PyValueError::new_err(
                "The given `fx_rates` pairs are not contained in the `FXRates` object.",
            ));
        }
        let mut fx_rates_: Vec<FXRate> = self.fx_rates.clone();
        for fxr in fx_rates.into_iter() {
            let idx = fx_rates_.iter().enumerate().fold(0_usize, |a, (i, v)| {
                if fxr.pair.eq(&v.pair) {
                    i
                } else {
                    a
                }
            });
            fx_rates_[idx] = fxr;
        }
        let new_fxr = FXRates::try_new(fx_rates_, Some(self.currencies[0]))?;
        self.fx_rates.clone_from(&new_fxr.fx_rates);
        self.currencies.clone_from(&new_fxr.currencies);
        self.fx_array = new_fxr.fx_array.clone();
        Ok(())
    }

    pub fn set_ad_order(&mut self, ad: usize) {
        match (ad, &self.fx_array) {
            (0, FXArray::F64(_)) | (1, FXArray::Dual(_)) | (2, FXArray::Dual2(_)) => {}
            (1, FXArray::Dual2(arr)) => {
                let n: usize = arr.len_of(Axis(0));
                let fx_array = FXArray::Dual(
                    Array2::<Dual>::from_shape_vec(
                        (n, n),
                        arr.clone().into_iter().map(|d| d.into()).collect(),
                    )
                        .unwrap(),
                );
                self.fx_array = fx_array;
            }
            (2, FXArray::Dual(arr)) => {
                let n: usize = arr.len_of(Axis(0));
                let fx_array = FXArray::Dual2(
                    Array2::<Dual2>::from_shape_vec(
                        (n, n),
                        arr.clone().into_iter().map(|d| d.into()).collect(),
                    )
                        .unwrap(),
                );
                self.fx_array = fx_array;
            }
            (0, FXArray::Dual(arr)) => {
                let n: usize = arr.len_of(Axis(0));
                let fx_array = FXArray::F64(
                    Array2::<f64>::from_shape_vec(
                        (n, n),
                        arr.clone().into_iter().map(|d| d.real).collect(),
                    )
                        .unwrap(),
                );
                self.fx_array = fx_array;
            }
            (0, FXArray::Dual2(arr)) => {
                let n: usize = arr.len_of(Axis(0));
                let fx_array = FXArray::F64(
                    Array2::<f64>::from_shape_vec(
                        (n, n),
                        arr.clone().into_iter().map(|d| d.real).collect(),
                    )
                        .unwrap(),
                );
                self.fx_array = fx_array;
            }
            (1, FXArray::F64(_)) => {
                let (mut fx_array, mut edges) =
                    FXRates::_create_initial_arrays(&self.currencies, &self.fx_rates);
                let _ = FXRates::_mut_arrays_remaining_elements(
                    fx_array.view_mut(),
                    edges.view_mut(),
                    HashSet::new(),
                );
                self.fx_array = FXArray::Dual(fx_array);
            }
            (2, FXArray::F64(_)) => {
                let (mut fx_array, mut edges) =
                    FXRates::_create_initial_arrays(&self.currencies, &self.fx_rates);
                let _ = FXRates::_mut_arrays_remaining_elements(
                    fx_array.view_mut(),
                    edges.view_mut(),
                    HashSet::new(),
                );
                let n: usize = fx_array.len_of(Axis(0));
                let fx_array2 = FXArray::Dual2(
                    Array2::<Dual2>::from_shape_vec(
                        (n, n),
                        fx_array.into_iter().map(|d| d.into()).collect(),
                    )
                        .unwrap(),
                );
                self.fx_array = fx_array2;
            }
            _ => panic!("unreachable pattern for AD: 0, 1, 2"),
        }
    }
}

impl JSON for FXRates {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::calendar::ndt;
    use ndarray::arr2;

    #[test]
    fn fxrates_rate() {
        let fxr = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", DualsOrF64::F64(1.08), Some(ndt(2004, 1, 1)))
                    .unwrap(),
                FXRate::try_new("usd", "jpy", DualsOrF64::F64(110.0), Some(ndt(2004, 1, 1)))
                    .unwrap(),
            ],
            None,
        )
            .unwrap();

        let expected = arr2(&[
            [1.0, 1.08, 118.8],
            [0.9259259, 1.0, 110.0],
            [0.0084175, 0.0090909, 1.0],
        ]);

        let arr: Vec<f64> = match fxr.fx_array {
            FXArray::Dual(arr) => arr.iter().map(|x| x.real()).collect(),
            _ => panic!("unreachable"),
        };
        assert!(arr
            .iter()
            .zip(expected.iter())
            .all(|(x, y)| (x - y).abs() < 1e-6))
    }

    #[test]
    fn fxrates_creation_error() {
        let fxr = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", DualsOrF64::F64(1.0), Some(ndt(2004, 1, 1))).unwrap(),
                FXRate::try_new("usd", "eur", DualsOrF64::F64(1.0), Some(ndt(2004, 1, 1))).unwrap(),
                FXRate::try_new("sek", "nok", DualsOrF64::F64(1.0), Some(ndt(2004, 1, 1))).unwrap(),
            ],
            None,
        );
        match fxr {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    #[test]
    fn fxrates_eq() {
        let fxr = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", DualsOrF64::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", DualsOrF64::F64(110.0), None).unwrap(),
            ],
            None,
        )
            .unwrap();

        let fxr2 = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", DualsOrF64::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", DualsOrF64::F64(110.0), None).unwrap(),
            ],
            None,
        )
            .unwrap();

        assert_eq!(fxr, fxr2)
    }

    #[test]
    fn fxrates_update() {
        let mut fxr = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", DualsOrF64::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", DualsOrF64::F64(110.0), None).unwrap(),
            ],
            None,
        )
            .unwrap();
        let _ = fxr.update(vec![FXRate::try_new(
            "usd",
            "jpy",
            DualsOrF64::F64(120.0),
            None,
        )
            .unwrap()]);
        let rate = fxr
            .rate(&Ccy::try_new("eur").unwrap(), &Ccy::try_new("usd").unwrap())
            .unwrap();
        match rate {
            DualsOrF64::Dual(d) => assert_eq!(d.real, 1.08),
            _ => panic!("failure"),
        };
        let rate = fxr
            .rate(&Ccy::try_new("usd").unwrap(), &Ccy::try_new("jpy").unwrap())
            .unwrap();
        match rate {
            DualsOrF64::Dual(d) => assert_eq!(d.real, 120.0),
            _ => panic!("failure"),
        }
    }
}
