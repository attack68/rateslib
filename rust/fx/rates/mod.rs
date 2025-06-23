//! Create objects related to the management and valuation of monetary amounts in different
//! currencies, measured at different settlement dates in time.

use crate::dual::linalg::argabsmax;
use crate::dual::{set_order_clone, ADOrder, Dual, Dual2, Number, NumberArray2};
use crate::json::JSON;
use chrono::prelude::*;
use indexmap::set::IndexSet;
use itertools::Itertools;
use ndarray::{Array2, ArrayViewMut2, Axis};
use num_traits::{One, Zero};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::ops::{Div, Mul};

pub(crate) mod ccy;
pub use crate::fx::rates::ccy::Ccy;

pub(crate) mod fxpair;
pub use crate::fx::rates::fxpair::FXPair;

pub(crate) mod fxrate;
pub use crate::fx::rates::fxrate::FXRate;

/// A multi-currency FX market deriving all crosses from a vector of `FXRate`s.
#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(from = "FXRatesDataModel")]
pub struct FXRates {
    pub(crate) fx_rates: Vec<FXRate>,
    pub(crate) currencies: IndexSet<Ccy>,
    #[serde(skip)]
    pub(crate) fx_array: NumberArray2,
}

#[derive(Deserialize)]
struct FXRatesDataModel {
    fx_rates: Vec<FXRate>,
    currencies: IndexSet<Ccy>,
}

impl std::convert::From<FXRatesDataModel> for FXRates {
    fn from(model: FXRatesDataModel) -> Self {
        let base = model.currencies.first().unwrap();
        Self::try_new(model.fx_rates, Some(*base)).expect("FXRates data model contains bad data.")
    }
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

        let fx_array = create_fx_array(&currencies, &fx_rates, ADOrder::One)?;
        Ok(FXRates {
            fx_rates,
            fx_array,
            currencies,
        })
    }

    pub fn get_ccy_index(&self, currency: &Ccy) -> Option<usize> {
        self.currencies.get_index_of(currency)
    }

    pub fn rate(&self, lhs: &Ccy, rhs: &Ccy) -> Option<Number> {
        let dom_idx = self.currencies.get_index_of(lhs)?;
        let for_idx = self.currencies.get_index_of(rhs)?;
        match &self.fx_array {
            NumberArray2::F64(arr) => Some(Number::F64(arr[[dom_idx, for_idx]])),
            NumberArray2::Dual(arr) => Some(Number::Dual(arr[[dom_idx, for_idx]].clone())),
            NumberArray2::Dual2(arr) => Some(Number::Dual2(arr[[dom_idx, for_idx]].clone())),
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

    pub fn set_ad_order(&mut self, ad: ADOrder) -> Result<(), PyErr> {
        match (ad, &self.fx_array) {
            (ADOrder::Zero, NumberArray2::F64(_))
            | (ADOrder::One, NumberArray2::Dual(_))
            | (ADOrder::Two, NumberArray2::Dual2(_)) => {
                // leave the NumberArray2 unchanged.
                Ok(())
            }
            (ADOrder::One, NumberArray2::F64(_)) => {
                // rebuild the derivatives
                let fx_array = create_fx_array(&self.currencies, &self.fx_rates, ADOrder::One)?;
                self.fx_array = fx_array;
                Ok(())
            }
            (ADOrder::Two, NumberArray2::F64(_)) => {
                // rebuild the derivatives
                let fx_array = create_fx_array(&self.currencies, &self.fx_rates, ADOrder::Two)?;
                self.fx_array = fx_array;
                Ok(())
            }
            (ADOrder::One, NumberArray2::Dual2(arr)) => {
                let n: usize = arr.len_of(Axis(0));
                let fx_array = NumberArray2::Dual(
                    Array2::<Dual>::from_shape_vec(
                        (n, n),
                        arr.clone().into_iter().map(|d| d.into()).collect(),
                    )
                    .unwrap(),
                );
                self.fx_array = fx_array;
                Ok(())
            }
            (ADOrder::Zero, NumberArray2::Dual(arr)) => {
                // covert dual into f64
                let n: usize = arr.len_of(Axis(0));
                let fx_array = NumberArray2::F64(
                    Array2::<f64>::from_shape_vec(
                        (n, n),
                        arr.clone().into_iter().map(|d| d.real).collect(),
                    )
                    .unwrap(),
                );
                self.fx_array = fx_array;
                Ok(())
            }
            (ADOrder::Zero, NumberArray2::Dual2(arr)) => {
                // covert dual into f64
                let n: usize = arr.len_of(Axis(0));
                let fx_array = NumberArray2::F64(
                    Array2::<f64>::from_shape_vec(
                        (n, n),
                        arr.clone().into_iter().map(|d| d.real).collect(),
                    )
                    .unwrap(),
                );
                self.fx_array = fx_array;
                Ok(())
            }
            (ADOrder::Two, NumberArray2::Dual(_)) => {
                // rebuild derivatives
                let fx_array = create_fx_array(&self.currencies, &self.fx_rates, ADOrder::Two)?;
                self.fx_array = fx_array;
                Ok(())
            }
        }
    }
}

/// Return a one-hot mapping, in 2-d array form of the initial connections between currencies,
/// given the pairs associated with the FX rates.
fn create_initial_edges(currencies: &IndexSet<Ccy>, fx_pairs: &[FXPair]) -> Array2<i16> {
    let mut edges: Array2<i16> = Array2::eye(currencies.len());
    for pair in fx_pairs.iter() {
        let row = currencies.get_index_of(&pair.0).unwrap();
        let col = currencies.get_index_of(&pair.1).unwrap();
        edges[[row, col]] = 1_i16;
        edges[[col, row]] = 1_i16;
    }
    edges
}

/// Return a 2-d array containing all calculated FX rates as initially provided.
///
/// T will be an f64, Dual or Dual2
fn create_initial_fx_array<T>(
    currencies: &IndexSet<Ccy>,
    fx_pairs: &[FXPair],
    fx_rates: &[T],
) -> Array2<T>
where
    T: Clone + One + Zero,
    for<'a> f64: Div<&'a T, Output = T>,
{
    assert_eq!(fx_pairs.len(), fx_rates.len());
    let mut fx_array: Array2<T> = Array2::eye(currencies.len());

    for (i, pair) in fx_pairs.iter().enumerate() {
        let row = currencies.get_index_of(&pair.0).unwrap();
        let col = currencies.get_index_of(&pair.1).unwrap();
        fx_array[[row, col]] = fx_rates[i].clone();
        fx_array[[col, row]] = 1_f64 / &fx_array[[row, col]];
    }
    fx_array
}

fn mut_arrays_remaining_elements<T>(
    mut fx_array: ArrayViewMut2<T>,
    mut edges: ArrayViewMut2<i16>,
    mut prev_value: HashSet<usize>,
) -> Result<bool, PyErr>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
    for<'a> f64: Div<&'a T, Output = T>,
{
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
        return mut_arrays_remaining_elements(fx_array.view_mut(), edges.view_mut(), prev_value);
    } else {
        return mut_arrays_remaining_elements(
            fx_array.view_mut(),
            edges.view_mut(),
            HashSet::from([node]),
        );
    }
}

/// Creates an FX Array with the sparse graph network algorithm defining Dual variables directly.
fn create_fx_array(
    currencies: &IndexSet<Ccy>,
    fx_rates: &[FXRate],
    ad: ADOrder,
) -> Result<NumberArray2, PyErr> {
    let fx_pairs: Vec<FXPair> = fx_rates.iter().map(|x| x.pair).collect();
    let vars: Vec<String> = fx_pairs.iter().map(|x| format!("fx_{}", x)).collect();
    let mut edges = create_initial_edges(currencies, &fx_pairs);
    let fx_rates_: Vec<Number> = fx_rates
        .iter()
        .enumerate()
        .map(|(i, x)| set_order_clone(&x.rate, ad, vec![vars[i].clone()]))
        .collect();
    match ad {
        ADOrder::Zero => {
            let fx_rates__: Vec<f64> = fx_rates_.iter().map(f64::from).collect();
            let mut fx_array_: Array2<f64> =
                create_initial_fx_array(currencies, &fx_pairs, &fx_rates__);
            let _ = mut_arrays_remaining_elements(
                fx_array_.view_mut(),
                edges.view_mut(),
                HashSet::new(),
            )?;
            Ok(NumberArray2::F64(fx_array_))
        }
        ADOrder::One => {
            let fx_rates__: Vec<Dual> = fx_rates_.iter().map(Dual::from).collect();
            let mut fx_array_: Array2<Dual> =
                create_initial_fx_array(currencies, &fx_pairs, &fx_rates__);
            let _ = mut_arrays_remaining_elements(
                fx_array_.view_mut(),
                edges.view_mut(),
                HashSet::new(),
            )?;
            Ok(NumberArray2::Dual(fx_array_))
        }
        ADOrder::Two => {
            let fx_rates__: Vec<Dual2> = fx_rates_.iter().map(Dual2::from).collect();
            let mut fx_array_: Array2<Dual2> =
                create_initial_fx_array(currencies, &fx_pairs, &fx_rates__);
            let _ = mut_arrays_remaining_elements(
                fx_array_.view_mut(),
                edges.view_mut(),
                HashSet::new(),
            )?;
            Ok(NumberArray2::Dual2(fx_array_))
        }
    }
}

impl JSON for FXRates {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;
    use ndarray::arr2;

    #[test]
    fn fxrates_rate() {
        let fxr = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", Number::F64(1.08), Some(ndt(2004, 1, 1))).unwrap(),
                FXRate::try_new("usd", "jpy", Number::F64(110.0), Some(ndt(2004, 1, 1))).unwrap(),
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
            NumberArray2::Dual(arr) => arr.iter().map(|x| x.real()).collect(),
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
                FXRate::try_new("eur", "usd", Number::F64(1.0), Some(ndt(2004, 1, 1))).unwrap(),
                FXRate::try_new("usd", "eur", Number::F64(1.0), Some(ndt(2004, 1, 1))).unwrap(),
                FXRate::try_new("sek", "nok", Number::F64(1.0), Some(ndt(2004, 1, 1))).unwrap(),
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
                FXRate::try_new("eur", "usd", Number::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", Number::F64(110.0), None).unwrap(),
            ],
            None,
        )
        .unwrap();

        let fxr2 = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", Number::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", Number::F64(110.0), None).unwrap(),
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
                FXRate::try_new("eur", "usd", Number::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", Number::F64(110.0), None).unwrap(),
            ],
            None,
        )
        .unwrap();
        let _ = fxr.update(vec![FXRate::try_new(
            "usd",
            "jpy",
            Number::F64(120.0),
            None,
        )
        .unwrap()]);
        let rate = fxr
            .rate(&Ccy::try_new("eur").unwrap(), &Ccy::try_new("usd").unwrap())
            .unwrap();
        match rate {
            Number::Dual(d) => assert_eq!(d.real, 1.08),
            _ => panic!("failure"),
        };
        let rate = fxr
            .rate(&Ccy::try_new("usd").unwrap(), &Ccy::try_new("jpy").unwrap())
            .unwrap();
        match rate {
            Number::Dual(d) => assert_eq!(d.real, 120.0),
            _ => panic!("failure"),
        }
    }

    #[test]
    fn second_order_gradients_on_set_order() {
        let mut fxr = FXRates::try_new(
            vec![
                FXRate::try_new("usd", "nok", Number::F64(10.0), None).unwrap(),
                FXRate::try_new("eur", "nok", Number::F64(8.0), None).unwrap(),
            ],
            None,
        )
        .unwrap();
        let _ = fxr.set_ad_order(ADOrder::Two);
        let d1 = Dual2::new(10.0, vec!["fx_usdnok".to_string()]);
        let d2 = Dual2::new(8.0, vec!["fx_eurnok".to_string()]);
        let d3 = d1 / d2;
        let rate: Dual2 = fxr
            .rate(&Ccy::try_new("usd").unwrap(), &Ccy::try_new("eur").unwrap())
            .unwrap()
            .into();
        assert_eq!(d3, rate)
    }
}
