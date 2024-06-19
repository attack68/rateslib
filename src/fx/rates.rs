//! Create objects related to the management and valuation of monetary amounts in different
//! currencies, measured at different settlement dates in time.

use crate::dual::dual1::Dual;
use crate::dual::dual2::Dual2;
use crate::dual::dual_py::DualsOrF64;
use crate::dual::linalg::{douter11_, dsolve};
use chrono::prelude::*;
use indexmap::set::IndexSet;
use internment::Intern;
use ndarray::{Array1, Array2};
use num_traits::identities::One;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use std::fmt;

/// Struct to define a currency.
#[pyclass]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ccy {
    name: Intern<String>,
}

impl Ccy {
    /// Constructs a new `Ccy`.
    ///
    /// Use **only** 3-ascii names. e.g. *"usd"*, aligned with ISO representation. `name` is converted
    /// to lowercase to promote performant equality between "USD" and "usd".
    ///
    /// Panics if `name` is not 3 bytes in length.
    pub fn try_new(name: &str) -> Result<Self, PyErr> {
        let ccy: String = name.to_string().to_lowercase();
        if ccy.len() != 3 {
            return Err(PyValueError::new_err(
                "`Ccy` must be 3 ascii character in length, e.g. 'usd'.",
            ));
        }
        Ok(Ccy {
            name: Intern::new(ccy),
        })
    }
}

/// Struct to define a currency cross.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FXPair(Ccy, Ccy);

impl FXPair {
    /// Constructs a new `FXCross`, as a combination of two distinct `Ccy`s.
    pub fn try_new(lhs: &str, rhs: &str) -> Result<Self, PyErr> {
        let lhs_ = Ccy::try_new(lhs)?;
        let rhs_ = Ccy::try_new(rhs)?;
        if lhs_ == rhs_ {
            return Err(PyValueError::new_err(
                "`FXPair` must be created from two distinct currencies, not same.",
            ));
        }
        Ok(FXPair(lhs_, rhs_))
    }
}

impl fmt::Display for FXPair {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.0.name, self.1.name)
    }
}

/// Struct to define an FXRate via FXCross, rate and settlement info.
#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone)]
pub struct FXRate {
    pub(crate) pair: FXPair,
    pub(crate) rate: DualsOrF64,
    pub(crate) settlement: Option<NaiveDateTime>,
    pub(crate) ad: u8,
}

impl FXRate {
    pub fn try_new(
        lhs: &str,
        rhs: &str,
        rate: DualsOrF64,
        settlement: Option<NaiveDateTime>,
    ) -> Result<Self, PyErr> {
        let ad = match rate {
            DualsOrF64::F64(_) => 0_u8,
            DualsOrF64::Dual(_) => 1_u8,
            DualsOrF64::Dual2(_) => 2_u8,
        };
        Ok(FXRate {
            pair: FXPair::try_new(lhs, rhs)?,
            rate,
            settlement,
            ad,
        })
    }
}

#[derive(Debug, Clone)]
pub enum FXVector {
    Dual(Array1<Dual>),
    Dual2(Array1<Dual2>),
}

#[derive(Debug, Clone)]
enum FXArray {
    Dual(Array2<Dual>),
    Dual2(Array2<Dual2>),
}

#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone)]
pub struct FXRates {
    pub(crate) fx_rates: Vec<FXRate>,
    currencies: IndexSet<Ccy>,
    pub(crate) fx_vector: FXVector,
    fx_array: FXArray,
    base: Ccy,
    pub(crate) ad: u8,
    // settlement : Option<NaiveDateTime>,
    // pairs : Vec<(Ccy, Ccy)>,
}

impl FXRates {
    pub fn try_new(
        fx_rates: Vec<FXRate>,
        settlement: NaiveDateTime,
        base: Option<Ccy>,
    ) -> Result<Self, PyErr> {
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
        for fxr in fx_rates.iter() {
            currencies.insert(fxr.pair.0);
            currencies.insert(fxr.pair.1);
        }
        let q = currencies.len();

        // 2.
        if q > (fx_rates.len() + 1) {
            return Err(PyValueError::new_err("`fx_rates` is underspecified."));
        } else if q < (fx_rates.len() + 1) {
            return Err(PyValueError::new_err("`fx_rates` is overspecified."));
        }

        // 3.
        if !(&fx_rates
            .iter()
            .all(|d| d.settlement.map_or(true, |v| v == settlement)))
        {
            return Err(PyValueError::new_err(
                "`fx_rates` must have consistent `settlement` dates across all rates.",
            ));
        }

        // aggregate information from FXPairs
        // let pairs: Vec<(Ccy, Ccy)> = fx_rates.iter().map(|fxp| fxp.pair).collect();
        //         let variables: Vec<String> = fx_rates.iter().map(
        //                 |fxp| "fx_".to_string() + &fxp.pair.0.to_string() + &fxp.pair.1.to_string()
        //             ).collect();

        let base = base.unwrap_or(currencies[0]);

        let mut a: Array2<Dual> = Array2::zeros((q, q));
        a[[0, 0]] = Dual::one();
        let mut b: Array1<Dual> = Array1::zeros(q);
        b[0] = Dual::one();

        for (i, fxr) in fx_rates.iter().enumerate() {
            let dom_idx = currencies.get_index_of(&(fxr.pair.0)).expect("pre checked");
            let for_idx = currencies.get_index_of(&(fxr.pair.1)).expect("pre checked");
            a[[i + 1, dom_idx]] = -1.0 * Dual::one();
            match &fxr.rate {
                DualsOrF64::F64(f) => {
                    let var = "fx_".to_string() + &format!("{}", fxr.pair).to_string();
                    a[[i + 1, for_idx]] = 1.0 / Dual::new(*f, vec![var]);
                }
                DualsOrF64::Dual(d) => {
                    a[[i + 1, for_idx]] = 1.0 / d;
                }
                _ => {
                    // 4. (validation)
                    return Err(PyValueError::new_err(
                        "`fx_rates` cannot be constructed with rates as Dual2 data types.",
                    ))
                }
            }
        }
        let x = dsolve(&a.view(), &b.view(), false);
        let y = Array1::from_iter(x.iter().map(|x| 1_f64 / x));
        let z = douter11_(&x.view(), &y.view());

        Ok(FXRates {
            fx_rates,
            fx_vector: FXVector::Dual(x),
            fx_array: FXArray::Dual(z),
            currencies,
            // pairs: pairs,
            base,
            ad: 1_u8,
            // settlement: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::calendar::ndt;

    #[test]
    fn ccy_creation() {
        let a = Ccy::try_new("usd").unwrap();
        let b = Ccy::try_new("USD").unwrap();
        assert_eq!(a, b)
    }

    #[test]
    fn ccy_creation_error() {
        match Ccy::try_new("FOUR") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    #[test]
    fn pair_creation() {
        let a = FXPair::try_new("usd", "eur").unwrap();
        let b = FXPair::try_new("USD", "EUR").unwrap();
        assert_eq!(a, b)
    }

    #[test]
    fn pair_creation_error() {
        match FXPair::try_new("usd", "USD") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    #[test]
    fn rate_creation() {
        FXRate::try_new("usd", "eur", DualsOrF64::F64(1.20), None).unwrap();
    }

    #[test]
    fn fxrates_new() {
        FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", DualsOrF64::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", DualsOrF64::F64(110.0), None).unwrap(),
            ],
            ndt(2004, 1, 1),
            None,
        ).unwrap();
    }
}
