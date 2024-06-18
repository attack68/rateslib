use crate::dual::dual1::Dual;
use crate::dual::dual_py::DualsOrF64;
use num_traits::identities::{One, Zero};
use crate::dual::linalg::dsolve;
use indexmap::set::IndexSet;
use ndarray::{Array1, Array2};
use std::fmt;
use internment::Intern;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use chrono::prelude::*;

/// Struct to define a currency.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ccy {
    name: Intern<String>
}

impl Ccy {
    /// Constructs a new `Ccy`, with `name` converted to lowercase. Panics if not 3-digit.
    pub fn new(name: &str) -> Self {
        let ccy: String = name.to_string().to_lowercase();
        assert!(ccy.len() == 3);
        Ccy {name: Intern::new(ccy)}
    }
}

/// Struct to define a currency cross.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FXPair (Ccy, Ccy);

impl FXPair {
    /// Constructs a new `FXCross`, as a combination of two constructed `Ccy`s. Panics if currencies are the same.
    pub fn new(lhs: &str, rhs: &str) -> Self {
        let lhs_ = Ccy::new(lhs);
        let rhs_ = Ccy::new(rhs);
        assert_ne!(lhs_, rhs_);
        FXPair(lhs_, rhs_)
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
    pub fn new(lhs: &str, rhs: &str, rate: DualsOrF64, settlement: Option<NaiveDateTime>) -> Self {
        let ad = match rate {
            DualsOrF64::F64(_) => 0_u8,
            DualsOrF64::Dual(_) => 1_u8,
            DualsOrF64::Dual2(_) => 2_u8,
        };
        FXRate {pair: FXPair::new(lhs, rhs), rate, settlement, ad}
    }
}

#[derive(Debug, Clone)]
pub struct FXRates {
    fx_rates : Vec<FXRate>,
    currencies: IndexSet<Ccy>,
    fx_array : Array1<Dual>,
    // pairs : Vec<(Ccy, Ccy)>,
    base : Ccy,
    // settlement : Option<NaiveDateTime>,
}

impl FXRates {

    pub fn new(fx_rates: Vec<FXRate>, base: Option<Ccy>) -> Self {
        // Validations:
        // 1. fx_rates is non-zero length
        // 2. currencies are not under or over overspecified
        // 3. settlement dates are all consistent.
        // 4. No Dual2 data types are provided as input

        // 1.
        assert!(fx_rates.len() > 0);

        let mut currencies: IndexSet<Ccy> = IndexSet::with_capacity(fx_rates.len() + 1_usize);
        for fxr in fx_rates.iter() {
            currencies.insert(fxr.pair.0);
            currencies.insert(fxr.pair.1);
        }
        let q = currencies.len();

        // 2.
        if q > (fx_rates.len() + 1) {
            panic!("`fx_rates` is underspecified. {q}")
        } else if q < (fx_rates.len() + 1) {
            panic!("`fx_rates` is overspecified.")
        }

        // 3.
        let s = fx_rates[0].settlement;
        assert!(&fx_rates[1..].iter().all(|d| d.settlement == s));

        // 4.
        assert!(!fx_rates.iter().any(|v| v.ad == 2_u8));

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
            a[[i+1, dom_idx]] = -1.0 * Dual::one();
            match &fxr.rate {
                DualsOrF64::F64(f) => {
                    let var = "fx_".to_string() + &format!("{}", fxr.pair).to_string();
                    a[[i+1, for_idx]] = 1.0 / Dual::new(*f, vec![var]);
                },
                DualsOrF64::Dual(d) => {
                    a[[i+1, for_idx]] = 1.0 / d;
                },
                _ => panic!("`FXRates` must be constructed with FXRate of float or Dual types only.")
            }
        }
        let x = dsolve(&a.view(), &b.view(), false);

        FXRates {
            fx_rates: fx_rates,
            fx_array: x,
            currencies: currencies,
            // pairs: pairs,
            base: base,
            // settlement: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccy_creation() {
        let a = Ccy::new("usd");
        let b = Ccy::new("USD");
        assert_eq!(a, b)
    }

    #[test]
    #[should_panic]
    fn ccy_failure() {
        Ccy::new("four");
    }

    #[test]
    fn pair_creation() {
        let a = FXPair::new("usd", "eur");
        let b = FXPair::new("USD", "EUR");
        assert_eq!(a, b)
    }

    #[test]
    #[should_panic]
    fn pair_failure() {
        FXPair::new("usd", "USD");
    }

    #[test]
    fn rate_creation() {
        FXRate::new("usd", "eur", DualsOrF64::F64(1.20), None);
    }

    #[test]
    fn fxrates_new() {
        let fxr = FXRates::new(
            vec![
                FXRate::new("eur", "usd", DualsOrF64::F64(1.08), None),
                FXRate::new("usd", "jpy", DualsOrF64::F64(110.0), None),
                ],
            None,
        );

        println!("{:?}", fxr);
        assert!(1 == 2);
    }
}