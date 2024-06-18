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
pub struct FXCross {
    name: (Ccy, Ccy)
}

impl FXCross {
    /// Constructs a new `FXCross`, as a combination of two constructed `Ccy`s. Panics if currencies are the same.
    pub fn new(lhs: &str, rhs: &str) -> Self {
        let lhs_ = Ccy::new(lhs);
        let rhs_ = Ccy::new(rhs);
        assert_ne!(lhs_, rhs_);
        FXCross { name: (lhs_, rhs_) }
    }
}

impl fmt::Display for FXCross {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.name.0.name, self.name.1.name)
    }
}

/// Struct to define an FXRate via FXCross, rate and settlement info.
#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone)]
pub struct FXRate {
    pub(crate) pair: FXCross,
    pub(crate) rate: DualsOrF64,
    pub(crate) settlement: Option<NaiveDateTime>,
}

impl FXRate {
    pub fn new(lhs: &str, rhs: &str, rate: DualsOrF64, settlement: Option<NaiveDateTime>) -> Self {
        FXRate {pair: FXCross::new(lhs, rhs), rate, settlement}
    }
}

// #[derive(Debug, Clone)]
// pub struct FXRates {
//     fx_rates : Vec<FXPair>,
//     currencies: IndexSet<Ccy>,
//     fx_array : Array1<Dual>,
//     pairs : Vec<(Ccy, Ccy)>,
//     base : Ccy,
//     settlement : Option<f64>,
// }
//
// impl FXRates {
//
//     pub fn new(fx_rates: Vec<FXPair>, base: Option<Ccy>) -> Self {
//         let mut currencies: IndexSet<Ccy> = IndexSet::with_capacity(fx_rates.len() + 1_usize);
//         for fxr in fx_rates.iter() {
//             currencies.insert(fxr.pair.0);
//             currencies.insert(fxr.pair.1);
//         }
//         let q = currencies.len();
//
//         // validate data
//         if q > (fx_rates.len() + 1) {
//             panic!("`fx_rates` is underspecified. {q}")
//         } else if q < (fx_rates.len() + 1) {
//             panic!("`fx_rates` is overspecified.")
//         }
//
//         // validations
//         // 1. settlements is same on all or None
//
//         // aggregate information from FXPairs
//         let pairs: Vec<(Ccy, Ccy)> = fx_rates.iter().map(|fxp| fxp.pair).collect();
//         let variables: Vec<String> = fx_rates.iter().map(
//                 |fxp| "fx_".to_string() + &fxp.pair.0.to_string() + &fxp.pair.1.to_string()
//             ).collect();
//
//         let base = base.unwrap_or(currencies[0]);
//
//
//         let mut a: Array2<Dual> = Array2::zeros((q, q));
//         a[[0, 0]] = Dual::one();
//         let mut b: Array1<Dual> = Array1::zeros(q);
//         b[0] = Dual::one();
//
//         for (i, fxr) in fx_rates.iter().enumerate() {
//             let dom_idx = currencies.get_index_of(&fxr.pair.0).expect("pre checked");
//             let for_idx = currencies.get_index_of(&fxr.pair.1).expect("pre checked");
//             a[[i+1, dom_idx]] = -1.0 * Dual::one();
//             a[[i+1, for_idx]] = 1.0 / Dual::new(fxr.rate, vec![variables[i].clone()]);
//         }
//         let x = dsolve(&a.view(), &b.view(), false);
//
//         FXRates {
//             fx_rates: fx_rates,
//             fx_array: x,
//             currencies: currencies,
//             pairs: pairs,
//             base: base,
//             settlement: None,
//         }
//     }
// }

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
    fn cross_creation() {
        let a = FXCross::new("usd", "eur");
        let b = FXCross::new("USD", "EUR");
        assert_eq!(a, b)
    }

    #[test]
    #[should_panic]
    fn cross_failure() {
        FXCross::new("usd", "USD");
    }

    #[test]
    fn rate_creation() {
        FXRate::new("usd", "eur", DualsOrF64::F64(1.20), None);
    }


//     #[test]
//     fn ccy_strings() {
//         let a = Ccy::usd;
//         let b = a.to_string();
//         assert_eq!("usd", b);
//     }
//
//     #[test]
//     fn fxrates_new() {
//         let fxr = FXRates::new(
//             vec![
//                 FXPair::new((Ccy::eur, Ccy::usd), 1.08, None),
//                 FXPair::new((Ccy::usd, Ccy::jpy), 110.0, None),
//                 ],
//             None,
//         );
//
//         println!("{:?}", fxr);
//         assert!(1 == 2);
//     }
}