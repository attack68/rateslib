use crate::fx::Ccy;
use crate::dual::dual1::Dual;
use num_traits::identities::{One, Zero};
use crate::dual::linalg::dsolve;
use indexmap::set::IndexSet;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct FXPair {
    pair: (Ccy, Ccy),
    rate: f64,
    settlement: Option<f64>,
}

impl FXPair {
    pub fn new(pair: (Ccy, Ccy), rate: f64, settlement: Option<f64>) -> Self {
        assert!(pair.0 != pair.1);
        FXPair {pair, rate, settlement}
    }
}

#[derive(Debug, Clone)]
pub struct FXRates {
    fx_rates : Vec<FXPair>,
    currencies: IndexSet<Ccy>,
    fx_array : Array1<Dual>,
    pairs : Vec<(Ccy, Ccy)>,
    base : Ccy,
    settlement : Option<f64>,
}

impl FXRates {

    pub fn new(fx_rates: Vec<FXPair>, base: Option<Ccy>) -> Self {
        let mut currencies: IndexSet<Ccy> = IndexSet::with_capacity(fx_rates.len() + 1_usize);
        for fxr in fx_rates.iter() {
            currencies.insert(fxr.pair.0);
            currencies.insert(fxr.pair.1);
        }
        let q = currencies.len();

        // validate data
        if q > (fx_rates.len() + 1) {
            panic!("`fx_rates` is underspecified. {q}")
        } else if q < (fx_rates.len() + 1) {
            panic!("`fx_rates` is overspecified.")
        }

        // validations
        // 1. settlements is same on all or None

        // aggregate information from FXPairs
        let pairs: Vec<(Ccy, Ccy)> = fx_rates.iter().map(|fxp| fxp.pair).collect();
        let variables: Vec<String> = fx_rates.iter().map(
                |fxp| "fx_".to_string() + &fxp.pair.0.to_string() + &fxp.pair.1.to_string()
            ).collect();

        let base = base.unwrap_or(currencies[0]);


        let mut a: Array2<Dual> = Array2::zeros((q, q));
        a[[0, 0]] = Dual::one();
        let mut b: Array1<Dual> = Array1::zeros(q);
        b[0] = Dual::one();

        for (i, fxr) in fx_rates.iter().enumerate() {
            let dom_idx = currencies.get_index_of(&fxr.pair.0).expect("pre checked");
            let for_idx = currencies.get_index_of(&fxr.pair.1).expect("pre checked");
            a[[i+1, dom_idx]] = -1.0 * Dual::one();
            a[[i+1, for_idx]] = 1.0 / Dual::new(fxr.rate, vec![variables[i].clone()], vec![]);
        }
        let x = dsolve(&a.view(), &b.view(), false);

        FXRates {
            fx_rates: fx_rates,
            fx_array: x,
            currencies: currencies,
            pairs: pairs,
            base: base,
            settlement: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccy_strings() {
        let a = Ccy::usd;
        let b = a.to_string();
        assert_eq!("usd", b);
    }

    #[test]
    fn fxrates_new() {
        let fxr = FXRates::new(
            vec![
                FXPair::new((Ccy::eur, Ccy::usd), 1.08, None),
                FXPair::new((Ccy::usd, Ccy::jpy), 110.0, None),
                ],
            None,
        );

        println!("{:?}", fxr);
        assert!(1 == 2);

    }
}