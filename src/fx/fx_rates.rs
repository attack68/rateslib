use crate::fx::Ccy;
use crate::dual::dual1::Dual;
use indexmap::set::IndexSet;

#[derive(Debug, Clone)]
pub struct FXPair {
    pair: (Ccy, Ccy),
    rate: f64,
    settlement: Option<f64>,
}

impl FXPair {
    pub fn new(pair: (Ccy, Ccy), rate: f64, settlement: Option<f64>) -> Self {
        assert!(pair.0 != pair.1)
        FXPair {pair, rate, settlement}
    }
}

#[derive(Debug, Clone)]
pub struct FXRatesDual {
    fx_rates : Vec<FXPair>,
    fx_array : Vec<(Ccy, Dual)>,
    pairs : Vec<(Ccy, Ccy)>,
    base : Ccy,
    settlement : Option<f64>,
}

impl FXRatesDual {

    pub fn new(fx_rates: Vec<FXPair>, base: Option<Ccy>) -> Self {

        // aggregate information from FXPairs
        let pairs: Vec<(Ccy, Ccy)> = fx_rates.iter().map(|fxp| fxp.pair).collect();
        let variables: Vec<String> = fx_rates.iter().map(
                |fxp| "fx_".to_string() + &fxp.pair.0.to_string() + &fxp.pair.1.to_string()
            ).collect();
        let mut currencies: IndexSet<Ccy> = IndexSet::with_capacity(fx_rates.len() + 1_usize);
        for fxr in fx_rates.iter() {
            currencies.insert(fxr.pair.0);
            currencies.insert(fxr.pair.1);
        }
        let base = base.unwrap_or(currencies[0]);

        // validations
        // 1. settlements is same on all or None


        FXRatesDual {
            fx_rates: fx_rates,
            fx_array: Vec::new(),
            pairs: pairs,
            base: Ccy::usd,
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
        let fxr = FXRatesDual::new(
            vec![
                FXPair::new((Ccy::eur, Ccy::usd), 1.08, None),
                FXPair::new((Ccy::usd, Ccy::eur), 1.08, None),
                ],
            None,
        );

        println!("{:?}", fxr);
        assert!(1 == 2);

    }
}