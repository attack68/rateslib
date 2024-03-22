use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ccy {
    usd,
    eur,
    gbp,
    jpy,
    sek,
    nok,
    cad,
    chf,
}

impl fmt::Display for Ccy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub mod fx_rates;