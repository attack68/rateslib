use crate::dual::dual::{Dual, Dual2};

impl From<Dual> for f64 {
    fn from(value: Dual) -> Self {
        value.real
    }
}

impl From<Dual2> for f64 {
    fn from(value: Dual2) -> Self {
        value.real
    }
}

impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Self.new(value, vec![])
    }
}

impl From<Dual2> for Dual {
    fn from(value: Dual2) -> Self {
        Dual {
            real: value.real,
            vars: value.vars.clone(),
            dual: value.dual,
        }
    }
}

impl From<f64> for Dual2 {
    fn from(value: f64) -> Self {
        Self.new(value, vec![])
    }
}

impl From<Dual> for Dual2 {
    fn from(value: Dual) -> Self {
        let n = value.dual.len_of(Axis(0));
        Dual2 {
            real: value.real,
            vars: value.vars.clone(),
            dual: value.dual,
            dual2: Array2::zeros((n, n)),
        }
    }
}