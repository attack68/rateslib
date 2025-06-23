use crate::dual::dual::{Dual, Dual2, Number};
use ndarray::{Array2, Axis};

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

impl From<&Dual> for f64 {
    fn from(value: &Dual) -> Self {
        value.real
    }
}

impl From<&Dual2> for f64 {
    fn from(value: &Dual2) -> Self {
        value.real
    }
}

impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Self::new(value, vec![])
    }
}

// impl From<Dual> for Dual {
//     fn from(value: Dual) -> Self {
//         value.clone()
//     }
// }

impl From<Dual2> for Dual {
    fn from(value: Dual2) -> Self {
        Dual {
            real: value.real,
            vars: value.vars.clone(),
            dual: value.dual,
        }
    }
}

impl From<&Dual2> for Dual {
    fn from(value: &Dual2) -> Self {
        Dual {
            real: value.real,
            vars: value.vars.clone(),
            dual: value.dual.clone(),
        }
    }
}

impl From<f64> for Dual2 {
    fn from(value: f64) -> Self {
        Self::new(value, vec![])
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

impl From<&Dual> for Dual2 {
    fn from(value: &Dual) -> Self {
        let n = value.dual.len_of(Axis(0));
        Dual2 {
            real: value.real,
            vars: value.vars.clone(),
            dual: value.dual.clone(),
            dual2: Array2::zeros((n, n)),
        }
    }
}

impl From<Number> for f64 {
    fn from(value: Number) -> Self {
        match value {
            Number::F64(f) => f,
            Number::Dual(d) => d.real,
            Number::Dual2(d) => d.real,
        }
    }
}

impl From<Number> for Dual {
    fn from(value: Number) -> Self {
        match value {
            Number::F64(f) => Dual::new(f, vec![]),
            Number::Dual(d) => d,
            Number::Dual2(d) => Dual::from(d),
        }
    }
}

impl From<Number> for Dual2 {
    fn from(value: Number) -> Self {
        match value {
            Number::F64(f) => Dual2::new(f, vec![]),
            Number::Dual(d) => Dual2::from(d),
            Number::Dual2(d) => d,
        }
    }
}

impl From<&Number> for f64 {
    fn from(value: &Number) -> Self {
        match value {
            Number::F64(f) => *f,
            Number::Dual(d) => d.real,
            Number::Dual2(d) => d.real,
        }
    }
}

impl From<&Number> for Dual {
    fn from(value: &Number) -> Self {
        match value {
            Number::F64(f) => Dual::new(*f, vec![]),
            Number::Dual(d) => d.clone(),
            Number::Dual2(d) => Dual::from(d),
        }
    }
}

impl From<&Number> for Dual2 {
    fn from(value: &Number) -> Self {
        match value {
            Number::F64(f) => Dual2::new(*f, vec![]),
            Number::Dual(d) => Dual2::from(d),
            Number::Dual2(d) => d.clone(),
        }
    }
}

// impl From<Dual2> for Dual2 {
//     fn from(value: Dual2) -> Self {
//         value.clone()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_dual_into_dual2() {
        let d1 = Dual::new(2.5, vec!["x".to_string(), "y".to_string()]);
        let d2: Dual2 = d1.into();
        let result = Dual2::new(2.5, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(d2, result);
    }

    #[test]
    fn from_dual_into_f64() {
        let d1 = Dual::new(2.5, vec!["x".to_string(), "y".to_string()]);
        let d2: f64 = d1.into();
        let result = 2.5_f64;
        assert_eq!(d2, result);
    }

    // #[test]
    // fn from_dual_to_dual_unchanged() {
    //     let d1 = Dual::new(2.5, vec!["x".to_string(), "y".to_string()]);
    //     let d2: Dual = Dual::from(d1);
    //     assert_eq!(d2, d1);
    // }

    #[test]
    fn from_f64_into_dual() {
        let d = Dual::from(4.0_f64);
        assert_eq!(d, Dual::new(4.0_f64, vec![]));
    }

    #[test]
    fn from_dual2_into_dual() {
        let d2: Dual2 = Dual2::new(2.0, vec!["x".to_string(), "y".to_string()]);
        let d1: Dual = d2.into();
        let result = Dual::new(2.0, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(d1, result);
    }
}
