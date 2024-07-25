use std::convert::From;
use crate::dual::dual::{DualsOrF64, ADOrder, Dual, Dual2};

/// Convert an enum of one ADOrder to another, including f64 conversion, and maintaining Vars efficiencies.
///
/// `vars` are only used when upcasting an f64 to a dual type and variables names are required.
pub fn set_order_with_convert(value: DualsOrF64, order: ADOrder, vars: Vec<String>) -> DualsOrF64 {
    match (value, order) {
        (DualsOrF64::F64(f), ADOrder::Zero) => { DualsOrF64::F64(f) }
        (DualsOrF64::Dual(d), ADOrder::Zero) => { DualsOrF64::F64(d.real) }
        (DualsOrF64::Dual2(d), ADOrder::Zero) => { DualsOrF64::F64(d.real) }
        (DualsOrF64::F64(f), ADOrder::One) => { DualsOrF64::Dual(Dual::new(f, vars)) }
        (DualsOrF64::Dual(d), ADOrder::One) => { DualsOrF64::Dual(d)}
        (DualsOrF64::Dual2(d), ADOrder::One) => { DualsOrF64::Dual(Dual::from(d)) }
        (DualsOrF64::F64(f), ADOrder::Two) => { DualsOrF64::Dual2(Dual2::new(f, vars)) }
        (DualsOrF64::Dual(d), ADOrder::Two) => { DualsOrF64::Dual2(Dual2::from(d)) }
        (DualsOrF64::Dual2(d), ADOrder::Two) => { DualsOrF64::Dual2(d) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_order_with_convert() {
        let f = 2.5_f64;
        let d = set_order_with_convert(DualsOrF64::F64(f), ADOrder::One, vec!["var1".to_string()]);
        assert_eq!(d, DualsOrF64::Dual(Dual::new(2.5, vec!["var1".to_string()])));

        let d2 = set_order_with_convert(d, ADOrder::Two, vec![]);
        assert_eq!(d2, DualsOrF64::Dual2(Dual2::new(2.5, vec!["var1".to_string()])));

        let f = set_order_with_convert(d2, ADOrder::Zero, vec![]);
        assert_eq!(f, DualsOrF64::F64(2.5_f64));
    }
}