use std::convert::From;
use crate::dual::dual::{DualsOrF64, ADOrder, Dual, Dual2};

/// Convert a `DualsOrF64` of one `ADOrder` to another.
///
/// **Why use this method?**
///
/// `From` is implemented to convert into all the types `f64`, `Dual` and `Dual2` from each other,
/// however, when doing so there is no variable information attached for an `f64`. For example,
///
/// ```rust
/// let d = Dual::from(2.5_f64);
/// assert_eq!(d, Dual::new(2.5_f64, vec![])); // true
/// ```
///
/// On the other hand using a `DualsOrF64` enum can convert any value to a dual data type tagged
/// with specific variables. `vars` are only used in this instance when converting an `f64` to a
/// dual type.
///
/// ```rust
/// let f_ = DualsOrF64::F64(2.5_f64);
/// let d_ = set_order_with_conversion(f_, ADOrder::One, vec!["x".to_string()]);
/// let d = Dual::from(d_);
/// assert_eq!(d, Dual::new(2.5_f64, vec!["x".to_string()])); // true
/// ```
///
pub fn set_order_with_conversion(value: &DualsOrF64, order: ADOrder, vars: Vec<String>) -> DualsOrF64 {
    match (value, order) {
        (DualsOrF64::F64(f), ADOrder::Zero) => { DualsOrF64::F64(*f) }
        (DualsOrF64::Dual(d), ADOrder::Zero) => { DualsOrF64::F64(d.real) }
        (DualsOrF64::Dual2(d), ADOrder::Zero) => { DualsOrF64::F64(d.real) }
        (DualsOrF64::F64(f), ADOrder::One) => { DualsOrF64::Dual(Dual::new(*f, vars)) }
        (DualsOrF64::Dual(d), ADOrder::One) => { DualsOrF64::Dual(d.clone())}
        (DualsOrF64::Dual2(d), ADOrder::One) => { DualsOrF64::Dual(Dual::from(d)) }
        (DualsOrF64::F64(f), ADOrder::Two) => { DualsOrF64::Dual2(Dual2::new(*f, vars)) }
        (DualsOrF64::Dual(d), ADOrder::Two) => { DualsOrF64::Dual2(Dual2::from(d)) }
        (DualsOrF64::Dual2(d), ADOrder::Two) => { DualsOrF64::Dual2(d.clone()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_order_with_conversion() {
        let f = 2.5_f64;
        let d = set_order_with_conversion(DualsOrF64::F64(f), ADOrder::One, vec!["var1".to_string()]);
        assert_eq!(d, DualsOrF64::Dual(Dual::new(2.5, vec!["var1".to_string()])));

        let d2 = set_order_with_conversion(d, ADOrder::Two, vec![]);
        assert_eq!(d2, DualsOrF64::Dual2(Dual2::new(2.5, vec!["var1".to_string()])));

        let f = set_order_with_conversion(d2, ADOrder::Zero, vec![]);
        assert_eq!(f, DualsOrF64::F64(2.5_f64));
    }

    #[test]
    fn test_docstring() {
        let f_ = DualsOrF64::F64(2.5_f64);
        let d_ = set_order_with_conversion(f_, ADOrder::One, vec!["x".to_string()]);
        let d = Dual::from(d_);
        assert_eq!(d, Dual::new(2.5_f64, vec!["x".to_string()])); // true
    }
}