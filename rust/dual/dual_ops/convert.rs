use crate::dual::dual::{ADOrder, Dual, Dual2, Number};
use std::convert::From;

/// Convert a `Number` of one `ADOrder` to another and consume the value.
///
/// **Why use this method?**
///
/// `From` is implemented to convert into all the types `f64`, `Dual` and `Dual2` from each other,
/// however, when doing so there is no variable information attached for an `f64`. For example,
///
/// ```rust
/// # use rateslib::dual::Dual;
/// let d = Dual::from(2.5_f64);
/// assert_eq!(d, Dual::new(2.5_f64, vec![]));
/// ```
///
/// On the other hand using a `Number` enum can convert any value to a dual data type tagged
/// with specific variables. `vars` are only used in this instance when converting an `f64` to a
/// dual type.
///
/// ```rust
/// # use rateslib::dual::{Number, set_order, ADOrder, Dual};
/// let f_ = Number::F64(2.5_f64);
/// let d_ = set_order(f_, ADOrder::One, vec!["x".to_string()]);
/// let d = Dual::from(d_);
/// assert_eq!(d, Dual::new(2.5_f64, vec!["x".to_string()]));
/// ```
///
pub fn set_order(value: Number, order: ADOrder, vars: Vec<String>) -> Number {
    match (value, order) {
        (Number::F64(f), ADOrder::Zero) => Number::F64(f),
        (Number::Dual(d), ADOrder::Zero) => Number::F64(d.real),
        (Number::Dual2(d), ADOrder::Zero) => Number::F64(d.real),
        (Number::F64(f), ADOrder::One) => Number::Dual(Dual::new(f, vars)),
        (Number::Dual(d), ADOrder::One) => Number::Dual(d),
        (Number::Dual2(d), ADOrder::One) => Number::Dual(Dual::from(d)),
        (Number::F64(f), ADOrder::Two) => Number::Dual2(Dual2::new(f, vars)),
        (Number::Dual(d), ADOrder::Two) => Number::Dual2(Dual2::from(d)),
        (Number::Dual2(d), ADOrder::Two) => Number::Dual2(d),
    }
}

/// Convert a `Number` of one `ADOrder` to another.
///
/// Similar to `set_order` except the value is not consumed during conversion.
pub fn set_order_clone(value: &Number, order: ADOrder, vars: Vec<String>) -> Number {
    match (value, order) {
        (Number::F64(f), ADOrder::Zero) => Number::F64(*f),
        (Number::Dual(d), ADOrder::Zero) => Number::F64(d.real),
        (Number::Dual2(d), ADOrder::Zero) => Number::F64(d.real),
        (Number::F64(f), ADOrder::One) => Number::Dual(Dual::new(*f, vars)),
        (Number::Dual(d), ADOrder::One) => Number::Dual(d.clone()),
        (Number::Dual2(d), ADOrder::One) => Number::Dual(Dual::from(d)),
        (Number::F64(f), ADOrder::Two) => Number::Dual2(Dual2::new(*f, vars)),
        (Number::Dual(d), ADOrder::Two) => Number::Dual2(Dual2::from(d)),
        (Number::Dual2(d), ADOrder::Two) => Number::Dual2(d.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_order_with_conversion() {
        let f = 2.5_f64;
        let d = set_order(Number::F64(f), ADOrder::One, vec!["var1".to_string()]);
        assert_eq!(d, Number::Dual(Dual::new(2.5, vec!["var1".to_string()])));

        let d2 = set_order(d, ADOrder::Two, vec![]);
        assert_eq!(d2, Number::Dual2(Dual2::new(2.5, vec!["var1".to_string()])));

        let f = set_order(d2, ADOrder::Zero, vec![]);
        assert_eq!(f, Number::F64(2.5_f64));
    }

    #[test]
    fn test_docstring() {
        let f_ = Number::F64(2.5_f64);
        let d_ = set_order(f_, ADOrder::One, vec!["x".to_string()]);
        let d = Dual::from(d_);
        assert_eq!(d, Dual::new(2.5_f64, vec!["x".to_string()])); // true
    }
}
