use crate::dual::dual::{Dual, Dual2, Number};
use auto_ops::impl_op;
use std::sync::Arc;

impl_op!(-|a: Dual| -> Dual {
    Dual {
        vars: a.vars,
        real: -a.real,
        dual: -a.dual,
    }
});
impl_op!(-|a: &Dual| -> Dual {
    Dual {
        vars: Arc::clone(&a.vars),
        real: -a.real,
        dual: &a.dual * -1.0,
    }
});

impl_op!(-|a: Dual2| -> Dual2 {
    Dual2 {
        vars: a.vars,
        real: -a.real,
        dual: -a.dual,
        dual2: -a.dual2,
    }
});

impl_op!(-|a: &Dual2| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&a.vars),
        real: -a.real,
        dual: &a.dual * -1.0,
        dual2: &a.dual2 * -1.0,
    }
});

// Neg for Number
impl_op!(-|a: &Number| -> Number {
    match a {
        Number::F64(f) => Number::F64(-f),
        Number::Dual(d) => Number::Dual(-d),
        Number::Dual2(d) => Number::Dual2(-d),
    }
});

// Neg for Number
impl_op!(-|a: Number| -> Number {
    match a {
        Number::F64(f) => Number::F64(-f),
        Number::Dual(d) => Number::Dual(-d),
        Number::Dual2(d) => Number::Dual2(-d),
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn negate() {
        let d = Dual::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([2., -1.4]),
        )
        .unwrap();
        let d2 = -d.clone();
        assert!(d2.real == -2.3);
        assert!(Arc::ptr_eq(&d.vars, &d2.vars));
        assert!(d2.dual[0] == -2.0);
        assert!(d2.dual[1] == 1.4);
    }

    #[test]
    fn neg_ref() {
        let d1 =
            Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.1, 2.2]).unwrap();
        let d2 = -&d1;
        assert_eq!(d2.real, -2.5);
        assert_eq!(d2.dual, Array1::from_vec(vec![-1.1, -2.2]));
    }

    #[test]
    fn negate2() {
        let d = Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([2., -1.4]),
            Vec::from([1.0, -1.0, -1.0, 2.0]),
        )
        .unwrap();
        let d2 = -d.clone();
        assert!(d2.real == -2.3);
        assert!(Arc::ptr_eq(&d.vars, &d2.vars));
        assert!(d2.dual[0] == -2.0);
        assert!(d2.dual[1] == 1.4);
        assert!(d2.dual2[[1, 0]] == 1.0);
    }

    #[test]
    fn negate_ref2() {
        let d = Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([2., -1.4]),
            Vec::from([1.0, -1.0, -1.0, 2.0]),
        )
        .unwrap();
        let d2 = -&d;
        assert!(d2.real == -2.3);
        assert!(Arc::ptr_eq(&d.vars, &d2.vars));
        assert!(d2.dual[0] == -2.0);
        assert!(d2.dual[1] == 1.4);
        assert!(d2.dual2[[1, 0]] == 1.0);
    }

    #[test]
    fn test_enum() {
        let f = Number::F64(2.0);
        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        assert_eq!(-f, Number::F64(-2.0));
        assert_eq!(
            -d,
            Number::Dual(Dual::try_new(-3.0, vec!["x".to_string()], vec![-1.0]).unwrap())
        );
    }
}
