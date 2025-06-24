use crate::dual::dual::{Dual, Dual2};
use crate::dual::enums::Number;
use auto_ops::impl_op_ex;
use std::sync::Arc;

impl_op_ex!(% |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real % b, dual: a.dual.clone()} });
impl_op_ex!(% |a: &f64, b: &Dual| -> Dual { Dual::new(*a, Vec::new()) % b });
impl_op_ex!(% |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real % b, dual: a.dual.clone(), dual2: a.dual2.clone()}
});
impl_op_ex!(% |a: &f64, b: &Dual2| -> Dual2 {
    Dual2::new(*a, Vec::new()) % b }
);

// impl REM for Dual
impl_op_ex!(% |a: &Dual, b: &Dual| -> Dual {
    let d = f64::trunc(a.real / b.real);
    a - d * b
});

// impl Rem for Dual2
impl_op_ex!(% |a: &Dual2, b: &Dual2| -> Dual2 {
    let d = f64::trunc(a.real / b.real);
    a - d * b
});

// Rem for Number
impl_op_ex!(% |a: &Number, b: &Number| -> Number {
    match (a,b) {
        (Number::F64(f), Number::F64(f2)) => Number::F64(f % f2),
        (Number::F64(f), Number::Dual(d2)) => Number::Dual(f % d2),
        (Number::F64(f), Number::Dual2(d2)) => Number::Dual2(f % d2),
        (Number::Dual(d), Number::F64(f2)) => Number::Dual(d % f2),
        (Number::Dual(d), Number::Dual(d2)) => Number::Dual(d % d2),
        (Number::Dual(_), Number::Dual2(_)) => panic!("Cannot mix dual types: Dual % Dual2"),
        (Number::Dual2(d), Number::F64(f2)) => Number::Dual2(d % f2),
        (Number::Dual2(_), Number::Dual(_)) => panic!("Cannot mix dual types: Dual2 % Dual"),
        (Number::Dual2(d), Number::Dual2(d2)) => Number::Dual2(d % d2),
    }
});

// Rem for Number
impl_op_ex!(% |a: &Number, b: &f64| -> Number {
    match a {
        Number::F64(f) => Number::F64(f % b),
        Number::Dual(d) => Number::Dual(d % b),
        Number::Dual2(d) => Number::Dual2(d % b),
    }
});

// Rem for Number
impl_op_ex!(% |a: &f64, b: &Number| -> Number {
    match b {
        Number::F64(f) => Number::F64(a % f),
        Number::Dual(d) => Number::Dual(a % d),
        Number::Dual2(d) => Number::Dual2(a % d),
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rem_() {
        let d1 = Dual::try_new(10.0, vec!["x".to_string()], vec![2.0]).unwrap();
        let d2 = Dual::new(3.0, vec!["x".to_string()]);
        let result = d1 % d2;
        let expected = Dual::try_new(1.0, vec!["x".to_string()], vec![-1.0]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn rem_f64_() {
        let d1 = Dual::try_new(10.0, vec!["x".to_string()], vec![2.0]).unwrap();
        let result = &d1 % 3.0_f64;
        assert_eq!(
            result,
            Dual::try_new(1.0, vec!["x".to_string()], vec![2.0]).unwrap()
        );

        let result = 11.0_f64 % d1;
        assert_eq!(
            result,
            Dual::try_new(1.0, vec!["x".to_string()], vec![-2.0]).unwrap()
        );
    }

    #[test]
    fn rem_2() {
        let d1 = Dual2::try_new(10.0, vec!["x".to_string()], vec![2.0], vec![]).unwrap();
        let d2 = Dual2::new(3.0, vec!["x".to_string()]);
        let result = d1 % d2;
        let expected = Dual2::try_new(1.0, vec!["x".to_string()], vec![-1.0], vec![]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn rem_f64_2() {
        let d1 = Dual2::try_new(10.0, vec!["x".to_string()], vec![2.0], vec![]).unwrap();
        let result = &d1 % 3.0_f64;
        assert_eq!(
            result,
            Dual2::try_new(1.0, vec!["x".to_string()], vec![2.0], vec![]).unwrap()
        );

        let result = 11.0_f64 % d1;
        assert_eq!(
            result,
            Dual2::try_new(1.0, vec!["x".to_string()], vec![-2.0], vec![]).unwrap()
        );
    }

    #[test]
    fn test_enum() {
        let f = Number::F64(4.0);
        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        assert_eq!(
            &f % &d,
            Number::Dual(Dual::try_new(1.0, vec!["x".to_string()], vec![-1.0]).unwrap())
        );

        assert_eq!(
            &d % &d,
            Number::Dual(Dual::try_new(0.0, vec!["x".to_string()], vec![0.0]).unwrap())
        );
    }

    #[test]
    #[should_panic]
    fn test_enum_panic() {
        let d = Number::Dual2(Dual2::new(2.0, vec!["y".to_string()]));
        let d2 = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let _ = d % d2;
    }

    #[test]
    fn test_enum_f64() {
        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let res = 2.0_f64 % d;
        assert_eq!(
            res,
            Number::Dual(2.0 % Dual::new(3.0, vec!["x".to_string()]))
        );

        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let res = d % 2.0_f64;
        assert_eq!(
            res,
            Number::Dual(Dual::new(3.0, vec!["x".to_string()]) % 2.0)
        );
    }
}
