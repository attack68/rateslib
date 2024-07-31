use crate::dual::dual::{Dual, Dual2, DualsOrF64};
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

// Rem for DualsOrF64
impl_op_ex!(% |a: &DualsOrF64, b: &DualsOrF64| -> DualsOrF64 {
    match (a,b) {
        (DualsOrF64::F64(f), DualsOrF64::F64(f2)) => DualsOrF64::F64(f % f2),
        (DualsOrF64::F64(f), DualsOrF64::Dual(d2)) => DualsOrF64::Dual(f % d2),
        (DualsOrF64::F64(f), DualsOrF64::Dual2(d2)) => DualsOrF64::Dual2(f % d2),
        (DualsOrF64::Dual(d), DualsOrF64::F64(f2)) => DualsOrF64::Dual(d % f2),
        (DualsOrF64::Dual(d), DualsOrF64::Dual(d2)) => DualsOrF64::Dual(d % d2),
        (DualsOrF64::Dual(_), DualsOrF64::Dual2(_)) => panic!("Cannot mix dual types: Dual % Dual2"),
        (DualsOrF64::Dual2(d), DualsOrF64::F64(f2)) => DualsOrF64::Dual2(d % f2),
        (DualsOrF64::Dual2(_), DualsOrF64::Dual(_)) => panic!("Cannot mix dual types: Dual2 % Dual"),
        (DualsOrF64::Dual2(d), DualsOrF64::Dual2(d2)) => DualsOrF64::Dual2(d % d2),
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
        let f = DualsOrF64::F64(4.0);
        let d = DualsOrF64::Dual(Dual::new(3.0, vec!["x".to_string()]));
        assert_eq!(&f%&d, DualsOrF64::Dual(Dual::try_new(1.0, vec!["x".to_string()], vec![-1.0]).unwrap()));

        assert_eq!(&d%&d, DualsOrF64::Dual(Dual::try_new(0.0, vec!["x".to_string()], vec![0.0]).unwrap()));
    }

    #[test]
    #[should_panic]
    fn test_enum_panic() {
        let d = DualsOrF64::Dual2(Dual2::new(2.0, vec!["y".to_string()]));
        let d2 = DualsOrF64::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let _ = d % d2;
    }
}
