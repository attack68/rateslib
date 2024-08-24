use crate::dual::dual::{Dual, Dual2, Number};
use auto_ops::impl_op_ex;
use num_traits::Pow;
use std::sync::Arc;

impl_op_ex!(/ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real / b, dual: (1_f64/b) * &a.dual} });
impl_op_ex!(/ |a: &f64, b: &Dual| -> Dual { a * b.clone().pow(-1.0) });
impl_op_ex!(/ |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real / b, dual: (1_f64/b) * &a.dual, dual2: (1_f64/b) * &a.dual2}
});
impl_op_ex!(/ |a: &f64, b: &Dual2| -> Dual2 { a * b.clone().pow(-1.0) });

// impl Div for Dual
impl_op_ex!(/ |a: &Dual, b: &Dual| -> Dual {
    let b_ = Dual {real: 1.0 / b.real, vars: Arc::clone(&b.vars), dual: -1.0 / (b.real * b.real) * &b.dual};
    a * b_
});

// impl Div for Dual2
impl_op_ex!(/ |a: &Dual2, b: &Dual2| -> Dual2 { a * b.clone().pow(-1.0) });

// Div for Number
impl_op_ex!(/ |a: &Number, b: &Number| -> Number {
    match (a,b) {
        (Number::F64(f), Number::F64(f2)) => Number::F64(f / f2),
        (Number::F64(f), Number::Dual(d2)) => Number::Dual(f / d2),
        (Number::F64(f), Number::Dual2(d2)) => Number::Dual2(f / d2),
        (Number::Dual(d), Number::F64(f2)) => Number::Dual(d / f2),
        (Number::Dual(d), Number::Dual(d2)) => Number::Dual(d / d2),
        (Number::Dual(_), Number::Dual2(_)) => panic!("Cannot mix dual types: Dual / Dual2"),
        (Number::Dual2(d), Number::F64(f2)) => Number::Dual2(d / f2),
        (Number::Dual2(_), Number::Dual(_)) => panic!("Cannot mix dual types: Dual2 / Dual"),
        (Number::Dual2(d), Number::Dual2(d2)) => Number::Dual2(d / d2),
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_f64() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = d1 / 2.0;
        let expected = Dual::try_new(
            0.5,
            vec!["v0".to_string(), "v1".to_string()],
            vec![0.5, 1.0],
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn f64_div() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = 2.0 / d1.clone();
        let expected = Dual::new(2.0, vec![]) / d1;
        assert_eq!(result, expected)
    }

    #[test]
    fn div() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let d2 = Dual::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
        )
        .unwrap();
        let expected = Dual::try_new(
            0.5,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![0.5, 1.0, -0.75],
        )
        .unwrap();
        let result = d1 / d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn div_f64_2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = d1 / 2.0;
        let expected = Dual2::try_new(
            0.5,
            vec!["v0".to_string(), "v1".to_string()],
            vec![0.5, 1.0],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn f64_div2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = 2.0 / d1.clone();
        let expected = Dual2::new(2.0, vec![]) / d1;
        assert_eq!(result, expected)
    }

    #[test]
    fn div2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let d2 = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        )
        .unwrap();
        let expected = Dual2::try_new(
            0.5,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![0.5, 1.0, -0.75],
            vec![0., 0., -0.375, 0., 0., -0.75, -0.375, -0.75, 1.125],
        )
        .unwrap();
        let result = d1 / d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_enum() {
        let f = Number::F64(2.0);
        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        assert_eq!(
            &f / &d,
            Number::Dual(
                Dual::try_new(2.0 / 3.0, vec!["x".to_string()], vec![-2.0 / 9.0]).unwrap()
            )
        );

        assert_eq!(
            &d / &d,
            Number::Dual(Dual::try_new(1.0, vec!["x".to_string()], vec![0.0]).unwrap())
        );
    }

    #[test]
    #[should_panic]
    fn test_enum_panic() {
        let d = Number::Dual2(Dual2::new(2.0, vec!["y".to_string()]));
        let d2 = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let _ = d / d2;
    }
}
