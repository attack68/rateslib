use crate::dual::dual::{Dual, Dual2, Vars, VarsRelationship, DualsOrF64};
use crate::dual::linalg_f64::fouter11_;
use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use ndarray::Array2;
use std::sync::Arc;

// Mul
impl_op_ex_commutative!(*|a: &Dual, b: &f64| -> Dual {
    Dual {
        vars: Arc::clone(&a.vars),
        real: a.real * b,
        dual: *b * &a.dual,
    }
});
impl_op_ex_commutative!(*|a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&a.vars),
        real: a.real * b,
        dual: *b * &a.dual,
        dual2: *b * &a.dual2,
    }
});

// impl Mul for Dual
impl_op_ex!(*|a: &Dual, b: &Dual| -> Dual {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => Dual {
            real: a.real * b.real,
            dual: &a.dual * b.real + &b.dual * a.real,
            vars: Arc::clone(&a.vars),
        },
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual {
                real: x.real * y.real,
                dual: &x.dual * y.real + &y.dual * x.real,
                vars: Arc::clone(&x.vars),
            }
        }
    }
});

// impl Mul for Dual2
impl_op_ex!(*|a: &Dual2, b: &Dual2| -> Dual2 {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
            let mut dual2: Array2<f64> = &a.dual2 * b.real + &b.dual2 * a.real;
            let cross_beta = fouter11_(&a.dual.view(), &b.dual.view());
            dual2 = dual2 + 0.5_f64 * (&cross_beta + &cross_beta.t());
            Dual2 {
                real: a.real * b.real,
                dual: &a.dual * b.real + &b.dual * a.real,
                vars: Arc::clone(&a.vars),
                dual2,
            }
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            let mut dual2: Array2<f64> = &x.dual2 * y.real + &y.dual2 * x.real;
            let cross_beta = fouter11_(&x.dual.view(), &y.dual.view());
            dual2 = dual2 + 0.5_f64 * (&cross_beta + &cross_beta.t());
            Dual2 {
                real: x.real * y.real,
                dual: &x.dual * y.real + &y.dual * x.real,
                vars: Arc::clone(&x.vars),
                dual2,
            }
        }
    }
});

// Mul for DualsOrF64
impl_op_ex!(* |a: &DualsOrF64, b: &DualsOrF64| -> DualsOrF64 {
    match (a,b) {
        (DualsOrF64::F64(f), DualsOrF64::F64(f2)) => DualsOrF64::F64(f * f2),
        (DualsOrF64::F64(f), DualsOrF64::Dual(d2)) => DualsOrF64::Dual(f * d2),
        (DualsOrF64::F64(f), DualsOrF64::Dual2(d2)) => DualsOrF64::Dual2(f * d2),
        (DualsOrF64::Dual(d), DualsOrF64::F64(f2)) => DualsOrF64::Dual(d * f2),
        (DualsOrF64::Dual(d), DualsOrF64::Dual(d2)) => DualsOrF64::Dual(d * d2),
        (DualsOrF64::Dual(_), DualsOrF64::Dual2(_)) => panic!("Cannot mix dual types: Dual * Dual2"),
        (DualsOrF64::Dual2(d), DualsOrF64::F64(f2)) => DualsOrF64::Dual2(d * f2),
        (DualsOrF64::Dual2(_), DualsOrF64::Dual(_)) => panic!("Cannot mix dual types: Dual2 * Dual"),
        (DualsOrF64::Dual2(d), DualsOrF64::Dual2(d2)) => DualsOrF64::Dual2(d * d2),
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_f64() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = 10.0 * d1 * 2.0;
        let expected = Dual::try_new(
            20.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![20.0, 40.0],
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn mul() {
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
            2.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![2.0, 4.0, 3.0],
        )
        .unwrap();
        let result = d1 * d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn mul_f64_2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = 10.0 * d1 * 2.0;
        let expected = Dual2::try_new(
            20.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![20.0, 40.0],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn mul2() {
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
            2.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![2.0, 4.0, 3.0],
            vec![0., 0., 1.5, 0., 0., 3., 1.5, 3., 0.],
        )
        .unwrap();
        let result = d1 * d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_enum() {
        let f = DualsOrF64::F64(2.0);
        let d = DualsOrF64::Dual(Dual::new(3.0, vec!["x".to_string()]));
        assert_eq!(&f*&d, DualsOrF64::Dual(Dual::try_new(6.0, vec!["x".to_string()], vec![2.0]).unwrap()));

        assert_eq!(&d*&d, DualsOrF64::Dual(Dual::try_new(9.0, vec!["x".to_string()], vec![6.0]).unwrap()));
    }

    #[test]
    #[should_panic]
    fn test_enum_panic() {
        let d = DualsOrF64::Dual2(Dual2::new(2.0, vec!["y".to_string()]));
        let d2 = DualsOrF64::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let _ = d * d2;
    }
}
