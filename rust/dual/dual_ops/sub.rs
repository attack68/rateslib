use crate::dual::dual::{Dual, Dual2, Vars, VarsRelationship};
use crate::dual::enums::Number;
use auto_ops::impl_op_ex;
use std::sync::Arc;

// Sub
impl_op_ex!(-|a: &Dual, b: &f64| -> Dual {
    Dual {
        vars: Arc::clone(&a.vars),
        real: a.real - b,
        dual: a.dual.clone(),
    }
});
impl_op_ex!(-|a: &f64, b: &Dual| -> Dual {
    Dual {
        vars: Arc::clone(&b.vars),
        real: a - b.real,
        dual: -(b.dual.clone()),
    }
});
impl_op_ex!(-|a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&a.vars),
        real: a.real - b,
        dual: a.dual.clone(),
        dual2: a.dual2.clone(),
    }
});
impl_op_ex!(-|a: &f64, b: &Dual2| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&b.vars),
        real: a - b.real,
        dual: -(b.dual.clone()),
        dual2: -(b.dual2.clone()),
    }
});

// impl Sub for Dual
impl_op_ex!(-|a: &Dual, b: &Dual| -> Dual {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => Dual {
            real: a.real - b.real,
            dual: &a.dual - &b.dual,
            vars: Arc::clone(&a.vars),
        },
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual {
                real: x.real - y.real,
                dual: &x.dual - &y.dual,
                vars: Arc::clone(&x.vars),
            }
        }
    }
});

// impl Sub
impl_op_ex!(-|a: &Dual2, b: &Dual2| -> Dual2 {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => Dual2 {
            real: a.real - b.real,
            dual: &a.dual - &b.dual,
            dual2: &a.dual2 - &b.dual2,
            vars: Arc::clone(&a.vars),
        },
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual2 {
                real: x.real - y.real,
                dual: &x.dual - &y.dual,
                dual2: &x.dual2 - &y.dual2,
                vars: Arc::clone(&x.vars),
            }
        }
    }
});

// Sub for Number
impl_op_ex!(-|a: &Number, b: &Number| -> Number {
    match (a, b) {
        (Number::F64(f), Number::F64(f2)) => Number::F64(f - f2),
        (Number::F64(f), Number::Dual(d2)) => Number::Dual(f - d2),
        (Number::F64(f), Number::Dual2(d2)) => Number::Dual2(f - d2),
        (Number::Dual(d), Number::F64(f2)) => Number::Dual(d - f2),
        (Number::Dual(d), Number::Dual(d2)) => Number::Dual(d - d2),
        (Number::Dual(_), Number::Dual2(_)) => {
            panic!("Cannot mix dual types: Dual - Dual2")
        }
        (Number::Dual2(d), Number::F64(f2)) => Number::Dual2(d - f2),
        (Number::Dual2(_), Number::Dual(_)) => {
            panic!("Cannot mix dual types: Dual2 - Dual")
        }
        (Number::Dual2(d), Number::Dual2(d2)) => Number::Dual2(d - d2),
    }
});

// Sub for Number
impl_op_ex!(-|a: &Number, b: &f64| -> Number {
    match a {
        Number::F64(f) => Number::F64(f - b),
        Number::Dual(d) => Number::Dual(d - b),
        Number::Dual2(d) => Number::Dual2(d - b),
    }
});

// Sub for Number
impl_op_ex!(-|a: &f64, b: &Number| -> Number {
    match b {
        Number::F64(f) => Number::F64(a - f),
        Number::Dual(d) => Number::Dual(a - d),
        Number::Dual2(d) => Number::Dual2(a - d),
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_f64() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = (10.0 - d1) - 15.0;
        let expected = Dual::try_new(
            -6.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn sub() {
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
            -1.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, -3.0],
        )
        .unwrap();
        let result = d1 - d2;
        assert_eq!(result, expected)
    }
    #[test]
    fn sub_f64_2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = (10.0 - d1) - 15.0;
        let expected = Dual2::try_new(
            -6.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn sub2() {
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
            -1.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, -3.0],
            Vec::new(),
        )
        .unwrap();
        let result = d1 - d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_enum() {
        let f = Number::F64(2.0);
        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        assert_eq!(
            &f - &d,
            Number::Dual(Dual::try_new(-1.0, vec!["x".to_string()], vec![-1.0]).unwrap())
        );

        assert_eq!(
            &d - &d,
            Number::Dual(Dual::try_new(0.0, vec!["x".to_string()], vec![0.0]).unwrap())
        );
    }

    #[test]
    #[should_panic]
    fn test_enum_panic() {
        let d = Number::Dual2(Dual2::new(2.0, vec!["y".to_string()]));
        let d2 = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let _ = d - d2;
    }

    #[test]
    fn test_enum_f64() {
        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let res = 2.5_f64 - d;
        assert_eq!(
            res,
            Number::Dual(Dual::new(0.5, vec!["x".to_string()]) * -1.0)
        );

        let d = Number::Dual(Dual::new(3.0, vec!["x".to_string()]));
        let res = d - 2.5_f64;
        assert_eq!(res, Number::Dual(Dual::new(0.5, vec!["x".to_string()])));
    }
}
