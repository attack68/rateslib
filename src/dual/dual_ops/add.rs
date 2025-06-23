use crate::dual::dual::{Dual, Dual2, Vars, VarsRelationship};
use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use std::sync::Arc;

// Add f64
impl_op_ex_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
impl_op_ex_commutative!(+ |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone(), dual2: a.dual2.clone()}
});

// Add for Dual
impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
            Dual {real: a.real + b.real, dual: &a.dual + &b.dual, vars: Arc::clone(&a.vars)}
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual {real: x.real + y.real, dual: &x.dual + &y.dual, vars: Arc::clone(&x.vars)}
        }
    }
});

// Add for Dual2
impl_op_ex!(+ |a: &Dual2, b: &Dual2| -> Dual2 {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
            Dual2 {
                real: a.real + b.real,
                dual: &a.dual + &b.dual,
                dual2: &a.dual2 + &b.dual2,
                vars: Arc::clone(&a.vars)}
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual2 {
                real: x.real + y.real,
                dual: &x.dual + &y.dual,
                dual2: &x.dual2 + &y.dual2,
                vars: Arc::clone(&x.vars)}
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_f64() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = 10.0 + d1 + 15.0;
        let expected = Dual::try_new(
            26.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn add() {
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
            3.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();
        let result = d1 + d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn add_f64_2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = 10.0 + d1 + 15.0;
        let expected = Dual2::try_new(
            26.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn add2() {
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
            3.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, 3.0],
            Vec::new(),
        )
        .unwrap();
        let result = d1 + d2;
        assert_eq!(result, expected)
    }
}
