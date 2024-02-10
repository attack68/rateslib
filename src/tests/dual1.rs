
use crate::dual::dual1::Dual;
use num_traits::Pow;
use std::sync::Arc;

#[test]
fn clone_arc() {
    let d1 = Dual::new(20.0, vec!["a".to_string()], vec![]);
    let d2 = d1.clone();
    assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
}

#[test]
fn to_new_ordered_vars() {
    let d1 = Dual::new(20.0, vec!["a".to_string()], vec![]);
    let d2 = Dual::new(20.0, vec!["a".to_string(), "b".to_string()], vec![]);
    let d3 = d1.to_new_ordered_vars(&d2.vars);
    assert!(Arc::ptr_eq(&d3.vars, &d2.vars));
    assert!(d3.dual.len() == 2);
    let d4 = d2.to_new_ordered_vars(&d1.vars);
    assert!(Arc::ptr_eq(&d4.vars, &d1.vars));
    assert!(d4.dual.len() == 1);
}

#[test]
fn new_dual() {
    Dual::new(2.3, Vec::from([String::from("a")]), Vec::new());
}

#[test]
#[should_panic]
fn new_dual_panic() {
    Dual::new(2.3, Vec::from([String::from("a"), String::from("b")]), Vec::from([1.0]));
}

#[test]
fn zero_init() {
    let d = Dual::new(2.3, Vec::from([String::from("a"), String::from("b")]), Vec::new());
    for (_, val) in d.dual.indexed_iter() {
        assert!(*val == 1.0)
    }
}

#[test]
fn negate() {
    let d = Dual::new(2.3, Vec::from([String::from("a"), String::from("b")]), Vec::from([2., -1.4]));
    let d2 = -d.clone();
    assert!(d2.real == -2.3);
    assert!(Arc::ptr_eq(&d.vars, &d2.vars));
    assert!(d2.dual[0] == -2.0);
    assert!(d2.dual[1] == 1.4);
}

#[test]
fn eq_ne() {
    // Dual with vars - f64
    assert!(Dual::new(0.0, Vec::from([String::from("a")]), Vec::new()) != 0.0);
    // Dual with no vars - f64 (+reverse)
    assert!(Dual::new(2.0, Vec::new(), Vec::new()) == 2.0);
    assert!(2.0 == Dual::new(2.0, Vec::new(), Vec::new()));
    // Dual - Dual (various real, vars, gradient mismatch)
    let d = Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([2.3]));
    assert!(d == Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([2.3])));
    assert!(d != Dual::new(2.0, Vec::from([String::from("b")]), Vec::from([2.3])));
    assert!(d != Dual::new(3.0, Vec::from([String::from("a")]), Vec::from([2.3])));
    assert!(d != Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([1.3])));
    // Dual - Dual (missing Vars are zero and upcasted)
    assert!(d == Dual::new(2.0, Vec::from([String::from("a"), String::from("b")]), Vec::from([2.3, 0.0])));
}

#[test]
fn add_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = 10.0 + d1 + 15.0;
    let expected = Dual::new(26.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    assert_eq!(result, expected)
}

#[test]
fn add() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(3.0, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![1.0, 2.0, 3.0]);
    let result = d1 + d2;
    assert_eq!(result, expected)
}

#[test]
fn sub_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = (10.0 - d1) -15.0;
    let expected = Dual::new(-6.0, vec!["v0".to_string(), "v1".to_string()], vec![-1.0, -2.0]);
    assert_eq!(result, expected)
}

#[test]
fn sub() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(-1.0, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![1.0, 2.0, -3.0]);
    let result = d1 - d2;
    assert_eq!(result, expected)
}

#[test]
fn mul_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = 10.0 * d1 * 2.0;
    let expected = Dual::new(20.0, vec!["v0".to_string(), "v1".to_string()], vec![20.0, 40.0]);
    assert_eq!(result, expected)
}

#[test]
fn mul() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(2.0, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![2.0, 4.0, 3.0]);
    let result = d1 * d2;
    assert_eq!(result, expected)
}

#[test]
fn inv() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1.clone() * d1.pow(-1.0);
    let expected = Dual::new(1.0, vec![], vec![]);
    assert_eq!(result, expected)
}

#[test]
fn abs() {
    let d1 = Dual::new(-2.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1.abs();
    let expected = 2.0;
    assert_eq!(result, expected)
}


#[test]
fn div_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1 / 2.0;
    let expected = Dual::new(0.5, vec!["v0".to_string(), "v1".to_string()], vec![0.5, 1.0]);
    assert_eq!(result, expected)
}

#[test]
fn f64_div() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = 2.0 / d1.clone();
    let expected = Dual::new(2.0, vec![], vec![]) / d1;
    assert_eq!(result, expected)
}

#[test]
fn div() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(0.5, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![0.5, 1.0, -0.75]);
    let result = d1 / d2;
    assert_eq!(result, expected)
}

