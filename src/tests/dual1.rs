
use crate::dual::dual1::Dual;
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
    let d4 = d2.to_new_ordered_vars(&d1.vars);
    assert!(Arc::ptr_eq(&d4.vars, &d1.vars));
}

#[test]
fn add_dual_float() {
    let result = 10.0 + Dual::new(20.0, vec!["a".to_string()], vec![]) + Dual::new(5.0, vec!["b".to_string()], vec![2.0]) + 10.0;
    assert_eq!(result, Dual::new(45.0, vec!["a".to_string(), "b".to_string()], vec![1.0, 2.0]))
}

#[test]
fn add_dual_same() {
    let result = Dual::new(20.0, vec!["a".to_string(), "b".to_string()], vec![]) + Dual::new(5.0, vec!["a".to_string(), "b".to_string()], vec![2.0, 2.5]);
    assert_eq!(result, Dual::new(25.0, vec!["a".to_string(), "b".to_string()], vec![3.0, 3.5]))
}

#[test]
fn new_dual() {
    let result = Dual::new(2.3, Vec::from([String::from("a")]), Vec::new());
}

#[test]
#[should_panic]
fn new_dual_panic() {
    let result = Dual::new(
        2.3, Vec::from([String::from("a"), String::from("b")]), Vec::from([1.0])
    );
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
    assert!(Dual::new(0.0, Vec::from([String::from("a")]), Vec::new()) != 0.0);
    assert!(Dual::new(2.0, Vec::new(), Vec::new()) == 2.0);
    assert!(2.0 == Dual::new(2.0, Vec::new(), Vec::new()));
    let d = Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([2.3]));
    assert!(d == Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([2.3])));
    assert!(d != Dual::new(2.0, Vec::from([String::from("b")]), Vec::from([2.3])));
    assert!(d != Dual::new(3.0, Vec::from([String::from("a")]), Vec::from([2.3])));
    assert!(d != Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([1.3])));
}

