
use crate::dual::dual1::Dual;
use std::sync::Arc;

#[test]
fn clone_arc() {
    let d1 = Dual::new(20.0, vec!["a".to_string()], vec![]);
    let d2 = d1.clone();
    assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
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
