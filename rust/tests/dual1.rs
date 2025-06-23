
use crate::dual::dual1::Dual;
use num_traits::Pow;
use std::sync::Arc;

#[test]
fn clone_arc() {
    let d1 = Dual::new(20.0, vec!["a".to_string()], vec![]);
    let d2 = d1.clone();
    assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
}
