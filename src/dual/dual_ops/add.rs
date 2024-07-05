use auto_ops::{impl_op, impl_op_ex, impl_op_ex_commutative};
use std::sync::Arc;
use crate::dual::dual::{Dual, Dual2, VarsState, Vars};

// Add f64
impl_op_ex_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
impl_op_ex_commutative!(+ |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone(), dual2: a.dual2.clone()}
});

// Add for Dual
impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsState::EquivByArc | VarsState::EquivByVal => {
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
        VarsState::EquivByArc | VarsState::EquivByVal => {
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