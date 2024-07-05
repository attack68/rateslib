use auto_ops::{impl_op, impl_op_ex, impl_op_ex_commutative};
use std::sync::Arc;
use crate::dual::dual::{Dual, Dual2, VarsState, Vars};
use crate::dual::linalg_f64::fouter11_;
use ndarray::Array2;

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
        VarsState::EquivByArc | VarsState::EquivByVal => Dual {
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
        VarsState::EquivByArc | VarsState::EquivByVal => {
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