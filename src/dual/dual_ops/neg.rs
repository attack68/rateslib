use auto_ops::{impl_op, impl_op_ex, impl_op_ex_commutative};
use std::sync::Arc;
use crate::dual::dual::{Dual, Dual2};

impl_op!(-|a: Dual| -> Dual {
    Dual { vars: a.vars, real: -a.real, dual: -a.dual }
});
impl_op!(-|a: &Dual| -> Dual {
    Dual { vars: Arc::clone(&a.vars), real: -a.real, dual: &a.dual * -1.0 }
});

impl_op!(-|a: Dual2| -> Dual2 {
    Dual2 { vars: a.vars, real: -a.real, dual: -a.dual, dual2: -a.dual2 }
});

impl_op!(-|a: &Dual2| -> Dual2 {
    Dual2 { vars: Arc::clone(&a.vars), real: -a.real, dual: &a.dual * -1.0, dual2: &a.dual2 * -1.0 }
});