//! Toolset for forward mode automatic differentiation (AD).
//!
//! # AD Architecture
//!
//! This library
//!
//! Create and use data types for calculating derivatives up to second order using automatic
//! differentiation (AD). The type of AD used in *rateslib* is forward mode, dual number based.
//!
//! A first order dual number represents a function value and a linear manifold of the
//! gradient at that point. A second order dual number represents a function value and
//! a quadratic manifold of the gradient at that point.
//!
//! Mathematical operations are defined to give dual numbers the ability to combine, and
//! flexibly reference different variables at any point during calculations.
//!

mod dual;
mod dual_ops;
pub(crate) mod dual_py;
pub mod linalg;
pub(crate) mod linalg_py;

pub use crate::dual::dual::{
    set_order, set_order_clone, ADOrder, Dual, Dual2, FieldOps, Gradient1, Gradient2, MathFuncs,
    Number, Vars, VarsRelationship,
};

/// Utility for creating an ordered list of variable tags from a string and enumerator
pub(crate) fn get_variable_tags(name: &str, range: usize) -> Vec<String> {
    Vec::from_iter((0..range).map(|i| name.to_string() + &i.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_variable_tags() {
        let result = get_variable_tags("x", 3);
        assert_eq!(
            result,
            vec!["x0".to_string(), "x1".to_string(), "x2".to_string()]
        )
    }
}
