//! Toolset for forward mode automatic differentiation (AD).
//!
//! # AD Architecture
//!
//! The entire *rateslib* library is built around three core numeric types: [f64],
//! [Dual] and [Dual2]. Obviously [f64] allows for traditional computation, which benefits
//! from efficient calculation leveraging BLAS, while [Dual] and [Dual2] reduce performance
//! of calculation but provide efficient calculation of first order and second order
//! derivatives, respectively. Derivatives are calculated using forward mode AD,
//! similar, but not identical, to the
//! [Julia ForwardDiff library](https://github.com/JuliaDiff/ForwardDiff.jl).
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
    set_order, set_order_clone, ADOrder, Dual, Dual2, Gradient1, Gradient2, MathFuncs, Number,
    NumberArray1, NumberArray2, NumberMapping, NumberOps, NumberVec, Vars, VarsRelationship,
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
