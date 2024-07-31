//! Toolset for automatic differentiation (AD).
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

pub use crate::dual::dual::{Dual, Dual2, ADOrder, DualsOrF64, VarsRelationship, FieldOps,
Gradient1, Gradient2, MathFuncs, Vars, set_order, set_order_clone
};
