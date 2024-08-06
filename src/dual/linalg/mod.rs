//! Perform linear algebra operations involving Arrays of `f64`, `Dual` and `Dual2`.

mod linalg_dual;
mod linalg_f64;

pub use crate::dual::linalg::linalg_dual::{dmul11_, dmul21_, dmul22_, douter11_, dsolve};
pub use crate::dual::linalg::linalg_f64::{
    dfmul21_, dfmul22_, fdmul11_, fdmul21_, fdmul22_, fdsolve, fouter11_,
};

pub(crate) use crate::dual::linalg::linalg_dual::argabsmax;
