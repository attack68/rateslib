//! Toolset to create one dimensional spline curves.

mod spline;
pub(crate) mod spline_py;

pub use crate::splines::spline::{
    bspldnev_single_dual, bspldnev_single_dual2, bspldnev_single_f64, bsplev_single_dual,
    bsplev_single_dual2, bsplev_single_f64, PPSpline,
};
