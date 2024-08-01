pub(crate) mod nodes;

pub(crate) mod interpolation;
pub use crate::curves::interpolation::traits::{CurveInterpolator, CurveInterpolation};
pub use crate::curves::interpolation::intp_log_linear::LogLinear;

pub(crate) mod curve;
pub use crate::curves::curve::{Curve};
