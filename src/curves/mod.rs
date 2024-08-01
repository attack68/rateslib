//! Create curves for calculating interest rates and discount factors.

pub(crate) mod nodes;
pub use crate::curves::nodes::Nodes;

pub(crate) mod interpolation;
pub use crate::curves::interpolation::traits::{CurveInterpolator, CurveInterpolation};
pub use crate::curves::interpolation::intp_log_linear::LogLinearInterpolator;

pub(crate) mod curve;
pub use crate::curves::curve::{Curve};
