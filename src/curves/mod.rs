//! Create curves for calculating interest rates and discount factors.

pub(crate) mod nodes;
pub use crate::curves::nodes::Nodes;

pub(crate) mod interpolation;
pub use crate::curves::interpolation::intp_linear::LinearInterpolator;
pub use crate::curves::interpolation::intp_linear_zero_rate::LinearZeroRateInterpolator;
pub use crate::curves::interpolation::intp_log_linear::LogLinearInterpolator;

pub(crate) mod curve;
pub use crate::curves::curve::{Curve, CurveInterpolation};

pub(crate) mod curve_py;
