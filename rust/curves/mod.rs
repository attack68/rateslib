//! Create curves for calculating interest rates and discount factors.

pub(crate) mod nodes;
pub use crate::curves::nodes::Nodes;

pub(crate) mod interpolation;
pub use crate::curves::interpolation::intp_flat_backward::FlatBackwardInterpolator;
pub use crate::curves::interpolation::intp_flat_forward::FlatForwardInterpolator;
pub use crate::curves::interpolation::intp_linear::LinearInterpolator;
pub use crate::curves::interpolation::intp_linear_zero_rate::LinearZeroRateInterpolator;
pub use crate::curves::interpolation::intp_log_linear::LogLinearInterpolator;
pub use crate::curves::interpolation::intp_null::NullInterpolator;

pub(crate) mod curve;
pub use crate::curves::curve::{CurveDF, CurveInterpolation, Modifier};

pub(crate) mod curve_py;
pub(crate) use crate::curves::curve_py::_get_modifier_str;

mod serde;
