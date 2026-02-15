// SPDX-License-Identifier: LicenseRef-Rateslib-Dual
//
// Copyright (c) 2026 Siffrorna Technology Limited
// This code cannot be used or copied externally
//
// Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
// Source-available, not open source.
//
// See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
// and/or contact info (at) rateslib (dot) com
////////////////////////////////////////////////////////////////////////////////////////////////////

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
