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

//! Toolset to create one dimensional spline curves.

mod spline;
pub(crate) mod spline_py;

pub use crate::splines::spline::{
    bspldnev_single_dual, bspldnev_single_dual2, bspldnev_single_f64, bsplev_single_dual,
    bsplev_single_dual2, bsplev_single_f64, PPSpline, PPSplineDual, PPSplineDual2, PPSplineF64,
};
