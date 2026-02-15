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

//! Perform linear algebra operations involving Arrays of [f64], [Dual](crate::dual::Dual) and [Dual2](crate::dual::Dual2).

mod linalg_dual;
mod linalg_f64;

pub use crate::dual::linalg::linalg_dual::{dmul11_, dmul21_, dmul22_, douter11_, dsolve};
pub use crate::dual::linalg::linalg_f64::{
    dfmul21_, dfmul22_, fdmul11_, fdmul21_, fdmul22_, fdsolve, fouter11_,
};
