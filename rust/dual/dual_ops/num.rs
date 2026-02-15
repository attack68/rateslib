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

use crate::dual::dual::{Dual, Dual2};
use crate::dual::enums::Number;
use num_traits::Num;

impl Num for Dual {
    // PartialEq + Zero + One + NumOps (Add + Sub + Mul + Div + Rem)
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual".to_string())
    }
}

impl Num for Dual2 {
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual2".to_string())
    }
}

impl Num for Number {
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Number".to_string())
    }
}
