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

//! Define a generic Western business weekday calendar without any specific holidays.

pub const WEEKMASK: &[u8] = &[5, 6]; // Saturday and Sunday weekend

// pub const RULES: &[&str] = &[];

pub const HOLIDAYS: &[&str] = &[]; // no specific holidays
