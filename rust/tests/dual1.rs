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

use crate::dual::Dual;
use std::sync::Arc;

#[test]
fn clone_arc() {
    let d1 = Dual::new(20.0, vec!["a".to_string()]);
    let d2 = d1.clone();
    assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
}
