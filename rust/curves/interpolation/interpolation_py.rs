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

use crate::curves::interpolation::utils::index_left;
use pyo3::pyfunction;

macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[pyfunction]
        #[pyo3(signature = (list_input, value, left_count=None))]
        pub fn $name(list_input: Vec<$type>, value: $type, left_count: Option<usize>) -> usize {
            index_left(&list_input[..], &value, left_count)
        }
    };
}

create_interface!(index_left_f64, f64);
