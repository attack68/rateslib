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

//! Allows serialization and deserialization to JSON, with the ``serde`` crate.

pub(crate) mod json_py;
pub(crate) use crate::json::json_py::DeserializedObj;

use serde::{Deserialize, Serialize};
use serde_json;

/// Handles the `to` and `from` JSON conversion.
pub trait JSON: Serialize + for<'de> Deserialize<'de> {
    /// Return a JSON string representing the object.
    fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Create an object from a JSON string representation.
    fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}
