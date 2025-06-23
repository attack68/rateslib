//! Allows serialization and deserialization to JSON, with the ``serde`` crate.

pub mod json_py;

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
