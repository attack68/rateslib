pub mod json_py;

use serde::{Deserialize, Serialize};
use serde_json;

pub trait JSON: Serialize + for<'de> Deserialize<'de> {
    fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}
