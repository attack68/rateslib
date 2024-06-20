use serde::{Serialize, Deserialize};
use serde_json;

pub trait JSON: Serialize + for<'a> Deserialize<'a> {
    fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

