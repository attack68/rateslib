// use crate::calendars::calendar_py::Cals;
// use crate::calendars::dcfs::;

use crate::json::JSON;
use serde::{Serialize, Deserialize};

// enum Nodes {
//     F64(IndexMap<datetime, f64>),
//     Dual(IndexMap<datetime, Dual>),
//     Dual2(IndexMap<datetime, Dual2>),
// }
// pub struct Curve {
//     nodes: Nodes,
//     calendar: Cals,
//     convention: Convention,
// }

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(from = "RectangleShadow")]
pub struct Rectangle {
    length: f64,
    #[serde(skip)]
    area: f64,
    #[serde(skip)]
    diagonal: f64,
}

impl Rectangle {
    fn new (length: f64) -> Self {
        Rectangle {length, area: length * length, diagonal: (1.41421 * length * length)}
    }
}

impl JSON for Rectangle {}

// The shadow type only for Deserialize
#[derive(Deserialize)]
pub struct RectangleShadow {
    length: f64,
}

impl std::convert::From<RectangleShadow> for Rectangle {
    fn from(shadow: RectangleShadow) -> Self {
        Rectangle::new(shadow.length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle() {
        let core = Rectangle::new(3.5);
        let json = core.to_json().unwrap();
        println!("{:?}", json);
        let derived = Rectangle::from_json(&json).unwrap();
        println!("{:?}", derived);
        assert!(false);
    }
}


