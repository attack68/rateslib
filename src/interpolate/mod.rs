use chrono::prelude::*;
use chrono::naive::{NaiveDate};
use chrono::Duration;

// use crate::dual::Duals;
// pub enum Linear {}
// pub enum Interpolation {
//     Linear,
// }

// pub fn interpolate_with_method(
//     x: &NaiveDate,
//     x_1: &NaiveDate,
//     y_1: Duals,
//     x_2: &NaiveDate,
//     y_2: Duals,
//     interpolation: &str,
//     start: Option<&NaiveDate>,
// ) -> Duals {
//     match interpolation {
//         "linear" => interpolate_linear(x, x_1, y_1, x_2, y_2),
//         _ => panic!["bad interpolation method specified"]
//     }
// }
//
// fn interpolate_linear(
//     x: &NaiveDate,
//     x_1: &NaiveDate,
//     y_1: Duals,
//     x_2: &NaiveDate,
//     y_2: Duals
// ) -> Duals {
//     let s = Duals::Float(
//         x.signed_duration_since(*x_1).num_days() as f64 /
//         x_2.signed_duration_since(*x_1).num_days() as f64
//     );
//     y_1.clone() + (y_2 - y_1) * s
// }