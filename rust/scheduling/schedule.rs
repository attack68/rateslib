// use crate::scheduling::{is_leap_year, Calendar, Modifier, RollDay};
// use chrono::prelude::*;
// use pyo3::exceptions::PyValueError;
// use pyo3::{pyclass, PyErr};
// use std::cmp::{Ordering, PartialEq};
//
// use crate::scheduling::enums::Frequency;

// pub struct Schedule {
//     ueffective: NaiveDateTime,
//     utermination: NaiveDateTime,
//     frequency: Frequency,
//     front_stub: Option<NaiveDateTime>,
//     back_stub: Option<NaiveDateTime>,
//     roll: RollDay,
//     modifier: Modifier,
//     calendar: Calendar,
//     payment_lag: i8,
//
//     // created data objects
//     uschedule: Vec<NaiveDateTime>,
//     aschedule: Vec<NaiveDateTime>,
//     pschedule: Vec<NaiveDateTime>,
// }

// impl Schedule {
//     pub fn try_new(
//         effective: NaiveDateTime,
//         termination: NaiveDateTime,
//         frequency: Frequency,
//         stub: Stub,
//         front_stub: Option<NaiveDateTime>,
//         back_stub: Option<NaiveDateTime>,
//         roll: Option<RollDay>,
//         eom: bool,
//         modifier: Modifier,
//         calendar: Calendar,
//         payment_lag: i8,
//     ) -> Result<Self, PyErr> {
//         OK()
//     }
// }

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::ndt;

    //     fn fixture_hol_cal() -> Cal {
    //         let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
    //         Cal::new(hols, vec![5, 6])
    //     }
}
