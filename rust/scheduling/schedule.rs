use crate::calendars::{is_leap_year, CalType, Modifier, RollDay};
use crate::scheduling::enums::{Frequency, Stub};
use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use std::cmp::{Ordering, PartialEq};

pub struct Schedule {
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    frequency: Frequency,
    roll: RollDay,
    front_stub: Option<NaiveDateTime>,
    back_stub: Option<NaiveDateTime>,
    modifier: Modifier,
    calendar: CalType,
    payment_lag: i8,

    // created data objects
    uschedule: Vec<NaiveDateTime>,
    aschedule: Vec<NaiveDateTime>,
    pschedule: Vec<NaiveDateTime>,
}

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
//         calendar: CalType,
//         payment_lag: i8,
//     ) -> Result<Self, PyErr> {
//         Ok()
//     }
// }

pub(crate) fn regular_unadjusted_schedule(
    ueffective: &NaiveDateTime,
    utermination: &NaiveDateTime,
    frequency: &Frequency,
    roll: &RollDay,
) -> Vec<NaiveDateTime> {
    // validation tests
    if roll.validate_date(&ueffective).is_some() {
        panic!("`ueffective` is not valid.")
    }
    if roll.validate_date(&utermination).is_some() {
        panic!("`utermination` is not valid.")
    }
    if !frequency.is_divisible(&ueffective, &utermination) {
        panic!("`frequency` is not valid on dates.")
    }

    let mut ret: Vec<NaiveDateTime> = vec![ueffective.clone()];
    while ret.last().unwrap() < utermination {
        ret.push(frequency.next_period(ret.last().unwrap(), roll));
    }
    ret
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;

    //     fn fixture_hol_cal() -> Cal {
    //         let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
    //         Cal::new(hols, vec![5, 6])
    //     }

    #[test]
    fn test_regular_unadjusted_schedule() {
        let result = regular_unadjusted_schedule(
            &ndt(2000, 1, 1),
            &ndt(2001, 1, 1),
            &Frequency::Quarterly,
            &RollDay::SoM {},
        );
        assert_eq!(
            result,
            vec![
                ndt(2000, 1, 1),
                ndt(2000, 4, 1),
                ndt(2000, 7, 1),
                ndt(2000, 10, 1),
                ndt(2001, 1, 1)
            ]
        );
    }

    #[test]
    fn test_regular_unadjusted_schedule_imm() {
        // test the example given in Coding Interest Rates
        let result = regular_unadjusted_schedule(
            &ndt(2023, 3, 15),
            &ndt(2023, 9, 20),
            &Frequency::Monthly,
            &RollDay::IMM {},
        );
        assert_eq!(
            result,
            vec![
                ndt(2023, 3, 15),
                ndt(2023, 4, 19),
                ndt(2023, 5, 17),
                ndt(2023, 6, 21),
                ndt(2023, 7, 19),
                ndt(2023, 8, 16),
                ndt(2023, 9, 20)
            ]
        );
    }
}
