use crate::calendars::{CalType, Modifier, RollDay, DateRoll};
use crate::scheduling::enums::{Frequency};
use chrono::prelude::*;
// use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};

// use std::cmp::{Ordering, PartialEq};
// use std::fmt;

// #[derive(Debug, Clone)]
// struct ScheduleError {
//     message: String
// }

pub struct Schedule {
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    frequency: Frequency,
    front_stub: Option<NaiveDateTime>,
    back_stub: Option<NaiveDateTime>,
    modifier: Modifier,
    calendar: CalType,
    payment_lag: i32,

    // created data objects
    uschedule: Vec<NaiveDateTime>,
    aschedule: Vec<NaiveDateTime>,
    pschedule: Vec<NaiveDateTime>,
}

impl Schedule {
    /// Try to generate a Schedule with direct inputs and no inference using unadjusted dates.
    pub fn try_new_unadjusted(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        front_stub: Option<NaiveDateTime>,
        back_stub: Option<NaiveDateTime>,
        modifier: Modifier,
        calendar: CalType,
        payment_lag: i32,
    ) -> Result<Self, PyErr> {
        let (regular_start, regular_end) = match (front_stub, back_stub) {
            (None, None) => (ueffective, utermination),
            (Some(v), None) =>  (v, utermination),
            (None, Some(v)) =>  (ueffective, v),
            (Some(v), Some(w)) => (v, w)
        };

        // test if the determined regular period is actually a regular period under Frequency
        let regular_uschedule = frequency.try_regular_uschedule(&regular_start, &regular_end)?;
        let uschedule = composite_uschedule(ueffective, utermination, front_stub, back_stub, regular_uschedule);
        Ok(Self {
            ueffective,
            utermination,
            frequency,
            front_stub,
            back_stub,
            modifier,
            calendar: calendar.clone(),
            payment_lag,
            uschedule: uschedule.clone(),
            aschedule: uschedule.iter().map(|dt| calendar.roll(&dt, &modifier, false)).collect(),
            pschedule: uschedule.iter().map(|dt| calendar.lag(&dt, payment_lag, true)).collect(),
        })
    }
}

/// Get unadjusted schedule dates assuming all inputs are correct and pre-validated.
fn composite_uschedule(
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    front_stub: Option<NaiveDateTime>,
    back_stub: Option<NaiveDateTime>,
    regular_uschedule: Vec<NaiveDateTime>,
) -> Vec<NaiveDateTime> {
    let mut uschedule: Vec<NaiveDateTime> = vec![];
    match (front_stub, back_stub) {
        (None, None) => {
            uschedule.extend(regular_uschedule);
        },
        (Some(_v), None) => {
            uschedule.push(ueffective);
            uschedule.extend(regular_uschedule);
        },
        (None, Some(_v)) => {
            uschedule.extend(regular_uschedule);
            uschedule.push(utermination);
        },
        (Some(_v), Some(_w)) => {
            uschedule.push(ueffective);
            uschedule.extend(regular_uschedule);
            uschedule.push(utermination);
        },
    }
    uschedule
}


// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::{ndt, Cal};

    //     fn fixture_hol_cal() -> Cal {
    //         let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
    //         Cal::new(hols, vec![5, 6])
    //     }

    #[test]
    fn test_get_uschedule() {
        let result = Frequency::Months{number: 3, roll: RollDay::SoM{}}.try_regular_uschedule(
            &ndt(2000, 1, 1),
            &ndt(2001, 1, 1),
        ).unwrap();
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
    fn test_get_uschedule_imm() {
        // test the example given in Coding Interest Rates
        let result = Frequency::Months{number: 1, roll: RollDay::IMM{}}.try_regular_uschedule(
            &ndt(2023, 3, 15),
            &ndt(2023, 9, 20),
        ).unwrap();
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

    #[test]
    fn test_try_new_unadjusted() {
        let s = Schedule::try_new_unadjusted(
            ndt(2000, 1, 1),
            ndt(2000, 12, 15),
            Frequency::Months{number: 3, roll: RollDay::Unspecified{}},
            Some(ndt(2000, 3, 15)),
            None,
            Modifier::ModF,
            CalType::Cal(Cal::new(vec![], vec![])),
            1,
        ).unwrap();
        let uschedule = vec![ndt(2000, 1, 1), ndt(2000, 3, 15), ndt(2000, 6, 15), ndt(2000, 9, 15), ndt(2000, 12, 15)];
        let pschedule = vec![ndt(2000, 1, 2), ndt(2000, 3, 16), ndt(2000, 6, 16), ndt(2000, 9, 16), ndt(2000, 12, 16)];
        assert_eq!(uschedule, s.uschedule);
        assert_eq!(pschedule, s.pschedule);
    }
}
