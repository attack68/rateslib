
use crate::calendars::{
    CalType,
    RollDay,
    Modifier,
};
use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use std::cmp::{Ordering, PartialEq};

/// A schedule frequency.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone)]
pub enum Frequency {
    /// Periods every month.
    Monthly,
    /// Periods every two months.
    BiMonthly,
    /// Periods every three months.
    Quarterly,
    /// Periods every four months.
    TriAnnually,
    /// Periods every six months.
    SemiAnnually,
    /// Periods every twelve months.
    Annually,
    /// Only every a single period.
    Zero,
}

impl Frequency {
    pub fn months(&self) -> u32 {
        match self {
            Frequency::Monthly => 1_u32,
            Frequency::BiMonthly => 2_u32,
            Frequency::Quarterly => 3_u32,
            Frequency::TriAnnually => 4_u32,
            Frequency::SemiAnnually => 6_u32,
            Frequency::Annually => 12_u32,
            Frequency::Zero => 120000_u32  // 10,000 years.
        }

    }
}

/// A stub type indicator for date inference.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone)]
pub enum Stub {
    /// Short front stub inference.
    ShortFront,
    /// Long front stub inference.
    LongFront,
    /// Short back stub inference.
    ShortBack,
    /// Long back stub inference.
    LongBack,
}

pub struct Schedule {
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    frequency: Frequency,
    front_stub: Option<NaiveDateTime>,
    back_stub: Option<NaiveDateTime>,
    roll: RollDay,
    modifier: Modifier,
    calendar: CalType,
    payment_lag: i8,

    // created data objects
    uschedule: Vec<NaiveDateTime>,
    aschedule: Vec<NaiveDateTime>,
    pschedule: Vec<NaiveDateTime>
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
//         OK()
//     }
// }


/// Test whether two dates' months define a period divisible by frequency months.
fn _is_divisible_months(start: &NaiveDateTime, end: &NaiveDateTime, frequency: &Frequency) -> bool {
    let months = end.month() - start.month();
    return (months % frequency.months()) == 0_u32
}

// /// Infer a roll day from given ueffective and utermination dates of a regular swap.
// fn _get_unadjusted_roll(ueffective: &NaiveDateTime, utermination: &NaiveDateTime, eom: bool) -> Option<RollDay> {
//     if ueffective.day() < 29 && utermination.day() < 28 {
//         if ueffective.day() == utermination.day() {
//             Some(RollDay::Int { day: ueffective.day() })
//         } else {
//             None
//         }
//     } else {
//         let non_eom_map: Vec<Vec<u32>> = vec![
//             vec![28, 28, 29, 30, 31, 30, 29, 28],
//             vec![28, 28, 0, 0, 0, 0, 0, 28],
//             vec![29, 0, 29, 30, 31, 30, 29, 0],
//             vec![30, 0, 30, 30, 31, 30, 0, 0],
//             vec![31, 0, 31, 31, 31, 0, 0, 0],
//             vec![30, 0, 30, 30, 0, 30, 0, 0],
//             vec![29, 0, 29, 0, 0, 0, 29, 0],
//             vec![28, 28, 0, 0, 0, 0, 0, 28],
//         ]
//         let eom_map: Vec<Vec<u32>> = [
//             [31, 28, 31, 31, 31, 30, 29, 28],
//             [28, 28, 0, 0, 0, 0, 0, 28],
//             [31, 0, 31, 31, 31, 30, 29, 0],
//             [31, 0, 31, 31, 31, 30, 0, 0],
//             [31, 0, 31, 31, 31, 0, 0, 0],
//             [30, 0, 30, 30, 0, 30, 0, 0],
//             [29, 0, 29, 0, 0, 0, 29, 0],
//             [28, 28, 0, 0, 0, 0, 0, 28],
//         ]
//     }
// }


/// Date categories to infer rolls on dates.
#[derive(Debug, PartialEq)]
enum RollDayCategory {
    /// Day is prior to any month end option.
    Pre28,
    /// Day can only be 28.
    Only28,
    /// Day can only be 29.
    Only29,
    /// Day can only be 30.
    Only30,
    /// Day can only be 31.
    Only31,
    /// Day can be 28, 29, 30, or 31.
    Post28,
    /// Day can be 29, 30 or 31.
    Post29,
    /// Day can be 30, or 31.
    Post30,
}

impl RollDayCategory {
    fn for_date(date: &NaiveDateTime) -> Self {
        match (date.day(), date.month()) {
            (d, _) if d < 28 => { Self::Pre28 }
            (d, 2) => {
                if is_leap_year(date.year()) {
                    if d == 28 { Self::Only28 } else { Self:: Post29 }
                } else { Self::Post28 }
            }
            (d, 1 | 3 | 5 | 7 | 8 | 10 | 12) => {
                if d == 31 { Self::Only31 }
                else if d == 30 { Self::Only30 }
                else if d == 29 { Self::Only29 }
                else { Self::Only28 }
            }
            (d, 4 | 6 | 9 | 11) => {
                if d == 30 { Self::Post30}
                else if d == 29 { Self::Only29 }
                else { Self::Only28 }
            }
            _ => panic!("Impossible match arm.")
        }
    }

    /// Combine two RollDayCategories (from a ueffective and utermination) to yield a RollDay.
    ///
    /// Panics on invalid combinations.
    fn combined(&self, other: &RollDayCategory, eom: bool) -> RollDay {
        match (self, other) {
            (RollDayCategory::Pre28, _) => panic!("Impossible roll day combinations"),
            (_, RollDayCategory::Pre28) => panic!("Impossible roll day combinations"),
            (RollDayCategory::Only28, RollDayCategory::Only28 | RollDayCategory::Post28) => RollDay::Int {day: 28},
            (RollDayCategory::Only28, RollDayCategory::Only29 | RollDayCategory::Only30 | RollDayCategory::Only31 | RollDayCategory::Post29 | RollDayCategory::Post30) => panic!("Impossible roll day combinations"),
            (RollDayCategory::Only29, RollDayCategory::Only29 | RollDayCategory::Post29 | RollDayCategory::Post28) => RollDay::Int {day: 29},
            (RollDayCategory::Only29, RollDayCategory::Only28 | RollDayCategory::Only30 | RollDayCategory::Only31 | RollDayCategory::Post30) => panic!("Impossible roll day combinations"),
            (RollDayCategory::Only30, RollDayCategory::Only30 | RollDayCategory::Post30 |RollDayCategory::Post29 | RollDayCategory::Post28) => RollDay::Int {day: 30},
            (RollDayCategory::Only30, RollDayCategory::Only28 | RollDayCategory::Only29 | RollDayCategory::Only31) => panic!("Impossible roll day combinations"),
            (RollDayCategory::Only31, RollDayCategory::Only31 | RollDayCategory::Post30 |RollDayCategory::Post29 | RollDayCategory::Post28) => RollDay::Int {day: 31},
            (RollDayCategory::Only31, RollDayCategory::Only28 | RollDayCategory::Only29 | RollDayCategory::Only30) => panic!("Impossible roll day combinations"),
            (RollDayCategory::Post28, RollDayCategory::Only28) => RollDay::Int {day: 28},
            (RollDayCategory::Post28, RollDayCategory::Only29) => RollDay::Int {day: 29},
            (RollDayCategory::Post28, RollDayCategory::Only30) => RollDay::Int {day: 30},
            (RollDayCategory::Post28, RollDayCategory::Only31) => RollDay::Int {day: 31},
            (RollDayCategory::Post28, RollDayCategory::Post28) => if eom { RollDay::EoM {} } else { RollDay::Int {day: 28} },
            (RollDayCategory::Post28, RollDayCategory::Post29) => if eom { RollDay::EoM {} } else { RollDay::Int {day: 29} },
            (RollDayCategory::Post28, RollDayCategory::Post30) => if eom { RollDay::EoM {} } else { RollDay::Int {day: 30} },
            (RollDayCategory::Post29, RollDayCategory::Only28) => panic!("Impossible roll day combinations"),
            (RollDayCategory::Post29, RollDayCategory::Only29) => RollDay::Int {day: 29},
            (RollDayCategory::Post29, RollDayCategory::Only30) => RollDay::Int {day: 30},
            (RollDayCategory::Post29, RollDayCategory::Only31) => RollDay::Int {day: 31},
            (RollDayCategory::Post29, RollDayCategory::Post28 | RollDayCategory::Post29) => if eom { RollDay::EoM {} } else { RollDay::Int {day: 29} },
            (RollDayCategory::Post29, RollDayCategory::Post30) => if eom { RollDay::EoM {} } else { RollDay::Int {day: 30} },
            (RollDayCategory::Post30, RollDayCategory::Only28 | RollDayCategory::Only29) => panic!("Impossible roll day combinations"),
            (RollDayCategory::Post30, RollDayCategory::Only30) => RollDay::Int {day: 30},
            (RollDayCategory::Post30, RollDayCategory::Only31) => RollDay::Int {day: 31},
            (RollDayCategory::Post30, RollDayCategory::Post28 | RollDayCategory::Post29 | RollDayCategory::Post30) => if eom { RollDay::EoM {} } else { RollDay::Int {day: 30} },
        }
    }
}

fn is_leap_year(year: i32) -> bool {
    NaiveDate::from_ymd_opt(year, 2, 29).is_some()
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
    fn test_is_divisible_months() {
        // test true
        let s1 = ndt(2022, 2, 2);
        let e1 = ndt(2022, 6, 6);
        assert_eq!(true, _is_divisible_months(&s1, &e1, &Frequency::BiMonthly));

        // test false
        let s2 = ndt(2022, 2, 2);
        let e2 = ndt(2022, 9, 6);
        assert_eq!(false, _is_divisible_months(&s2, &e2, &Frequency::Quarterly));
    }

    #[test]
    fn test_rolldaycategory() {
        assert_eq!(RollDayCategory::Pre28, RollDayCategory::for_date(&ndt(2000, 1, 1)));
        assert_eq!(RollDayCategory::Post28, RollDayCategory::for_date(&ndt(2022, 2, 28)));
        assert_eq!(RollDayCategory::Only28, RollDayCategory::for_date(&ndt(2024, 2, 28)));
        assert_eq!(RollDayCategory::Post29, RollDayCategory::for_date(&ndt(2024, 2, 29)));
        assert_eq!(RollDayCategory::Post30, RollDayCategory::for_date(&ndt(2024, 4, 30)));
        assert_eq!(RollDayCategory::Only31, RollDayCategory::for_date(&ndt(2000, 1, 31)));
        assert_eq!(RollDayCategory::Only30, RollDayCategory::for_date(&ndt(2000, 1, 30)));
        assert_eq!(RollDayCategory::Only29, RollDayCategory::for_date(&ndt(2000, 9, 29)));
        assert_eq!(RollDayCategory::Only28, RollDayCategory::for_date(&ndt(2000, 9, 28)));
    }

    #[test]
    fn test_get_unadjusted_roll_non_eom() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, RollDay) = vec![
            (ndt(2022, 2, 28), ndt(2022, 2, 28), RollDay::Int { day:28 })
        ];

//         ndt(2022, 2, 28),
//         ndt(2024, 2, 28),
//         ndt(2024, 2, 29),
//         ndt(2024, 4, 28),
//         ndt(2024, 4, 29),
//         ndt(2024, 4, 30),
//         ndt(2024, 7, 28),
//         nst(2024, 7, 29),
//         ndt(2024, 7, 30),
//         ndt(2024, 7, 31),
//         ]

        let inputs: Vec<Vec<NaiveDateTime>> = vec![
            vec![ndt(2022, 2, 22), ndt(2024, 2, 22)],
            vec![ndt(2022, 2, 22), ndt(2024, 2, 22)],
            vec![ndt(2022, 2, 28), ndt(2024, 2, 22)],
            vec![ndt(2022, 2, 22), ndt(2024, 2, 22)],
            vec![ndt(2022, 2, 22), ndt(2024, 2, 22)],
            vec![ndt(2022, 2, 22), ndt(2024, 2, 22)],
            vec![ndt(2022, 2, 22), ndt(2024, 2, 22)],
        ]
    }
}