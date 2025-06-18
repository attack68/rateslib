use crate::calendars::{is_leap_year, Cal, DateRoll, Modifier, RollDay, ndt};
use chrono::prelude::*;
use pyo3::{pyclass, PyErr};
use pyo3::exceptions::PyValueError;

// /// An indicator for valid or invalid schedules.
// pub(crate) enum ValidateSchedule {
//     Invalid {
//         error: String,
//     },
//     Valid {
//         ueffective: NaiveDateTime,
//         utermination: NaiveDateTime,
//         front_stub: Option<NaiveDateTime>,
//         back_stub: Option<NaiveDateTime>,
//         frequency: Frequency,
//         roll: RollDay,
//         eom: bool,
//     },
// }

/// A stub type indicator for date inference on one side of the schedule.
#[pyclass(module = "rateslib.rs", eq, eq_int)]
#[derive(Copy, Clone, PartialEq)]
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

/// A schedule frequency.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Frequency {
    /// A set number of calendar weeks.
    Weeks { number: u32 },
    /// A set number of calendar months.
    Months { number: u32, roll: RollDay },
    /// Only ever a single period
    Zero {},
}

impl Frequency {
    /// Calculate the next unadjusted period date in a schedule given a valid `ueffective` date.
    ///
    /// Note `ueffective` should be valid relative to the roll. If unsure, call
    /// ``roll.validate_date(&ueffective)``.
    pub fn next_period(&self, ueffective: &NaiveDateTime) -> NaiveDateTime {
        let cal = Cal::new(vec![], vec![]);
        match self {
            Frequency::Weeks{ number: n } => cal.add_days(ueffective, *n as i32 * 7, &Modifier::Act, true),
            Frequency::Months{ number: n, roll: r } => cal.add_months(ueffective, *n as i32, &Modifier::Act, r, true),
            Frequency::Zero{} => ndt(9999, 1, 1),
        }
    }

    /// Return a `uschedule` if the the dates satisfy constraints.
    pub fn try_regular_uschedule(&self, regular_ustart: &NaiveDateTime, regular_uend: &NaiveDateTime) -> Result<Vec<NaiveDateTime>, PyErr> {
        let mut v: Vec<NaiveDateTime> = vec![];
        let mut date = *regular_ustart;
        while date < *regular_uend {
            v.push(date);
            date = self.next_period(&date);
        }
        if date == *regular_uend {
            v.push(*regular_uend);
            Ok(v)
        } else {
            Err(PyValueError::new_err("Input dates to Frequency do not define a regular unadjusted schedule"))
        }
    }

}

/// Date categories to infer rolls on dates.
#[derive(Debug, PartialEq)]
pub(crate) enum RollDayCategory {
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
    pub(crate) fn from_date(date: &NaiveDateTime) -> Self {
        match (date.day(), date.month()) {
            (d, _) if d < 28 => Self::Pre28,
            (d, 2) => {
                if is_leap_year(date.year()) {
                    if d == 28 {
                        Self::Only28
                    } else {
                        Self::Post29
                    }
                } else {
                    Self::Post28
                }
            }
            (d, 1 | 3 | 5 | 7 | 8 | 10 | 12) => {
                if d == 31 {
                    Self::Only31
                } else if d == 30 {
                    Self::Only30
                } else if d == 29 {
                    Self::Only29
                } else {
                    Self::Only28
                }
            }
            (d, 4 | 6 | 9 | 11) => {
                if d == 30 {
                    Self::Post30
                } else if d == 29 {
                    Self::Only29
                } else {
                    Self::Only28
                }
            }
            _ => panic!("Impossible match arm."),
        }
    }

    /// Combine two RollDayCategories (from a ueffective and utermination) to yield a RollDay.
    ///
    /// Returns `None` if the combination is impossible or the dates do not correspond to
    /// end of month possibilities
    pub(crate) fn get_rollday_from_eom_categories(
        &self,
        other: &RollDayCategory,
        eom: bool,
    ) -> Option<RollDay> {
        match (self, other) {
            (RollDayCategory::Pre28, _) => None,
            (_, RollDayCategory::Pre28) => None,
            (RollDayCategory::Only28, RollDayCategory::Only28 | RollDayCategory::Post28) => {
                Some(RollDay::Int { day: 28 })
            }
            (
                RollDayCategory::Only28,
                RollDayCategory::Only29
                | RollDayCategory::Only30
                | RollDayCategory::Only31
                | RollDayCategory::Post29
                | RollDayCategory::Post30,
            ) => None,
            (
                RollDayCategory::Only29,
                RollDayCategory::Only29 | RollDayCategory::Post29 | RollDayCategory::Post28,
            ) => Some(RollDay::Int { day: 29 }),
            (
                RollDayCategory::Only29,
                RollDayCategory::Only28
                | RollDayCategory::Only30
                | RollDayCategory::Only31
                | RollDayCategory::Post30,
            ) => None,
            (
                RollDayCategory::Only30,
                RollDayCategory::Only30
                | RollDayCategory::Post30
                | RollDayCategory::Post29
                | RollDayCategory::Post28,
            ) => Some(RollDay::Int { day: 30 }),
            (
                RollDayCategory::Only30,
                RollDayCategory::Only28 | RollDayCategory::Only29 | RollDayCategory::Only31,
            ) => None,
            (
                RollDayCategory::Only31,
                RollDayCategory::Only31
                | RollDayCategory::Post30
                | RollDayCategory::Post29
                | RollDayCategory::Post28,
            ) => Some(RollDay::Int { day: 31 }),
            (
                RollDayCategory::Only31,
                RollDayCategory::Only28 | RollDayCategory::Only29 | RollDayCategory::Only30,
            ) => None,
            (RollDayCategory::Post28, RollDayCategory::Only28) => Some(RollDay::Int { day: 28 }),
            (RollDayCategory::Post28, RollDayCategory::Only29) => Some(RollDay::Int { day: 29 }),
            (RollDayCategory::Post28, RollDayCategory::Only30) => Some(RollDay::Int { day: 30 }),
            (RollDayCategory::Post28, RollDayCategory::Only31) => Some(RollDay::Int { day: 31 }),
            (RollDayCategory::Post28, RollDayCategory::Post28) => Some(if eom {
                RollDay::EoM {}
            } else {
                RollDay::Int { day: 28 }
            }),
            (RollDayCategory::Post28, RollDayCategory::Post29) => Some(if eom {
                RollDay::EoM {}
            } else {
                RollDay::Int { day: 29 }
            }),
            (RollDayCategory::Post28, RollDayCategory::Post30) => Some(if eom {
                RollDay::EoM {}
            } else {
                RollDay::Int { day: 30 }
            }),
            (RollDayCategory::Post29, RollDayCategory::Only28) => None,
            (RollDayCategory::Post29, RollDayCategory::Only29) => Some(RollDay::Int { day: 29 }),
            (RollDayCategory::Post29, RollDayCategory::Only30) => Some(RollDay::Int { day: 30 }),
            (RollDayCategory::Post29, RollDayCategory::Only31) => Some(RollDay::Int { day: 31 }),
            (RollDayCategory::Post29, RollDayCategory::Post28 | RollDayCategory::Post29) => {
                Some(if eom {
                    RollDay::EoM {}
                } else {
                    RollDay::Int { day: 29 }
                })
            }
            (RollDayCategory::Post29, RollDayCategory::Post30) => Some(if eom {
                RollDay::EoM {}
            } else {
                RollDay::Int { day: 30 }
            }),
            (RollDayCategory::Post30, RollDayCategory::Only28 | RollDayCategory::Only29) => None,
            (RollDayCategory::Post30, RollDayCategory::Only30) => Some(RollDay::Int { day: 30 }),
            (RollDayCategory::Post30, RollDayCategory::Only31) => Some(RollDay::Int { day: 31 }),
            (
                RollDayCategory::Post30,
                RollDayCategory::Post28 | RollDayCategory::Post29 | RollDayCategory::Post30,
            ) => Some(if eom {
                RollDay::EoM {}
            } else {
                RollDay::Int { day: 30 }
            }),
        }
    }
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
    fn test_try_regular_months() {
        // test true
        let s1 = ndt(2022, 2, 2);
        let e1 = ndt(2022, 6, 2);
        assert_eq!(true, Frequency::Months{number:2, roll: RollDay::Unspecified{}}.try_regular_uschedule(&s1, &e1).is_ok());

        // test false
        let s2 = ndt(2022, 2, 2);
        let e2 = ndt(2022, 9, 6);
        assert_eq!(true, Frequency::Months{number:3, roll: RollDay::Unspecified{}}.try_regular_uschedule(&s2, &e2).is_err());

        // test true
        let s2 = ndt(2023, 2, 1);
        let e2 = ndt(2023, 3, 15);
        assert_eq!(true, Frequency::Weeks{number:3}.try_regular_uschedule(&s2, &e2).is_ok());

        // test true
        let s2 = ndt(2023, 2, 1);
        let e2 = ndt(2023, 3, 16);
        assert_eq!(true, Frequency::Weeks{number:3}.try_regular_uschedule(&s2, &e2).is_err());
    }

    #[test]
    fn test_rolldaycategory() {
        assert_eq!(
            RollDayCategory::Pre28,
            RollDayCategory::from_date(&ndt(2000, 1, 1))
        );
        assert_eq!(
            RollDayCategory::Post28,
            RollDayCategory::from_date(&ndt(2022, 2, 28))
        );
        assert_eq!(
            RollDayCategory::Only28,
            RollDayCategory::from_date(&ndt(2024, 2, 28))
        );
        assert_eq!(
            RollDayCategory::Post29,
            RollDayCategory::from_date(&ndt(2024, 2, 29))
        );
        assert_eq!(
            RollDayCategory::Post30,
            RollDayCategory::from_date(&ndt(2024, 4, 30))
        );
        assert_eq!(
            RollDayCategory::Only31,
            RollDayCategory::from_date(&ndt(2000, 1, 31))
        );
        assert_eq!(
            RollDayCategory::Only30,
            RollDayCategory::from_date(&ndt(2000, 1, 30))
        );
        assert_eq!(
            RollDayCategory::Only29,
            RollDayCategory::from_date(&ndt(2000, 9, 29))
        );
        assert_eq!(
            RollDayCategory::Only28,
            RollDayCategory::from_date(&ndt(2000, 9, 28))
        );
    }

    #[test]
    fn test_get_unadjusted_roll_non_eom() {
        // test the returned RollDay inference combining RollDayCategories
        let options: Vec<(NaiveDateTime, NaiveDateTime, RollDay)> = vec![
            (ndt(2022, 2, 28), ndt(2022, 2, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 2, 28), ndt(2024, 2, 29), RollDay::Int { day: 29 }),
            (ndt(2024, 2, 29), ndt(2022, 2, 28), RollDay::Int { day: 29 }),
            (ndt(2022, 2, 28), ndt(2022, 4, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 2, 28), ndt(2022, 4, 29), RollDay::Int { day: 29 }),
            (ndt(2022, 2, 28), ndt(2022, 4, 30), RollDay::Int { day: 30 }),
            (ndt(2022, 4, 28), ndt(2022, 2, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 4, 29), ndt(2022, 2, 28), RollDay::Int { day: 29 }),
            (ndt(2022, 4, 30), ndt(2022, 2, 28), RollDay::Int { day: 30 }),
            (ndt(2022, 2, 28), ndt(2022, 7, 31), RollDay::Int { day: 31 }),
            (ndt(2022, 7, 31), ndt(2022, 2, 28), RollDay::Int { day: 31 }),
            (ndt(2022, 4, 28), ndt(2022, 7, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 4, 29), ndt(2022, 7, 29), RollDay::Int { day: 29 }),
            (ndt(2022, 4, 30), ndt(2022, 7, 30), RollDay::Int { day: 30 }),
            (ndt(2022, 4, 30), ndt(2022, 7, 31), RollDay::Int { day: 31 }),
            (ndt(2022, 7, 28), ndt(2022, 4, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 7, 29), ndt(2022, 4, 29), RollDay::Int { day: 29 }),
            (ndt(2022, 7, 30), ndt(2022, 4, 30), RollDay::Int { day: 30 }),
            (ndt(2022, 7, 31), ndt(2022, 4, 30), RollDay::Int { day: 31 }),
        ];

        for option in options.iter() {
            assert_eq!(
                option.2,
                RollDayCategory::from_date(&option.0)
                    .get_rollday_from_eom_categories(&RollDayCategory::from_date(&option.1), false)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_get_unadjusted_roll_eom() {
        // test the returned RollDay inference combining RollDayCategories, when EoM in place.
        let options: Vec<(NaiveDateTime, NaiveDateTime, RollDay)> = vec![
            (ndt(2022, 2, 28), ndt(2022, 2, 28), RollDay::EoM {}),
            (ndt(2022, 2, 28), ndt(2024, 2, 29), RollDay::EoM {}),
            (ndt(2024, 2, 29), ndt(2022, 2, 28), RollDay::EoM {}),
            (ndt(2022, 2, 28), ndt(2022, 4, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 2, 28), ndt(2022, 4, 29), RollDay::Int { day: 29 }),
            (ndt(2022, 2, 28), ndt(2022, 4, 30), RollDay::EoM {}),
            (ndt(2022, 4, 28), ndt(2022, 2, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 4, 29), ndt(2022, 2, 28), RollDay::Int { day: 29 }),
            (ndt(2022, 4, 30), ndt(2022, 2, 28), RollDay::EoM {}),
            (ndt(2022, 2, 28), ndt(2022, 7, 31), RollDay::Int { day: 31 }),
            (ndt(2022, 7, 31), ndt(2022, 2, 28), RollDay::Int { day: 31 }),
            (ndt(2022, 4, 28), ndt(2022, 7, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 4, 29), ndt(2022, 7, 29), RollDay::Int { day: 29 }),
            (ndt(2022, 4, 30), ndt(2022, 7, 30), RollDay::Int { day: 30 }),
            (ndt(2022, 4, 30), ndt(2022, 7, 31), RollDay::Int { day: 31 }),
            (ndt(2022, 7, 28), ndt(2022, 4, 28), RollDay::Int { day: 28 }),
            (ndt(2022, 7, 29), ndt(2022, 4, 29), RollDay::Int { day: 29 }),
            (ndt(2022, 7, 30), ndt(2022, 4, 30), RollDay::Int { day: 30 }),
            (ndt(2022, 7, 31), ndt(2022, 4, 30), RollDay::Int { day: 31 }),
        ];

        for option in options.iter() {
            assert_eq!(
                option.2,
                RollDayCategory::from_date(&option.0)
                    .get_rollday_from_eom_categories(&RollDayCategory::from_date(&option.1), true)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_get_next_period() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime)> = vec![
            (
                Frequency::Months{number:1, roll: RollDay::Unspecified{}},
                ndt(2022, 7, 30),
                ndt(2022, 8, 30),
            ),
            (
                Frequency::Months{number:2, roll: RollDay::Unspecified{}},
                ndt(2022, 7, 30),
                ndt(2022, 9, 30),
            ),
            (
                Frequency::Months{number:3, roll: RollDay::Unspecified{}},
                ndt(2022, 7, 30),
                ndt(2022, 10, 30),
            ),
            (
                Frequency::Months{number:4, roll: RollDay::Unspecified{}},
                ndt(2022, 7, 30),
                ndt(2022, 11, 30),
            ),
            (
                Frequency::Months{number:6, roll: RollDay::Unspecified{}},
                ndt(2022, 7, 30),
                ndt(2023, 1, 30),
            ),
            (
                Frequency::Months{number:12, roll: RollDay::Unspecified{}},
                ndt(2022, 7, 30),
                ndt(2023, 7, 30),
            ),
            (
                Frequency::Months{number:1, roll: RollDay::EoM {}},
                ndt(2022, 6, 30),
                ndt(2022, 7, 31),
            ),
            (
                Frequency::Months{number:1, roll: RollDay::IMM{}},
                ndt(2022, 6, 15),
                ndt(2022, 7, 20),
            ),
        ];
        for option in options.iter() {
            assert_eq!(option.2, option.0.next_period(&option.1));
        }
    }
}
