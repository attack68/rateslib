use crate::scheduling::{is_leap_year, RollDay};
use chrono::prelude::*;
use pyo3::pyclass;

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
    use crate::scheduling::ndt;

    //     fn fixture_hol_cal() -> Cal {
    //         let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
    //         Cal::new(hols, vec![5, 6])
    //     }

    #[test]
    fn test_is_divisible_months() {
        // test true
        let s1 = ndt(2022, 2, 2);
        let e1 = ndt(2022, 6, 6);
        assert_eq!(true, Frequency::BiMonthly.is_divisible(&s1, &e1));

        // test false
        let s2 = ndt(2022, 2, 2);
        let e2 = ndt(2022, 9, 6);
        assert_eq!(false, Frequency::Quarterly.is_divisible(&s2, &e2));
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
}
