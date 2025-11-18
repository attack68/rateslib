use crate::scheduling::DateRoll;
use chrono::prelude::*;
use chrono::Days;
use serde::{Deserialize, Serialize};

/// Specifier for date adjustment rules.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum Adjuster {
    /// Actual date without adjustment.
    Actual {},
    /// Following adjustment rule.
    Following {},
    /// Modified following adjustment rule.
    ModifiedFollowing {},
    /// Previous adjustment rule.
    Previous {},
    /// Modified previous adjustment rule.
    ModifiedPrevious {},
    /// Following adjustment rule, enforcing settlement calendar.
    FollowingSettle {},
    /// Modified following adjustment rule, enforcing settlement calendar.
    ModifiedFollowingSettle {},
    /// Previous adjustment rule, enforcing settlement calendar.
    PreviousSettle {},
    /// Modified previous adjustment rule, enforcing settlement calendar.
    ModifiedPreviousSettle {},
    /// A set number of business days, defined by a given calendar,
    /// using calendar lag rules and enforcing settlement calendars.
    BusDaysLagSettle(i32),
    /// A set number of calendar days enforcing settlement calendars, defined by a
    /// given calendar.
    CalDaysLagSettle(i32),
    /// Following adjustment rule except uses actual date for the last date in a vector.
    FollowingExLast {},
    /// Following adjustment rule, enforcing settlement, except uses actual date for the last date in a vector.
    FollowingExLastSettle {},
    /// A set number of business days, enforcing settlement, except uses the period start date for a vector.
    BusDaysLagSettleInAdvance(i32),
}

/// Perform date adjustment according to calendar definitions, i.e. a known [`DateRoll`].
pub trait Adjustment {
    /// Adjust a date under an adjustment rule.
    fn adjust<T: DateRoll>(&self, udate: &NaiveDateTime, calendar: &T) -> NaiveDateTime;

    /// Perform a reverse adjustment to derive potential unadjusted date candidates.
    fn reverse<T: DateRoll>(&self, adate: &NaiveDateTime, calendar: &T) -> Vec<NaiveDateTime>;

    /// Adjust a vector of dates under an adjustment rule;
    fn adjusts<T: DateRoll>(&self, udates: &Vec<NaiveDateTime>, calendar: &T)
        -> Vec<NaiveDateTime>;
}

/// Perform date adjustment according to adjustment rules, i.e. a given [`Adjuster`].
pub trait CalendarAdjustment {
    /// Adjust a date under an adjustment rule.
    fn adjust(&self, udate: &NaiveDateTime, adjuster: &Adjuster) -> NaiveDateTime
    where
        Self: Sized + DateRoll,
    {
        adjuster.adjust(udate, self)
    }

    /// Adjust a vector of dates under an adjustment rule;
    fn adjusts(&self, udates: &Vec<NaiveDateTime>, adjuster: &Adjuster) -> Vec<NaiveDateTime>
    where
        Self: Sized + DateRoll,
    {
        adjuster.adjusts(udates, self)
    }
}

impl Adjustment for Adjuster {
    fn adjust<T: DateRoll>(&self, udate: &NaiveDateTime, calendar: &T) -> NaiveDateTime {
        match self {
            Adjuster::Actual {} => *udate,
            Adjuster::Following {} => calendar.roll_forward_bus_day(udate),
            Adjuster::Previous {} => calendar.roll_backward_bus_day(udate),
            Adjuster::ModifiedFollowing {} => calendar.roll_mod_forward_bus_day(udate),
            Adjuster::ModifiedPrevious {} => calendar.roll_mod_backward_bus_day(udate),
            Adjuster::FollowingSettle {} => calendar.roll_forward_settled_bus_day(udate),
            Adjuster::PreviousSettle {} => calendar.roll_backward_settled_bus_day(udate),
            Adjuster::ModifiedFollowingSettle {} => {
                calendar.roll_forward_mod_settled_bus_day(udate)
            }
            Adjuster::ModifiedPreviousSettle {} => {
                calendar.roll_backward_mod_settled_bus_day(udate)
            }
            Adjuster::BusDaysLagSettle(n) => calendar.lag_bus_days(udate, *n, true),
            Adjuster::CalDaysLagSettle(n) => {
                let adj = if *n < 0 {
                    Adjuster::PreviousSettle {}
                } else {
                    Adjuster::FollowingSettle {}
                };
                calendar.add_cal_days(udate, *n, &adj)
            }
            Adjuster::FollowingExLast {} => calendar.roll_forward_bus_day(udate), // no vector
            Adjuster::FollowingExLastSettle {} => calendar.roll_forward_settled_bus_day(udate), // no vector
            Adjuster::BusDaysLagSettleInAdvance(n) => calendar.lag_bus_days(udate, *n, true), // no vector
        }
    }

    fn reverse<T: DateRoll>(&self, adate: &NaiveDateTime, calendar: &T) -> Vec<NaiveDateTime> {
        match self {
            Adjuster::Actual {} => vec![*adate],
            Adjuster::Following {} => reverse_forward_type(adate, self, calendar),
            Adjuster::Previous {} => reverse_backward_type(adate, self, calendar),
            Adjuster::ModifiedFollowing {} => reverse_modified_type(adate, self, calendar),
            Adjuster::ModifiedPrevious {} => reverse_modified_type(adate, self, calendar),
            Adjuster::FollowingSettle {} => reverse_forward_type(adate, self, calendar),
            Adjuster::PreviousSettle {} => reverse_backward_type(adate, self, calendar),
            Adjuster::ModifiedFollowingSettle {} => reverse_modified_type(adate, self, calendar),
            Adjuster::ModifiedPreviousSettle {} => reverse_modified_type(adate, self, calendar),
            Adjuster::BusDaysLagSettle(n) => {
                if Adjuster::BusDaysLagSettle(0).adjust(adate, calendar) != *adate {
                    return vec![]; // input has no valid reversal
                }
                let date = calendar.add_bus_days(adate, -n, false).unwrap();
                Adjuster::Previous {}.reverse(&date, calendar)
            }
            Adjuster::CalDaysLagSettle(n) => {
                let ret: Vec<NaiveDateTime> = if *n < 0 {
                    Adjuster::PreviousSettle {}.reverse(adate, calendar)
                } else {
                    Adjuster::FollowingSettle {}.reverse(adate, calendar)
                };
                ret.into_iter()
                    .map(|d| calendar.add_cal_days(&d, -n, &Adjuster::Actual {}))
                    .collect()
            }
            Adjuster::FollowingExLast {} => reverse_forward_type(adate, self, calendar), // no vector
            Adjuster::FollowingExLastSettle {} => reverse_forward_type(adate, self, calendar), // no vector
            Adjuster::BusDaysLagSettleInAdvance(n) => {
                if Adjuster::BusDaysLagSettle(0).adjust(adate, calendar) != *adate {
                    return vec![]; // input has no valid reversal
                }
                let date = calendar.add_bus_days(adate, -n, false).unwrap();
                Adjuster::Previous {}.reverse(&date, calendar)
            }, // no vector
        }
    }

    fn adjusts<T: DateRoll>(
        &self,
        udates: &Vec<NaiveDateTime>,
        calendar: &T,
    ) -> Vec<NaiveDateTime> {
        let mut non_vector_adates: Vec<NaiveDateTime> = udates
            .iter()
            .map(|udate| self.adjust(udate, calendar))
            .collect();

        // mutate for vector adjustment
        match self {
            Adjuster::FollowingExLast {} | Adjuster::FollowingExLastSettle {} => {
                non_vector_adates[udates.len() - 1] = udates[udates.len() - 1];
            }
            Adjuster::BusDaysLagSettleInAdvance(_n) => {
                for i in (1..udates.len()).rev() {
                    non_vector_adates[i] = non_vector_adates[i-1];
                }
            }
            _ => {}
        }
        non_vector_adates
    }
}

fn reverse_forward_type<T: DateRoll>(
    adate: &NaiveDateTime,
    adjuster: &Adjuster,
    calendar: &T,
) -> Vec<NaiveDateTime> {
    let mut ret: Vec<NaiveDateTime>;
    if (*adjuster).adjust(adate, calendar) == *adate {
        // adate is valid reversal
        ret = vec![*adate];
    } else {
        // adate is an unadjusted date and is not valid: it has no reversal.
        return vec![];
    }
    let mut date = *adate - Days::new(1);
    while (*adjuster).adjust(&date, calendar) == *adate {
        ret.push(date);
        date = date - Days::new(1);
    }
    ret
}

fn reverse_backward_type<T: DateRoll>(
    adate: &NaiveDateTime,
    adjuster: &Adjuster,
    calendar: &T,
) -> Vec<NaiveDateTime> {
    let mut ret: Vec<NaiveDateTime>;
    if (*adjuster).adjust(adate, calendar) == *adate {
        // adate is valid reversal
        ret = vec![*adate];
    } else {
        // adate is an unadjusted date and is not valid: it has no reversal.
        return vec![];
    }
    let mut date = *adate + Days::new(1);
    while (*adjuster).adjust(&date, calendar) == *adate {
        ret.push(date);
        date = date + Days::new(1);
    }
    ret
}

fn reverse_modified_type<T: DateRoll>(
    adate: &NaiveDateTime,
    adjuster: &Adjuster,
    calendar: &T,
) -> Vec<NaiveDateTime> {
    let mut ret: Vec<NaiveDateTime>;
    if (*adjuster).adjust(adate, calendar) == *adate {
        // adate is valid reversal of itself
        ret = vec![*adate];
    } else {
        // adate is an unadjusted date and is not valid: it has no reversal.
        return vec![];
    }
    let mut date = *adate - Days::new(1);
    let mut adj = (*adjuster).adjust(&date, calendar);
    while adj == *adate && date.month() == adate.month() {
        ret.push(date);
        date = date - Days::new(1);
        adj = (*adjuster).adjust(&date, calendar);
    }
    date = *adate + Days::new(1);
    adj = (*adjuster).adjust(&date, calendar);
    while adj == *adate && date.month() == adate.month() {
        ret.push(date);
        date = date + Days::new(1);
        adj = (*adjuster).adjust(&date, calendar);
    }
    ret
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt, Cal, Calendar};

    fn fixture_hol_cal() -> Cal {
        let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
        Cal::new(hols, vec![5, 6])
    }

    #[test]
    fn test_equality() {
        assert_eq!(Adjuster::Following {}, Adjuster::Following {});
        assert_eq!(Adjuster::BusDaysLagSettle(3), Adjuster::BusDaysLagSettle(3));
        assert_ne!(Adjuster::BusDaysLagSettle(3), Adjuster::BusDaysLagSettle(5));
    }

    #[test]
    fn test_adjusts() {
        let cal = fixture_hol_cal();
        let udates = vec![
            ndt(2015, 9, 4),
            ndt(2015, 9, 5),
            ndt(2015, 9, 6),
            ndt(2015, 9, 7),
        ];
        let result = Adjuster::Following {}.adjusts(&udates, &cal);
        assert_eq!(
            result,
            vec![
                ndt(2015, 9, 4),
                ndt(2015, 9, 8),
                ndt(2015, 9, 8),
                ndt(2015, 9, 8)
            ]
        );
    }

    #[test]
    fn test_adjusts_ex_last() {
        // the last date in the vector is unadjusted
        let cal = fixture_hol_cal();
        let udates = vec![
            ndt(2015, 9, 4),
            ndt(2015, 9, 5),
            ndt(2015, 9, 6),
            ndt(2015, 9, 7),
        ];
        let result = Adjuster::FollowingExLast {}.adjusts(&udates, &cal);
        assert_eq!(
            result,
            vec![
                ndt(2015, 9, 4),
                ndt(2015, 9, 8),
                ndt(2015, 9, 8),
                ndt(2015, 9, 7)
            ]
        );
    }

    #[test]
    fn test_adjusts_in_advance() {
        // the vector is adjusted to in advance
        let cal = fixture_hol_cal();
        let udates = vec![
            ndt(2015, 9, 4),
            ndt(2015, 9, 5),
            ndt(2015, 9, 6),
            ndt(2015, 9, 7),
        ];
        let result = Adjuster::BusDaysLagSettleInAdvance(0).adjusts(&udates, &cal);
        assert_eq!(
            result,
            vec![
                ndt(2015, 9, 4),
                ndt(2015, 9, 4),
                ndt(2015, 9, 8),
                ndt(2015, 9, 8)
            ]
        );
    }

    #[test]
    fn test_reverse() {
        let cal = Cal::new(
            vec![
                ndt(2000, 1, 31),
                ndt(2000, 1, 29),
                ndt(2000, 1, 10),
                ndt(2000, 1, 11),
                ndt(2000, 1, 16),
                ndt(2000, 1, 1),
                ndt(2000, 1, 3),
            ],
            vec![],
        );

        let options: Vec<(NaiveDateTime, Adjuster, Vec<NaiveDateTime>)> = vec![
            // No reversals for holidays
            (ndt(2000, 1, 1), Adjuster::Following {}, vec![]),
            (ndt(2000, 1, 1), Adjuster::Previous {}, vec![]),
            (ndt(2000, 1, 1), Adjuster::ModifiedPrevious {}, vec![]),
            (ndt(2000, 1, 11), Adjuster::CalDaysLagSettle(3), vec![]),
            (ndt(2000, 1, 11), Adjuster::BusDaysLagSettle(3), vec![]),
            // Valid reversals for adjusted dates.
            (
                ndt(2000, 1, 2),
                Adjuster::Following {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::Following {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 29)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::FollowingSettle {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::FollowingSettle {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 29)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::Previous {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 3)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::Previous {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 31)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::PreviousSettle {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 3)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::PreviousSettle {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 31)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::FollowingExLast {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::FollowingExLast {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 29)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::FollowingExLastSettle {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::FollowingExLastSettle {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 29)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::ModifiedFollowing {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::ModifiedFollowing {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 29), ndt(2000, 1, 31)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::ModifiedPrevious {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1), ndt(2000, 1, 3)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::ModifiedPrevious {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 31)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::ModifiedFollowingSettle {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::ModifiedFollowingSettle {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 29), ndt(2000, 1, 31)],
            ),
            (
                ndt(2000, 1, 2),
                Adjuster::ModifiedPreviousSettle {},
                vec![ndt(2000, 1, 2), ndt(2000, 1, 1), ndt(2000, 1, 3)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::ModifiedPreviousSettle {},
                vec![ndt(2000, 1, 30), ndt(2000, 1, 31)],
            ),
            (ndt(2000, 1, 2), Adjuster::Actual {}, vec![ndt(2000, 1, 2)]),
            (
                ndt(2000, 1, 30),
                Adjuster::Actual {},
                vec![ndt(2000, 1, 30)],
            ),
            (
                ndt(2000, 1, 30),
                Adjuster::Actual {},
                vec![ndt(2000, 1, 30)],
            ),
            (
                ndt(2000, 1, 15),
                Adjuster::CalDaysLagSettle(5),
                vec![ndt(2000, 1, 10)],
            ),
            (
                ndt(2000, 1, 12),
                Adjuster::CalDaysLagSettle(5),
                vec![ndt(2000, 1, 7), ndt(2000, 1, 6), ndt(2000, 1, 5)],
            ),
            (
                ndt(2000, 1, 18),
                Adjuster::BusDaysLagSettle(5),
                vec![ndt(2000, 1, 12)],
            ),
            (
                ndt(2000, 1, 17),
                Adjuster::BusDaysLagSettle(5),
                vec![ndt(2000, 1, 9), ndt(2000, 1, 10), ndt(2000, 1, 11)],
            ),
        ];
        for option in options {
            let result = option.1.reverse(&option.0, &Calendar::Cal(cal.clone()));
            assert_eq!(result, option.2)
        }
    }
}
