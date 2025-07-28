use crate::scheduling::DateRoll;
use chrono::prelude::*;

/// A list of rules for performing date adjustment.
#[derive(Debug, Copy, Clone, PartialEq)]
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
    BusDaysLagSettle { number: i32 },
    /// A set number of calendar days enforcing settlement calendars, defined by a
    /// given calendar.
    CalDaysLagSettle { number: i32 },
}

/// Perform date adjustment according to calendar definitions, i.e. a known [`DateRoll`].
pub trait Adjustment {
    /// Adjust a date under an adjustment rule.
    fn adjust<T: DateRoll>(&self, udate: &NaiveDateTime, calendar: &T) -> NaiveDateTime;

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
            Adjuster::BusDaysLagSettle { number: n } => calendar.lag_bus_days(udate, *n, true),
            Adjuster::CalDaysLagSettle { number: n } => {
                let adj = if *n < 0 {
                    Adjuster::PreviousSettle {}
                } else {
                    Adjuster::FollowingSettle {}
                };
                calendar.add_cal_days(udate, *n, &adj)
            }
        }
    }

    fn adjusts<T: DateRoll>(
        &self,
        udates: &Vec<NaiveDateTime>,
        calendar: &T,
    ) -> Vec<NaiveDateTime> {
        match self {
            _ => udates
                .iter()
                .map(|udate| self.adjust(udate, calendar))
                .collect(),
        }
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt, Cal};

    fn fixture_hol_cal() -> Cal {
        let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
        Cal::new(hols, vec![5, 6])
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
}
