use chrono::prelude::*;
use chrono::Days;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::cmp::Ordering;

use crate::scheduling::{Adjuster, Adjustment};

/// Simple date adjustment defining business, settleable and holidays and rolling.
pub trait DateRoll {
    /// Returns whether the date is part of the general working week.
    fn is_weekday(&self, date: &NaiveDateTime) -> bool;

    /// Returns whether the date is a specific holiday excluded from the regular working week.
    fn is_holiday(&self, date: &NaiveDateTime) -> bool;

    /// Returns whether the date is valid relative to an associated settlement calendar.
    ///
    /// If the holiday calendar object has no associated settlement calendar this should return `true`
    /// for any date.
    fn is_settlement(&self, date: &NaiveDateTime) -> bool;

    /// Returns whether the date is a business day, i.e. part of the working week and not a holiday.
    fn is_bus_day(&self, date: &NaiveDateTime) -> bool {
        self.is_weekday(date) && !self.is_holiday(date)
    }

    /// Returns whether the date is not a business day, i.e. either not in working week or a specific holiday.
    fn is_non_bus_day(&self, date: &NaiveDateTime) -> bool {
        !self.is_bus_day(date)
    }

    /// Return the `date`, if a business day, or get the next business date after `date`.
    fn roll_forward_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = *date;
        while !self.is_bus_day(&new_date) {
            new_date = new_date + Days::new(1);
        }
        new_date
    }

    /// Return the `date`, if a business day, or get the business day preceding `date`.
    fn roll_backward_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = *date;
        while !self.is_bus_day(&new_date) {
            new_date = new_date - Days::new(1);
        }
        new_date
    }

    /// Return the `date`, if a business day, or get the proceeding business date, without rolling
    /// into a new month.
    fn roll_mod_forward_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.roll_forward_bus_day(date);
        if new_date.month() != date.month() {
            self.roll_backward_bus_day(date)
        } else {
            new_date
        }
    }

    /// Return the `date`, if a business day, or get the proceeding business date, without rolling
    /// into a new month.
    fn roll_mod_backward_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.roll_backward_bus_day(date);
        if new_date.month() != date.month() {
            self.roll_forward_bus_day(date)
        } else {
            new_date
        }
    }

    /// Return the date, if a business day that can be settled, or the proceeding date that is such.
    ///
    /// If the calendar has no associated settlement calendar this is identical to `roll_forward_bus_day`.
    fn roll_forward_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = self.roll_forward_bus_day(date);
        while !self.is_settlement(&new_date) {
            new_date = self.roll_forward_bus_day(&(new_date + Days::new(1)));
        }
        new_date
    }

    /// Return the date, if a business day that can be settled, or the preceding date that is such.
    ///
    /// If the calendar has no associated settlement calendar this is identical to `roll_backward_bus_day`.
    fn roll_backward_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = self.roll_backward_bus_day(date);
        while !self.is_settlement(&new_date) {
            new_date = self.roll_backward_bus_day(&(new_date - Days::new(1)));
        }
        new_date
    }

    /// Return the `date`, if a business day that can be settled, or get the proceeding
    /// such date, without rolling into a new month.
    fn roll_forward_mod_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.roll_forward_settled_bus_day(date);
        if new_date.month() != date.month() {
            self.roll_backward_settled_bus_day(date)
        } else {
            new_date
        }
    }

    /// Return the `date`, if a business day that can be settled, or get the preceding such date, without rolling
    /// into a new month.
    fn roll_backward_mod_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.roll_backward_settled_bus_day(date);
        if new_date.month() != date.month() {
            self.roll_forward_settled_bus_day(date)
        } else {
            new_date
        }
    }

    /// Adjust a date by a number of business days, under lag rules.
    ///
    /// *Note*: if the number of business days is **zero** a non-business day will be rolled
    /// **forwards**.
    ///
    /// *Note*: `settlement` enforcement is handled post date determination. If the number of
    /// business `days` is zero or greater the date is rolled forwards to the nearest settleable
    /// day if not already one.
    /// If the number of business `days` is less than zero then the date is rolled backwards
    /// to the nearest settleable date.
    ///
    /// *Note*: if the given `date` is a non-business date adding or subtracting 1 business
    /// day is equivalent to the rolling forwards or backwards, respectively.
    fn lag_bus_days(&self, date: &NaiveDateTime, days: i32, settlement: bool) -> NaiveDateTime {
        if self.is_bus_day(date) {
            return self.add_bus_days(date, days, settlement).unwrap();
        }
        match days.cmp(&0_i32) {
            Ordering::Equal => self
                .add_bus_days(&self.roll_forward_bus_day(date), 0, settlement)
                .unwrap(),
            Ordering::Less => self
                .add_bus_days(&self.roll_backward_bus_day(date), days + 1, settlement)
                .unwrap(),
            Ordering::Greater => self
                .add_bus_days(&self.roll_forward_bus_day(date), days - 1, settlement)
                .unwrap(),
        }
    }

    /// Add a given number of calendar days to a `date` with the result adjusted to a business day that may or may not
    /// allow `settlement`.
    fn add_cal_days(&self, date: &NaiveDateTime, days: i32, adjuster: &Adjuster) -> NaiveDateTime
    where
        Self: Sized,
    {
        let new_date = if days < 0 {
            *date - Days::new(u64::try_from(-days).unwrap())
        } else {
            *date + Days::new(u64::try_from(days).unwrap())
        };
        adjuster.adjust(&new_date, self)
    }

    /// Add a given number of business days to a `date` with the result adjusted to a business day that may or may
    /// not allow `settlement`.
    ///
    /// *Note*: When adding a positive number of business days the only sensible modifier is
    /// `Modifier::F` and when subtracting business days it is `Modifier::P`.
    fn add_bus_days(
        &self,
        date: &NaiveDateTime,
        days: i32,
        settlement: bool,
    ) -> Result<NaiveDateTime, PyErr> {
        if self.is_non_bus_day(date) {
            return Err(PyValueError::new_err(
                "Cannot add business days to an input `date` that is not a business day.",
            ));
        }
        let mut new_date = *date;
        let mut counter: i32 = 0;
        if days < 0 {
            // then we subtract business days
            while counter > days {
                new_date = self.roll_backward_bus_day(&(new_date - Days::new(1)));
                counter -= 1;
            }
        } else {
            // add business days
            while counter < days {
                new_date = self.roll_forward_bus_day(&(new_date + Days::new(1)));
                counter += 1;
            }
        }

        if !settlement {
            Ok(new_date)
        } else if days < 0 {
            Ok(self.roll_backward_settled_bus_day(&new_date))
        } else {
            Ok(self.roll_forward_settled_bus_day(&new_date))
        }
    }

    /// Return a vector of business dates between a start and end, inclusive.
    fn bus_date_range(
        &self,
        start: &NaiveDateTime,
        end: &NaiveDateTime,
    ) -> Result<Vec<NaiveDateTime>, PyErr> {
        if self.is_non_bus_day(start) || self.is_non_bus_day(end) {
            return Err(PyValueError::new_err("`start` and `end` for a calendar `bus_date_range` must both be valid business days"));
        }
        let mut vec = Vec::new();
        let mut sample_date = *start;
        while sample_date <= *end {
            vec.push(sample_date);
            sample_date = self.add_bus_days(&sample_date, 1, false)?;
        }
        Ok(vec)
    }

    /// Return a vector of calendar dates between a start and end, inclusive
    fn cal_date_range(
        &self,
        start: &NaiveDateTime,
        end: &NaiveDateTime,
    ) -> Result<Vec<NaiveDateTime>, PyErr> {
        let mut vec = Vec::new();
        let mut sample_date = *start;
        while sample_date <= *end {
            vec.push(sample_date);
            sample_date = sample_date + Days::new(1);
        }
        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt, Cal, CalendarAdjustment, UnionCal};

    fn fixture_hol_cal() -> Cal {
        let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
        Cal::new(hols, vec![5, 6])
    }

    #[test]
    fn test_roll_forward_bus_day() {
        let cal = fixture_hol_cal();
        let hol =
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.roll_forward_bus_day(&hol);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let sat =
            NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.roll_forward_bus_day(&sat);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let fri =
            NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.roll_forward_bus_day(&fri);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        )
    }

    #[test]
    fn test_roll_backward_bus_day() {
        let cal = fixture_hol_cal();
        let hol =
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.roll_backward_bus_day(&hol);
        assert_eq!(
            prev,
            NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let fri =
            NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.roll_backward_bus_day(&fri);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        )
    }

    #[test]
    fn test_is_business_day() {
        let cal = fixture_hol_cal();
        let hol =
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol =
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday =
            NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(!cal.is_bus_day(&hol)); // Monday in Hol list
        assert!(cal.is_bus_day(&no_hol)); //Thursday
        assert!(!cal.is_bus_day(&saturday)); // Saturday
    }

    #[test]
    fn test_is_non_business_day() {
        let cal = fixture_hol_cal();
        let hol =
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol =
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday =
            NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(cal.is_non_bus_day(&hol)); // Monday in Hol list
        assert!(!cal.is_non_bus_day(&no_hol)); //Thursday
        assert!(cal.is_non_bus_day(&saturday)); // Saturday
    }

    #[test]
    fn test_lag_bus_days() {
        let cal = fixture_hol_cal();
        let result = cal.lag_bus_days(&ndt(2015, 9, 7), 1, true);
        assert_eq!(result, ndt(2015, 9, 8));

        let result = cal.lag_bus_days(&ndt(2025, 2, 15), -1, true);
        assert_eq!(result, ndt(2025, 2, 14));

        let result = cal.lag_bus_days(&ndt(2015, 9, 7), 0, true);
        assert_eq!(result, ndt(2015, 9, 8))
    }

    #[test]
    fn test_add_days() {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let settle =
            vec![
                NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            ];
        let hcal = Cal::new(hols, vec![5, 6]);
        let scal = Cal::new(settle, vec![5, 6]);
        let cal = UnionCal::new(vec![hcal], vec![scal].into());

        // without settlement constraint 11th is a valid forward roll date
        let tue =
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.add_cal_days(&tue, 2, &Adjuster::Following {});
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        // with settlement constraint 11th is invalid. Pushed to 14th over weekend.-
        let tue =
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.add_cal_days(&tue, 2, &Adjuster::FollowingSettle {});
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-14 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        // without settlement constraint 11th is a valid previous roll date
        let tue =
            NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_cal_days(&tue, -2, &Adjuster::Previous {});
        assert_eq!(
            prev,
            NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        // with settlement constraint 11th is invalid. Pushed to 9th over holiday.
        let tue =
            NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_cal_days(&tue, -2, &Adjuster::PreviousSettle {});
        assert_eq!(
            prev,
            NaiveDateTime::parse_from_str("2015-09-09 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );
    }

    #[test]
    fn test_add_bus_days() {
        let hols = vec![ndt(2015, 9, 8), ndt(2015, 9, 10)];
        let settle = vec![ndt(2015, 9, 11)];

        let hcal = Cal::new(hols, vec![5, 6]);
        let scal = Cal::new(settle, vec![5, 6]);
        let cal = UnionCal::new(vec![hcal], vec![scal].into());

        // without settlement constraint 11th is a valid forward roll date
        let mon = ndt(2015, 9, 7);
        let next = cal.add_bus_days(&mon, 2, false).unwrap();
        assert_eq!(next, ndt(2015, 9, 11));

        // with settlement constraint 11th is invalid. Pushed to 14th over weekend.-
        let next = cal.add_bus_days(&mon, 2, true).unwrap();
        assert_eq!(next, ndt(2015, 9, 14));

        // without settlement constraint 11th is a valid previous roll date
        let tue = ndt(2015, 9, 15);
        let prev = cal.add_bus_days(&tue, -2, false).unwrap();
        assert_eq!(prev, ndt(2015, 9, 11));

        // with settlement constraint 11th is invalid. Pushed to 9th over holiday.
        let prev = cal.add_bus_days(&tue, -2, true).unwrap();
        assert_eq!(prev, ndt(2015, 9, 9));
    }

    #[test]
    fn test_add_bus_days_error() {
        let cal = fixture_hol_cal();
        match cal.add_bus_days(&ndt(2015, 9, 7), 3, true) {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    #[test]
    fn test_add_bus_days_with_settlement() {
        let cal = Cal::new(vec![ndt(2024, 6, 5)], vec![5, 6]);
        let settle = Cal::new(vec![ndt(2024, 6, 4), ndt(2024, 6, 6)], vec![5, 6]);
        let union = UnionCal::new(vec![cal], Some(vec![settle]));

        let result = union.add_bus_days(&ndt(2024, 6, 4), 1, false).unwrap();
        assert_eq!(result, ndt(2024, 6, 6)); //
        let result = union.add_bus_days(&ndt(2024, 6, 4), 1, true).unwrap();
        assert_eq!(result, ndt(2024, 6, 7)); //

        let result = union.add_bus_days(&ndt(2024, 6, 6), -1, false).unwrap();
        assert_eq!(result, ndt(2024, 6, 4)); //
        let result = union.add_bus_days(&ndt(2024, 6, 6), -1, true).unwrap();
        assert_eq!(result, ndt(2024, 6, 3)); //
    }

    #[test]
    fn test_rolls() {
        let cal = fixture_hol_cal();
        let udates = vec![
            ndt(2015, 9, 4),
            ndt(2015, 9, 5),
            ndt(2015, 9, 6),
            ndt(2015, 9, 7),
        ];
        let result = cal.adjusts(&udates, &Adjuster::Following {});
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
    fn test_lag_bus_days_zero_with_settlement() {
        // Test ModifiedPrevious and ModifiedPreviousSettle from the book diagram
        let cal = Cal::new(vec![ndt(2000, 6, 27)], vec![]);
        let settle = Cal::new(vec![ndt(2000, 6, 26), ndt(2000, 6, 28)], vec![]);
        let uni = UnionCal::new(vec![cal], Some(vec![settle]));

        // adding zero bus days not settleable yields 28th June
        assert_eq!(
            ndt(2000, 6, 28),
            uni.lag_bus_days(&ndt(2000, 6, 27), 0, false)
        );

        // adding zero bus days settleable yields 29th June
        assert_eq!(
            ndt(2000, 6, 29),
            uni.lag_bus_days(&ndt(2000, 6, 27), 0, true)
        );

        // adding zero bus days not settleable yields 28th June
        assert_eq!(
            ndt(2000, 6, 28),
            uni.lag_bus_days(&ndt(2000, 6, 28), 0, false)
        );

        // adding zero bus days settleable yields 29th June
        assert_eq!(
            ndt(2000, 6, 29),
            uni.lag_bus_days(&ndt(2000, 6, 28), 0, true)
        );
    }
}
