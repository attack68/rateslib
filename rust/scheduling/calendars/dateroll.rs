// SPDX-License-Identifier: LicenseRef-Rateslib-Dual
//
// Copyright (c) 2026 Siffrorna Technology Limited
// This code cannot be used or copied externally
//
// Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
// Source-available, not open source.
//
// See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
// and/or contact info (at) rateslib (dot) com
////////////////////////////////////////////////////////////////////////////////////////////////////

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

    /// Print a representation of the month of the object.
    fn print_month(&self, year: i32, month: u8) -> String {
        let _map: Vec<String> = vec![
            format!("        January {}\n", year),
            format!("       February {}\n", year),
            format!("          March {}\n", year),
            format!("          April {}\n", year),
            format!("            May {}\n", year),
            format!("           June {}\n", year),
            format!("           July {}\n", year),
            format!("         August {}\n", year),
            format!("      September {}\n", year),
            format!("        October {}\n", year),
            format!("       November {}\n", year),
            format!("       December {}\n", year),
        ];
        let mut output = _map[(month - 1) as usize].clone();
        output += "Su Mo Tu We Th Fr Sa\n";

        let month_obj = Month::try_from(month).unwrap();
        let days: u8 = month_obj.num_days(year).unwrap();
        let weekday = NaiveDate::from_ymd_opt(year, month.into(), 1)
            .unwrap()
            .weekday()
            .num_days_from_monday();
        let idx_start: u32 = (weekday + 1) % 7;

        let mut arr: [String; 42] = std::array::from_fn(|_| String::from("  "));
        for i in 0..days {
            let date = NaiveDate::from_ymd_opt(year, month.into(), (i + 1).into())
                .expect("`year`, `month` `day` are invalid.")
                .and_hms_opt(0, 0, 0)
                .unwrap();
            let s: String = {
                if self.is_bus_day(&date) && self.is_settlement(&date) {
                    format!("{:>2}", i + 1)
                } else if self.is_bus_day(&date) && !self.is_settlement(&date) {
                    " X".to_string()
                } else if !self.is_bus_day(&date)
                    && matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
                {
                    " .".to_string()
                } else {
                    " *".to_string()
                }
            };
            let index: u32 = i as u32 + idx_start;
            arr[index as usize] = s;
        }

        for row in 0..6 {
            output += &format!(
                "{} {} {} {} {} {} {}\n",
                &arr[row * 7],
                &arr[row * 7 + 1],
                &arr[row * 7 + 2],
                &arr[row * 7 + 3],
                &arr[row * 7 + 4],
                &arr[row * 7 + 5],
                &arr[row * 7 + 6]
            );
        }
        output
    }

    /// Print a representation of a year of the object.
    fn print_year(&self, year: i32) -> String {
        let mut data: Vec<Vec<String>> = vec![];
        for i in 1..13 {
            data.push(
                self.print_month(year, i)
                    .lines()
                    .map(|s| s.to_string())
                    .collect(),
            );
        }
        let mut output = "\n".to_string();
        for i in 0..8 {
            output += &format!(
                "{}   {}   {}   {}\n",
                data[0][i], data[3][i], data[6][i], data[9][i]
            );
        }
        for i in 0..8 {
            output += &format!(
                "{}   {}   {}   {}\n",
                data[1][i], data[4][i], data[7][i], data[10][i]
            );
        }
        for i in 0..8 {
            output += &format!(
                "{}   {}   {}   {}\n",
                data[2][i], data[5][i], data[8][i], data[11][i]
            );
        }
        output += "Legend:\n";
        output += "'1-31': Settleable business day         'X': Non-settleable business day\n";
        output += "   '.': Non-business weekend            '*': Non-business day\n";
        output
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

    #[test]
    fn test_print_month() {
        let cal = Cal::new(vec![ndt(2026, 1, 1), ndt(2026, 1, 19)], vec![5, 6]);
        let result = cal.print_month(2026, 1);
        let raw_output = r#"        January 2026
Su Mo Tu We Th Fr Sa
             *  2  .
 .  5  6  7  8  9  .
 . 12 13 14 15 16  .
 .  * 20 21 22 23  .
 . 26 27 28 29 30  .
$$$$$$$$$$$$$$$$$$$$
"#;
        let expected = raw_output.replace("$", " ");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_print_year() {
        let cal = Cal::new(vec![ndt(2026, 1, 1), ndt(2026, 1, 19)], vec![5, 6]);
        let result = cal.print_year(2026);
        println!("{}", result);
        let raw_output = r#"
        January 2026             April 2026              July 2026           October 2026
Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
             *  2  .             1  2  3  .             1  2  3  .                1  2  .
 .  5  6  7  8  9  .    .  6  7  8  9 10  .    .  6  7  8  9 10  .    .  5  6  7  8  9  .
 . 12 13 14 15 16  .    . 13 14 15 16 17  .    . 13 14 15 16 17  .    . 12 13 14 15 16  .
 .  * 20 21 22 23  .    . 20 21 22 23 24  .    . 20 21 22 23 24  .    . 19 20 21 22 23  .
 . 26 27 28 29 30  .    . 27 28 29 30          . 27 28 29 30 31       . 26 27 28 29 30  .
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
       February 2026               May 2026            August 2026          November 2026
Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
 .  2  3  4  5  6  .                   1  .                      .    .  2  3  4  5  6  .
 .  9 10 11 12 13  .    .  4  5  6  7  8  .    .  3  4  5  6  7  .    .  9 10 11 12 13  .
 . 16 17 18 19 20  .    . 11 12 13 14 15  .    . 10 11 12 13 14  .    . 16 17 18 19 20  .
 . 23 24 25 26 27  .    . 18 19 20 21 22  .    . 17 18 19 20 21  .    . 23 24 25 26 27  .
                        . 25 26 27 28 29  .    . 24 25 26 27 28  .    . 30$$$$$$$$$$$$$$$
                        .                      . 31$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
          March 2026              June 2026         September 2026          December 2026
Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa   Su Mo Tu We Th Fr Sa
 .  2  3  4  5  6  .       1  2  3  4  5  .          1  2  3  4  .          1  2  3  4  .
 .  9 10 11 12 13  .    .  8  9 10 11 12  .    .  7  8  9 10 11  .    .  7  8  9 10 11  .
 . 16 17 18 19 20  .    . 15 16 17 18 19  .    . 14 15 16 17 18  .    . 14 15 16 17 18  .
 . 23 24 25 26 27  .    . 22 23 24 25 26  .    . 21 22 23 24 25  .    . 21 22 23 24 25  .
 . 30 31                . 29 30                . 28 29 30             . 28 29 30 31$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Legend:
'1-31': Settleable business day         'X': Non-settleable business day
   '.': Non-business weekend            '*': Non-business day
"#;
        let expected = raw_output.replace("$", " ");

        let result_lines: Vec<&str> = result.lines().collect();
        let expected_lines: Vec<&str> = expected.lines().collect();
        for i in 0..result_lines.len() {
            assert_eq!(expected_lines[i], result_lines[i]);
        }
    }
}
