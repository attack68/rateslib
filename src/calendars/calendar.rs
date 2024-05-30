//! Provides functionality to create business day calendars and perform financial date manipulation.
//!
//! ### Basic usage
//!
//! The `Cal` struct allows the construction of a single business day calendar, e.g.
//! a particular currency calendar. The below constructs two separate calendars, one for London holidays and
//! one for Tokyo holidays in 2017.
//!
//! ```rust
//! // UK Monday 1st May Bank Holiday
//! let ldn = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! // Japan Constitution Memorial Day, Greenery Day, Children's Day
//! let tky = Cal::new(vec![ndt(2017, 5, 3), ndt(2017, 5, 4), ndt(2017, 5, 5)], vec![5, 6]);
//! ```
//! These calendars are used to manipulate dates e.g.
//!
//! ```rust
//! let date = ndt(2017, 4, 28);  // Friday 28th April 2017
//! let spot = ldn.add_bus_days(&date, 2, &Modifier::F, true);
//! // Wednesday 3rd May 2017, observing the holiday.
//! ```
//!
//! ### Combination usage
//!
//! For use with multi-currency products calendars often need to be combined.
//!
//! ```rust
//! let ldn_tky = UnionCal::new(vec![ldn, tky], None);
//! let spot = ldn_tky.add_bus_days(&date, 2, &Modifier::F, true);
//! // Monday 8th May 2017, observing all holidays.


use chrono::prelude::*;
use indexmap::set::IndexSet;
use std::collections::{HashSet};
use chrono::{Days, Weekday};
use pyo3::pyclass;

/// Define a single, business day calendar.
///
/// A business day calendar is formed of 2 components:
///
/// - `week_mask`: which defines the days of the week that are not general business days. In Western culture these
///   are typically `[5, 6]` for Saturday and Sunday.
/// - `holidays`: which defines specific dates that may be exceptions to the general working week, and cannot be
///   business days.
///
#[pyclass]
#[derive(Clone, Default, Debug, PartialEq)]
pub struct Cal {
    holidays: IndexSet<NaiveDateTime>,
    week_mask: HashSet<Weekday>,
}

impl Cal {

    /// Create a calendar.
    ///
    /// `holidays` provide a vector of dates that cannot be business days. `week_mask` is a vector of days
    /// (0=Mon,.., 6=Sun) that are excluded from the working week.
    pub fn new(holidays: Vec<NaiveDateTime>, week_mask: Vec<u8>) -> Self {
        Cal {
            holidays: IndexSet::from_iter(holidays),
            week_mask: HashSet::from_iter(week_mask.into_iter().map(|v| Weekday::try_from(v).unwrap())),
        }
    }
}


/// Define a business day calendar which is the potential union of multiple calendars,
/// with the additional constraint of also ensuring settlement compliance with one or more
/// other calendars.
///
/// When the union of a business day calendar is observed the following are true:
///
/// - a weekday is such if it is a weekday in all calendars.
/// - a holiday is such if it is a holiday in any calendar.
/// - a business day is such if it is a business day in all calendars.
///
/// A business day is defined as allowing settlement relative to an associated calendar if:
///
/// - the date in question is also a business day in the associated settlement calendar.
#[pyclass]
#[derive(Clone, Default, Debug, PartialEq)]
pub struct UnionCal {
    calendars: Vec<Cal>,
    settlement_calendars: Option<Vec<Cal>>,
}

impl UnionCal {
    pub fn new(calendars: Vec<Cal>, settlement_calendars: Option<Vec<Cal>>) -> Self {
        UnionCal { calendars, settlement_calendars }
    }
}

/// A trait to control business day management and date rolling.
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

    /// Return the date, if a business day, or get the proceeding business date.
    fn next_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = date.clone();
        while !self.is_bus_day(&new_date) {
            new_date = new_date + Days::new(1);
        }
        new_date
    }

    /// Return the date, if a business day that can be settled, or the proceeding date that is such.
    fn next_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = self.next_bus_day(date);
        while !self.is_settlement(&new_date) {
            new_date = self.next_bus_day(&(new_date + Days::new(1)));
        }
        new_date
    }

    /// Return the date, if a business day, or get the preceding business date.
    fn prev_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = date.clone();
        while !self.is_bus_day(&new_date) {
            new_date = new_date - Days::new(1);
        }
        new_date
    }

    /// Return the date, if a business day that can be settled, or the preceding date that is such.
    fn prev_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut new_date = self.prev_bus_day(date);
        while !self.is_settlement(&new_date) {
            new_date = self.prev_bus_day(&(new_date - Days::new(1)));
        }
        new_date
    }

    /// Return the date, if a business day, or get the proceeding business date, without rolling
    /// into a new month.
    fn mod_next_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.next_bus_day(date);
        if new_date.month() != date.month() { self.prev_bus_day(date) } else { new_date }
    }

    /// Return the date, if a business day that can be settled, or get the proceeding such date, without rolling
    /// into a new month.
    fn mod_next_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.next_settled_bus_day(date);
        if new_date.month() != date.month() { self.prev_settled_bus_day(date) } else { new_date }
    }

    /// Return the date, if a business day, or get the proceeding business date, without rolling
    /// into a new month.
    fn mod_prev_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.prev_bus_day(date);
        if new_date.month() != date.month() { self.next_bus_day(date) } else { new_date }
    }

    /// Return the date, if a business day that can be settled, or get the preceding such date, without rolling
    /// into a new month.
    fn mod_prev_settled_bus_day(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let new_date = self.prev_settled_bus_day(date);
        if new_date.month() != date.month() { self.next_settled_bus_day(date) } else { new_date }
    }

    /// Add a given number of calendar days to a `date` with the result adjusted to a business day that may or may not
    /// allow `settlement`.
    fn add_days(&self, date: &NaiveDateTime, days: i8, modifier: &Modifier, settlement: bool) -> NaiveDateTime
    where Self: Sized
    {
        let new_date;
        if days < 0 {
            new_date = *date - Days::new(u64::try_from(-days).unwrap())
        } else {
            new_date = *date + Days::new(u64::try_from(days).unwrap())
        }
        self.adjust(&new_date, modifier, settlement)
    }

    /// Add a given number of business days to a `date` with the result adjusted to a business day that may or may
    /// not allow `settlement`.
    fn add_bus_days(&self, date: &NaiveDateTime, days: i8, modifier: &Modifier, settlement: bool) -> NaiveDateTime
    where Self: Sized
    {
        let mut new_date = date.clone();
        let mut counter: i8 = 0;
        if days < 0 {  // then we subtract business days
            while counter > days {
                new_date = self.prev_bus_day(&(new_date - Days::new(1)));
                counter -= 1;
            }
        } else {  // add business days
            while counter < days {
                new_date = self.next_bus_day(&(new_date + Days::new(1)));
                counter += 1;
            }
        }
        self.adjust(&new_date, modifier, settlement)
    }


    /// Adjust a date under a date roll modifier, either to a business day enforcing settlement or a business day that
    /// may not allow settlement.
    fn adjust(&self, date: &NaiveDateTime, modifier: &Modifier, settlement: bool) -> NaiveDateTime
    where Self: Sized
    {
        if settlement {
            adjust_with_settlement(date, self, modifier)
        } else {
            adjust_without_settlement(date, self, modifier)
        }
    }

}

impl DateRoll for Cal {

    fn is_weekday(&self, date: &NaiveDateTime) -> bool {
        !self.week_mask.contains(&date.weekday())
    }

    fn is_holiday(&self, date: &NaiveDateTime) -> bool {
        self.holidays.contains(date)
    }

    fn is_settlement(&self, _date: &NaiveDateTime) -> bool {
        true
    }

}

impl DateRoll for UnionCal {

    fn is_weekday(&self, date: &NaiveDateTime) -> bool {
        self.calendars.iter().all(|cal| cal.is_weekday(date))
    }

    fn is_holiday(&self, date: &NaiveDateTime) -> bool {
        self.calendars.iter().any(|cal| cal.is_holiday(date))
    }

    fn is_settlement(&self, date: &NaiveDateTime) -> bool {
        match &self.settlement_calendars {
            None => true,
            Some(cals) => !cals.iter().any(|cal| cal.is_holiday(date))
        }
    }

}

/// Enum defining the rule to adjust a non-business day to a business day.
#[derive(Copy, Clone)]
pub enum Modifier {
    /// Actual: date is unchanged, even if it is a non-business day.
    Act,
    /// Following: date is rolled to the next business day.
    F,
    /// Modified following: date is rolled to the next except if it changes month.
    ModF,
    /// Previous: date is rolled to the previous business day.
    P,
    /// Modified previous: date is rolled to the previous except if it changes month.
    ModP,
}

fn adjust_with_settlement(date: &NaiveDateTime, cal: &dyn DateRoll, modifier: &Modifier) -> NaiveDateTime {
    match modifier {
        Modifier::Act => date.clone(),
        Modifier::F => cal.next_settled_bus_day(date),
        Modifier::P => cal.prev_settled_bus_day(date),
        Modifier::ModF => cal.mod_next_settled_bus_day(date),
        Modifier::ModP => cal.mod_prev_settled_bus_day(date),
    }
}

fn adjust_without_settlement(date: &NaiveDateTime, cal: &dyn DateRoll, modifier: &Modifier) -> NaiveDateTime {
    match modifier {
        Modifier::Act => date.clone(),
        Modifier::F => cal.next_bus_day(date),
        Modifier::P => cal.prev_bus_day(date),
        Modifier::ModF => cal.mod_next_bus_day(date),
        Modifier::ModP => cal.mod_prev_bus_day(date),
    }
}

/// Create a `NaiveDateTime` with default null time.
///
/// Panics if date values are invalid.
pub fn ndt(year: i32, month: u32, day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(year, month, day).expect("`year`, `month` `day` are invalid.").and_hms_opt(0,0,0).unwrap()
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_hol_cal() -> Cal {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),  // saturday
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),  // monday
        ];
        Cal::new(hols, vec![5, 6])
    }

    #[test]
    fn test_is_holiday() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol = NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday = NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(cal.is_holiday(&hol));  // In hol list
        assert!(!cal.is_holiday(&no_hol));  // Not in hol list
        assert!(!cal.is_holiday(&saturday));  // Not in hol list
    }

    #[test]
    fn test_is_weekday() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol = NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday = NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let sunday = NaiveDateTime::parse_from_str("2024-01-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(cal.is_weekday(&hol));  // Monday
        assert!(cal.is_weekday(&no_hol));  //Thursday
        assert!(!cal.is_weekday(&saturday));  // Saturday
        assert!(!cal.is_weekday(&sunday));  // Sunday
    }

    #[test]
    fn test_is_business_day() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol = NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday = NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(!cal.is_bus_day(&hol));  // Monday in Hol list
        assert!(cal.is_bus_day(&no_hol));  //Thursday
        assert!(!cal.is_bus_day(&saturday));  // Saturday
    }

    #[test]
    fn test_is_non_business_day() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol = NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday = NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(cal.is_non_bus_day(&hol));  // Monday in Hol list
        assert!(!cal.is_non_bus_day(&no_hol));  //Thursday
        assert!(cal.is_non_bus_day(&saturday));  // Saturday
    }

    #[test]
    fn test_next_bus_day() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.next_bus_day(&hol);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let sat = NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.next_bus_day(&sat);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let fri = NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.next_bus_day(&fri);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap())
    }

    #[test]
    fn test_prev_bus_day() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.prev_bus_day(&hol);
        assert_eq!(prev, NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let fri = NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.prev_bus_day(&fri);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap())
    }

    #[test]
    fn test_adjust_with_settlement() {
        let cal = fixture_hol_cal();
        let non_bus = NaiveDateTime::parse_from_str("2024-03-30 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();

        let res = adjust_with_settlement(&non_bus, &cal, &Modifier::F);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-04-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let res = adjust_with_settlement(&non_bus, &cal, &Modifier::P);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-03-29 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let res = adjust_with_settlement(&non_bus, &cal, &Modifier::ModF);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-03-29 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let res = adjust_with_settlement(&non_bus, &cal, &Modifier::Act);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-03-30 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let non_bus = NaiveDateTime::parse_from_str("2024-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let res = adjust_with_settlement(&non_bus, &cal, &Modifier::ModP);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-12-02 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());
    }

    fn fixture_hol_cal2() -> Cal {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-09 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        Cal::new(hols, vec![5, 6])
    }

    #[test]
    fn test_union_cal() {
        let cal1 = fixture_hol_cal();
        let cal2 = fixture_hol_cal2();
        let ucal = UnionCal::new(vec![cal1, cal2], None);

        let sat = NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = ucal.next_bus_day(&sat);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());
    }

    #[test]
    fn test_union_cal_with_settle() {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-09 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let scal = Cal::new(hols, vec![5, 6]);
        let cal = Cal::new(vec![], vec![5,6]);
        let ucal = UnionCal::new(vec![cal], vec![scal].into());


        let mon = NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = ucal.next_bus_day(&mon);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());
    }

    #[test]
    fn test_add_days() {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let settle = vec![
            NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let hcal = Cal::new(hols, vec![5, 6]);
        let scal = Cal::new(settle, vec![5,6]);
        let cal = UnionCal::new(vec![hcal], vec![scal].into());

        // without settlement constraint 11th is a valid forward roll date
        let tue = NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.add_days(&tue, 2, &Modifier::F, false);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        // with settlement constraint 11th is invalid. Pushed to 14th over weekend.-
        let tue = NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.add_days(&tue, 2, &Modifier::F, true);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-14 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        // without settlement constraint 11th is a valid previous roll date
        let tue = NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_days(&tue, -2, &Modifier::P, false);
        assert_eq!(prev, NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        // with settlement constraint 11th is invalid. Pushed to 9th over holiday.
        let tue = NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_days(&tue, -2, &Modifier::P, true);
        assert_eq!(prev, NaiveDateTime::parse_from_str("2015-09-09 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());
    }

    #[test]
    fn test_add_bus_days() {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let settle = vec![
            NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let hcal = Cal::new(hols, vec![5, 6]);
        let scal = Cal::new(settle, vec![5,6]);
        let cal = UnionCal::new(vec![hcal], vec![scal].into());

        // without settlement constraint 11th is a valid forward roll date
        let mon = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.add_bus_days(&mon, 2, &Modifier::F, false);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        // with settlement constraint 11th is invalid. Pushed to 14th over weekend.-
        let mon = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.add_bus_days(&mon, 2, &Modifier::F, true);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-14 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        // without settlement constraint 11th is a valid previous roll date
        let tue = NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_bus_days(&tue, -2, &Modifier::P, false);
        assert_eq!(prev, NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        // with settlement constraint 11th is invalid. Pushed to 9th over holiday.
        let tue = NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_bus_days(&tue, -2, &Modifier::P, true);
        assert_eq!(prev, NaiveDateTime::parse_from_str("2015-09-09 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());
    }

    #[test]
    fn test_docstring() {
        let ldn = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);  // UK Monday 1st May Bank Holiday
        let tky = Cal::new(vec![ndt(2017, 5, 3), ndt(2017, 5, 4), ndt(2017, 5, 5)], vec![5, 6]);

        let date = ndt(2017, 4, 28);  // Friday 28th April 2017
        let spot = ldn.add_bus_days(&date, 2, &Modifier::F, true);
        assert_eq!(spot, ndt(2017, 5, 3));

        let ldn_tky = UnionCal::new(vec![ldn, tky], None);
        let spot = ldn_tky.add_bus_days(&date, 2, &Modifier::F, true);
        assert_eq!(spot, ndt(2017, 5, 8));
    }

}
