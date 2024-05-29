//! Create holiday calendars for financial date manipulation.
//!
//! Calendars are defined by a specific set of holiday dates and generalised work week mask.

use chrono::prelude::*;
use indexmap::set::IndexSet;
use std::collections::{HashSet};
use chrono::{Days, Weekday};
use pyo3::pyclass;

/// Struct for defining a holiday calendar.
#[pyclass]
#[derive(Clone, Default, Debug, PartialEq)]
pub struct HolCal {
    holidays: IndexSet<NaiveDateTime>,
    week_mask: HashSet<Weekday>,
}

/// Enum defining the rule to adjust a non-business day to a business day.
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


impl HolCal {
    pub fn new(holidays: Vec<NaiveDateTime>, week_mask: Vec<u8>) -> Self {
        HolCal {
            holidays: IndexSet::from_iter(holidays),
            week_mask: HashSet::from_iter(week_mask.into_iter().map(|v| Weekday::try_from(v).unwrap())),
        }
    }

    pub fn is_weekday(&self, date: &NaiveDateTime) -> bool {
        !self.week_mask.contains(&date.weekday())
    }

    pub fn is_holiday(&self, date: &NaiveDateTime) -> bool {
        self.holidays.contains(date)
    }

    pub fn is_bus_day(&self, date: &NaiveDateTime) -> bool {
        self.is_weekday(date) && !self.is_holiday(date)
    }
}

fn next_bus_day(date: &NaiveDateTime, cal: &HolCal) -> NaiveDateTime {
    let mut new_date = date.clone();
    while !cal.is_bus_day(&new_date) {
        new_date = new_date + Days::new(1);
    }
    new_date
}

fn prev_bus_day(date: &NaiveDateTime, cal: &HolCal) -> NaiveDateTime {
    let mut new_date = date.clone();
    while !cal.is_bus_day(&new_date) {
        new_date = new_date - Days::new(1);
    }
    new_date
}

fn mod_next_bus_day(date: &NaiveDateTime, cal: &HolCal) -> NaiveDateTime {
    let new_date = next_bus_day(date, cal);
    if new_date.month() != date.month() { prev_bus_day(date, cal) } else { new_date }
}

fn mod_prev_bus_day(date: &NaiveDateTime, cal: &HolCal) -> NaiveDateTime {
    let new_date = prev_bus_day(date, cal);
    if new_date.month() != date.month() { next_bus_day(date, cal) } else { new_date }
}

pub fn adjust(date: &NaiveDateTime, cal:&HolCal, modifier: &Modifier) -> NaiveDateTime {
    match modifier {
        Modifier::Act => date.clone(),
        Modifier::F => next_bus_day(date, cal),
        Modifier::P => prev_bus_day(date, cal),
        Modifier::ModF => mod_next_bus_day(date, cal),
        Modifier::ModP => mod_prev_bus_day(date, cal),
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_hol_cal() -> HolCal {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        HolCal::new(hols, vec![5, 6])
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
        assert!(cal.is_weekday(&hol));  // Monday
        assert!(cal.is_weekday(&no_hol));  //Thursday
        assert!(!cal.is_weekday(&saturday));  // Saturday
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
    fn test_next_bus_day() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = next_bus_day(&hol, &cal);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let sat = NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = next_bus_day(&sat, &cal);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let fri = NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = next_bus_day(&fri, &cal);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap())
    }

    #[test]
    fn test_prev_bus_day() {
        let cal = fixture_hol_cal();
        let hol = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = prev_bus_day(&hol, &cal);
        assert_eq!(prev, NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let fri = NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = prev_bus_day(&fri, &cal);
        assert_eq!(next, NaiveDateTime::parse_from_str("2015-09-04 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap())
    }

    #[test]
    fn test_adjust() {
        let cal = fixture_hol_cal();
        let non_bus = NaiveDateTime::parse_from_str("2024-03-30 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();

        let res = adjust(&non_bus, &cal, &Modifier::F);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-04-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let res = adjust(&non_bus, &cal, &Modifier::P);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-03-29 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let res = adjust(&non_bus, &cal, &Modifier::ModF);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-03-29 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let res = adjust(&non_bus, &cal, &Modifier::Act);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-03-30 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());

        let non_bus = NaiveDateTime::parse_from_str("2024-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let res = adjust(&non_bus, &cal, &Modifier::ModP);
        assert_eq!(res, NaiveDateTime::parse_from_str("2024-12-02 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap());
    }
}
