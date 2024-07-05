//! Utilities to create business day calendars and perform financial date manipulation.
//!
//! ### Basic usage
//!
//! The `Cal` struct allows the construction of a single business day calendar, e.g.
//! a particular currency calendar. The below constructs two separate calendars,
//! one for some London holidays and
//! one for some Tokyo holidays in 2017.
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
//! let spot = ldn.add_bus_days(&date, 2, true)?;
//! // Wednesday 3rd May 2017, observing the holiday.
//! ```
//!
//! ### Combination usage
//!
//! For use with multi-currency products calendars often need to be combined.
//!
//! ```rust
//! let ldn_tky = UnionCal::new(vec![ldn, tky], None);
//! let spot = ldn_tky.add_bus_days(&date, 2, true)?;
//! // Monday 8th May 2017, observing all holidays.
//! ```
//!
//! Particularly when adjusting for FX transaction calendars the non-USD calendars may be used
//! for date determination but the US calendar is used to validate eligible settlement.
//! This is also a union of calendars but it is enforced via the `settlement_calendars` field.
//!
//! ```rust
//! let tgt = Cal::new(vec![], vec![5, 6]);
//! let nyc = Cal::new(vec![ndt(2023, 6, 19)], vec![5, 6]);  // Juneteenth Holiday
//! let tgt__nyc = UnionCal::new(vec![tgt], vec![nyc].into());
//! ```
//!
//! The spot (T+2) date as measured from Friday 16th June 2023 ignores the US calendar for date
//! determination and allows Tuesday 20th June 2023 since the US holiday is on the Monday.
//!
//! ```rust
//! let date = ndt(2023, 6, 16);  // Friday 16th June 2023
//! let spot = tgt__nyc.add_bus_days(&date, 2, true)?;
//! // Tuesday 20th June 2023, ignoring the US holiday on Monday.
//! ```
//!
//! On the other hand as measured from Thursday 15th June 2023 the spot cannot be on the Monday
//! when `settlement` is enforced over the US calendar.
//!
//! ```rust
//! let date = ndt(2023, 6, 15);  // Thursday 15th June 2023
//! let spot = tgt__nyc.add_bus_days(&date, 2, true)?;
//! // Tuesday 20th June 2023, enforcing no settlement on US holiday.
//! ```
//!
//! If `settlement` is not enforced spot can be set as the Monday for this calendar, since it is
//! not a European holiday.
//!
//! ```rust
//! let spot = tgt__nyc.add_bus_days(&date, 2, false)?;
//! // Monday 19th June 2023, ignoring the US holiday settlement requirement.
//! ```

use crate::json::JSON;
use chrono::prelude::*;
use chrono::{Days, Weekday};
use indexmap::set::IndexSet;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

/// A business day calendar with a singular list of holidays.
///
/// A business day calendar is formed of 2 components:
///
/// - `week_mask`: which defines the days of the week that are not general business days. In Western culture these
///   are typically `[5, 6]` for Saturday and Sunday.
/// - `holidays`: which defines specific dates that may be exceptions to the general working week, and cannot be
///   business days.
///
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cal {
    pub(crate) holidays: IndexSet<NaiveDateTime>,
    pub(crate) week_mask: HashSet<Weekday>,
    // pub(crate) meta: Vec<String>,
}

impl JSON for Cal {}

impl Cal {
    /// Create a calendar.
    ///
    /// `holidays` provide a vector of dates that cannot be business days. `week_mask` is a vector of days
    /// (0=Mon,.., 6=Sun) that are excluded from the working week.
    pub fn new(
        holidays: Vec<NaiveDateTime>,
        week_mask: Vec<u8>,
        // rules: Vec<&str>
    ) -> Self {
        Cal {
            holidays: IndexSet::from_iter(holidays),
            week_mask: HashSet::from_iter(
                week_mask.into_iter().map(|v| Weekday::try_from(v).unwrap()),
            ),
            // meta: rules.into_iter().map(|x| x.to_string()).collect(),
        }
    }
}

/// A business day calendar which is the potential union of multiple calendars,
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
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct UnionCal {
    pub(crate) calendars: Vec<Cal>,
    pub(crate) settlement_calendars: Option<Vec<Cal>>,
}

impl UnionCal {
    pub fn new(calendars: Vec<Cal>, settlement_calendars: Option<Vec<Cal>>) -> Self {
        UnionCal {
            calendars,
            settlement_calendars,
        }
    }
}

impl JSON for UnionCal {}

/// Used to control business day management and date rolling.
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

    /// Adjust a date under a date roll `modifier`, either to a business day enforcing `settlement` or a
    /// business day that may not allow settlement.
    ///
    /// *Note*: if the `modifier` is *'Act'*, then a business day may not be returned and the `settlement` flag
    /// is disregarded - it is ambiguous in this case whether to move forward or backward datewise.
    fn roll(&self, date: &NaiveDateTime, modifier: &Modifier, settlement: bool) -> NaiveDateTime
    where
        Self: Sized,
    {
        if settlement {
            roll_with_settlement(date, self, modifier)
        } else {
            roll_without_settlement(date, self, modifier)
        }
    }

    /// Adjust a date by a number of business days, under lag rules.
    ///
    /// *Note*: if the number of business days is **zero** a non-business day will be rolled
    /// **forwards**.
    ///
    /// *Note*: if the given `date` is a non-business date adding or subtracting 1 business
    /// day is equivalent to the rolling forwards or backwards, respectively.
    fn lag(&self, date: &NaiveDateTime, days: i8, settlement: bool) -> NaiveDateTime {
        if self.is_bus_day(date) {
            return self.add_bus_days(date, days, settlement).unwrap();
        }
        match days.cmp(&0_i8) {
            Ordering::Equal => self.roll_forward_bus_day(date),
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
    ///
    /// *Note*: When adding a positive number of days the only sensible modifiers are
    /// `Modifier::F` or `Modifier::Act` and when subtracting business days one should
    /// use `Modifier::P` or `Modifier::Act`.
    fn add_days(
        &self,
        date: &NaiveDateTime,
        days: i8,
        modifier: &Modifier,
        settlement: bool,
    ) -> NaiveDateTime
    where
        Self: Sized,
    {
        let new_date = if days < 0 {
            *date - Days::new(u64::try_from(-days).unwrap())
        } else {
            *date + Days::new(u64::try_from(days).unwrap())
        };
        self.roll(&new_date, modifier, settlement)
    }

    /// Add a given number of business days to a `date` with the result adjusted to a business day that may or may
    /// not allow `settlement`.
    ///
    /// *Note*: When adding a positive number of business days the only sensible modifier is
    /// `Modifier::F` and when subtracting business days it is `Modifier::P`.
    fn add_bus_days(
        &self,
        date: &NaiveDateTime,
        days: i8,
        settlement: bool,
    ) -> Result<NaiveDateTime, PyErr> {
        if self.is_non_bus_day(date) {
            return Err(PyValueError::new_err(
                "Cannot add business days to an input `date` that is not a business day.",
            ));
        }
        let mut new_date = *date;
        let mut counter: i8 = 0;
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

    /// Add a given number of months to a `date`, factoring a `roll` day, with the result adjusted
    /// to a business day that may or may not allow `settlement`.
    fn add_months(
        &self,
        date: &NaiveDateTime,
        months: i32,
        modifier: &Modifier,
        roll: &RollDay,
        settlement: bool,
    ) -> NaiveDateTime
    where
        Self: Sized,
    {
        // refactor roll day
        let roll_ = match roll {
            RollDay::Unspecified {} => RollDay::Int { day: date.day() },
            _ => *roll,
        };

        // convert months to a set of years and remainder months
        let mut yr_roll = (months.abs() / 12) * months.signum();
        let rem_months = months - yr_roll * 12;

        // determine the new month
        let mut new_month = i32::try_from(date.month()).unwrap() + rem_months;
        if new_month <= 0 {
            yr_roll -= 1;
            new_month = new_month.rem_euclid(12);
        } else if new_month >= 13 {
            yr_roll += 1;
            new_month = new_month.rem_euclid(12);
        }
        if new_month == 0 {
            new_month = 12;
        }

        // perform the date roll
        let new_date =
            get_roll(date.year() + yr_roll, new_month.try_into().unwrap(), &roll_).unwrap();
        self.roll(&new_date, modifier, settlement)
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
        self.settlement_calendars
            .as_ref()
            .map_or(true, |v| !v.iter().any(|cal| cal.is_non_bus_day(date)))
    }
}

impl<T> PartialEq<T> for UnionCal
where
    T: DateRoll,
{
    fn eq(&self, other: &T) -> bool {
        let cd1 = self
            .cal_date_range(&ndt(1970, 1, 1), &ndt(2200, 12, 31))
            .unwrap();
        let cd2 = other
            .cal_date_range(&ndt(1970, 1, 1), &ndt(2200, 12, 31))
            .unwrap();
        cd1.iter().zip(cd2.iter()).all(|(x, y)| {
            self.is_bus_day(x) == other.is_bus_day(x)
                && self.is_settlement(x) == other.is_settlement(y)
        })
    }
}

impl PartialEq<UnionCal> for Cal {
    fn eq(&self, other: &UnionCal) -> bool {
        let cd1 = self
            .cal_date_range(&ndt(1970, 1, 1), &ndt(2200, 12, 31))
            .unwrap();
        let cd2 = other
            .cal_date_range(&ndt(1970, 1, 1), &ndt(2200, 12, 31))
            .unwrap();
        cd1.iter().zip(cd2.iter()).all(|(x, y)| {
            self.is_bus_day(x) == other.is_bus_day(x)
                && self.is_settlement(x) == other.is_settlement(y)
        })
    }
}

/// A rule to adjust a non-business day to a business day.
#[pyclass(module = "rateslib.rs")]
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

/// A roll day.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone)]
pub enum RollDay {
    /// Inherit the day of the input date as the roll.
    Unspecified {},
    /// A day of the month in [1, 31].
    Int { day: u32 },
    /// The last day of the month (semantically equivalent to 31).
    EoM {},
    /// The first day of the month (semantically equivalent to 1).
    SoM {},
    /// The third Wednesday of the month.
    IMM {},
}

fn roll_with_settlement(
    date: &NaiveDateTime,
    cal: &dyn DateRoll,
    modifier: &Modifier,
) -> NaiveDateTime {
    match modifier {
        Modifier::Act => *date,
        Modifier::F => cal.roll_forward_settled_bus_day(date),
        Modifier::P => cal.roll_backward_settled_bus_day(date),
        Modifier::ModF => cal.roll_forward_mod_settled_bus_day(date),
        Modifier::ModP => cal.roll_backward_mod_settled_bus_day(date),
    }
}

fn roll_without_settlement(
    date: &NaiveDateTime,
    cal: &dyn DateRoll,
    modifier: &Modifier,
) -> NaiveDateTime {
    match modifier {
        Modifier::Act => *date,
        Modifier::F => cal.roll_forward_bus_day(date),
        Modifier::P => cal.roll_backward_bus_day(date),
        Modifier::ModF => cal.roll_mod_forward_bus_day(date),
        Modifier::ModP => cal.roll_mod_backward_bus_day(date),
    }
}

/// Create a `NaiveDateTime` with default null time.
///
/// Panics if date values are invalid.
pub fn ndt(year: i32, month: u32, day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(year, month, day)
        .expect("`year`, `month` `day` are invalid.")
        .and_hms_opt(0, 0, 0)
        .unwrap()
}

/// Return a specific roll date given the `month`, `year` and `roll`.
pub fn get_roll(year: i32, month: u32, roll: &RollDay) -> Result<NaiveDateTime, PyErr> {
    match roll {
        RollDay::Int { day: val } => Ok(get_roll_by_day(year, month, *val)),
        RollDay::EoM {} => Ok(get_roll_by_day(year, month, 31)),
        RollDay::SoM {} => Ok(get_roll_by_day(year, month, 1)),
        RollDay::IMM {} => Ok(get_imm(year, month)),
        RollDay::Unspecified {} => Err(PyValueError::new_err("`roll` cannot be unspecified.")),
    }
}

/// Return a specific roll date given the `month`, `year` and `roll`.
fn get_roll_by_day(year: i32, month: u32, day: u32) -> NaiveDateTime {
    let d = NaiveDate::from_ymd_opt(year, month, day);
    match d {
        Some(date) => NaiveDateTime::new(date, NaiveTime::from_hms_opt(0, 0, 0).unwrap()),
        None => {
            if day > 28 {
                get_roll_by_day(year, month, day - 1)
            } else {
                panic!("Unexpected error in `get_roll_by_day`")
            }
        }
    }
}

/// Return an IMM date (third Wednesday) for given month and year.
pub fn get_imm(year: i32, month: u32) -> NaiveDateTime {
    match ndt(year, month, 1).weekday() {
        Weekday::Mon => ndt(year, month, 17),
        Weekday::Tue => ndt(year, month, 16),
        Weekday::Wed => ndt(year, month, 15),
        Weekday::Thu => ndt(year, month, 21),
        Weekday::Fri => ndt(year, month, 20),
        Weekday::Sat => ndt(year, month, 19),
        Weekday::Sun => ndt(year, month, 18),
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::named::get_calendar_by_name;

    fn fixture_hol_cal() -> Cal {
        let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
        Cal::new(hols, vec![5, 6])
    }

    #[test]
    fn test_is_holiday() {
        let cal = fixture_hol_cal();
        let hol =
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol =
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday =
            NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(cal.is_holiday(&hol)); // In hol list
        assert!(!cal.is_holiday(&no_hol)); // Not in hol list
        assert!(!cal.is_holiday(&saturday)); // Not in hol list
    }

    #[test]
    fn test_is_weekday() {
        let cal = fixture_hol_cal();
        let hol =
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let no_hol =
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let saturday =
            NaiveDateTime::parse_from_str("2024-01-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let sunday =
            NaiveDateTime::parse_from_str("2024-01-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(cal.is_weekday(&hol)); // Monday
        assert!(cal.is_weekday(&no_hol)); //Thursday
        assert!(!cal.is_weekday(&saturday)); // Saturday
        assert!(!cal.is_weekday(&sunday)); // Sunday
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
    fn test_roll_with_settlement() {
        let cal = fixture_hol_cal();
        let non_bus =
            NaiveDateTime::parse_from_str("2024-03-30 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();

        let res = roll_with_settlement(&non_bus, &cal, &Modifier::F);
        assert_eq!(
            res,
            NaiveDateTime::parse_from_str("2024-04-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let res = roll_with_settlement(&non_bus, &cal, &Modifier::P);
        assert_eq!(
            res,
            NaiveDateTime::parse_from_str("2024-03-29 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let res = roll_with_settlement(&non_bus, &cal, &Modifier::ModF);
        assert_eq!(
            res,
            NaiveDateTime::parse_from_str("2024-03-29 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let res = roll_with_settlement(&non_bus, &cal, &Modifier::Act);
        assert_eq!(
            res,
            NaiveDateTime::parse_from_str("2024-03-30 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let non_bus =
            NaiveDateTime::parse_from_str("2024-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let res = roll_with_settlement(&non_bus, &cal, &Modifier::ModP);
        assert_eq!(
            res,
            NaiveDateTime::parse_from_str("2024-12-02 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );
    }

    #[test]
    fn test_lag() {
        let cal = fixture_hol_cal();
        let result = cal.lag(&ndt(2015, 9, 7), 1, true);
        assert_eq!(result, ndt(2015, 9, 8));

        let result = cal.lag(&ndt(2025, 2, 15), -1, true);
        assert_eq!(result, ndt(2025, 2, 14));

        let result = cal.lag(&ndt(2015, 9, 7), 0, true);
        assert_eq!(result, ndt(2015, 9, 8))
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

        let sat =
            NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = ucal.roll_forward_bus_day(&sat);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );
    }

    #[test]
    fn test_union_cal_with_settle() {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-09 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let scal = Cal::new(hols, vec![5, 6]);
        let cal = Cal::new(vec![], vec![5, 6]);
        let ucal = UnionCal::new(vec![cal], vec![scal].into());

        let mon =
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = ucal.roll_forward_bus_day(&mon);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );
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
        let next = cal.add_days(&tue, 2, &Modifier::F, false);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        // with settlement constraint 11th is invalid. Pushed to 14th over weekend.-
        let tue =
            NaiveDateTime::parse_from_str("2015-09-08 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let next = cal.add_days(&tue, 2, &Modifier::F, true);
        assert_eq!(
            next,
            NaiveDateTime::parse_from_str("2015-09-14 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        // without settlement constraint 11th is a valid previous roll date
        let tue =
            NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_days(&tue, -2, &Modifier::P, false);
        assert_eq!(
            prev,
            NaiveDateTime::parse_from_str("2015-09-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        // with settlement constraint 11th is invalid. Pushed to 9th over holiday.
        let tue =
            NaiveDateTime::parse_from_str("2015-09-15 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let prev = cal.add_days(&tue, -2, &Modifier::P, true);
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
    fn test_docstring() {
        let ldn = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]); // UK Monday 1st May Bank Holiday
        let tky = Cal::new(
            vec![ndt(2017, 5, 3), ndt(2017, 5, 4), ndt(2017, 5, 5)],
            vec![5, 6],
        );

        let date = ndt(2017, 4, 28); // Friday 28th April 2017
        let spot = ldn.add_bus_days(&date, 2, true).unwrap();
        assert_eq!(spot, ndt(2017, 5, 3));

        let ldn_tky = UnionCal::new(vec![ldn, tky], None);
        let spot = ldn_tky.add_bus_days(&date, 2, true).unwrap();
        assert_eq!(spot, ndt(2017, 5, 8));

        let tgt = Cal::new(vec![], vec![5, 6]);
        let nyc = Cal::new(vec![ndt(2023, 6, 19)], vec![5, 6]); // Juneteenth Holiday
        let tgt_nyc = UnionCal::new(vec![tgt], vec![nyc].into());

        let date = ndt(2023, 6, 16);
        let spot = tgt_nyc.add_bus_days(&date, 2, true).unwrap();
        assert_eq!(spot, ndt(2023, 6, 20));

        let date = ndt(2023, 6, 15);
        let spot = tgt_nyc.add_bus_days(&date, 2, true).unwrap();
        assert_eq!(spot, ndt(2023, 6, 20));

        let spot = tgt_nyc.add_bus_days(&date, 2, false).unwrap();
        assert_eq!(spot, ndt(2023, 6, 19));
    }

    #[test]
    fn test_add_37_months() {
        let cal = get_calendar_by_name("all").unwrap();

        let dates = vec![
            (ndt(2000, 1, 1), ndt(2003, 2, 1)),
            (ndt(2000, 2, 1), ndt(2003, 3, 1)),
            (ndt(2000, 3, 1), ndt(2003, 4, 1)),
            (ndt(2000, 4, 1), ndt(2003, 5, 1)),
            (ndt(2000, 5, 1), ndt(2003, 6, 1)),
            (ndt(2000, 6, 1), ndt(2003, 7, 1)),
            (ndt(2000, 7, 1), ndt(2003, 8, 1)),
            (ndt(2000, 8, 1), ndt(2003, 9, 1)),
            (ndt(2000, 9, 1), ndt(2003, 10, 1)),
            (ndt(2000, 10, 1), ndt(2003, 11, 1)),
            (ndt(2000, 11, 1), ndt(2003, 12, 1)),
            (ndt(2000, 12, 1), ndt(2004, 1, 1)),
        ];
        for i in 0..12 {
            assert_eq!(
                cal.add_months(
                    &dates[i].0,
                    37,
                    &Modifier::Act,
                    &RollDay::Unspecified {},
                    true
                ),
                dates[i].1
            )
        }
    }

    #[test]
    fn test_sub_37_months() {
        let cal = get_calendar_by_name("all").unwrap();

        let dates = vec![
            (ndt(2000, 1, 1), ndt(1996, 12, 1)),
            (ndt(2000, 2, 1), ndt(1997, 1, 1)),
            (ndt(2000, 3, 1), ndt(1997, 2, 1)),
            (ndt(2000, 4, 1), ndt(1997, 3, 1)),
            (ndt(2000, 5, 1), ndt(1997, 4, 1)),
            (ndt(2000, 6, 1), ndt(1997, 5, 1)),
            (ndt(2000, 7, 1), ndt(1997, 6, 1)),
            (ndt(2000, 8, 1), ndt(1997, 7, 1)),
            (ndt(2000, 9, 1), ndt(1997, 8, 1)),
            (ndt(2000, 10, 1), ndt(1997, 9, 1)),
            (ndt(2000, 11, 1), ndt(1997, 10, 1)),
            (ndt(2000, 12, 1), ndt(1997, 11, 1)),
        ];
        for i in 0..12 {
            assert_eq!(
                cal.add_months(
                    &dates[i].0,
                    -37,
                    &Modifier::Act,
                    &RollDay::Unspecified {},
                    true
                ),
                dates[i].1
            )
        }
    }

    #[test]
    fn test_add_months_roll() {
        let cal = get_calendar_by_name("all").unwrap();
        let roll = vec![
            (RollDay::Unspecified {}, ndt(1996, 12, 7)),
            (RollDay::Int { day: 21 }, ndt(1996, 12, 21)),
            (RollDay::EoM {}, ndt(1996, 12, 31)),
            (RollDay::SoM {}, ndt(1996, 12, 1)),
            (RollDay::IMM {}, ndt(1996, 12, 18)),
        ];
        for i in 0..5 {
            assert_eq!(
                cal.add_months(&ndt(1998, 3, 7), -15, &Modifier::Act, &roll[i].0, true),
                roll[i].1
            );
        }
    }

    #[test]
    fn test_add_months_modifier() {
        let cal = get_calendar_by_name("bus").unwrap();
        let modi = vec![
            (Modifier::Act, ndt(2023, 9, 30)),  // Saturday
            (Modifier::F, ndt(2023, 10, 2)),    // Monday
            (Modifier::ModF, ndt(2023, 9, 29)), // Friday
            (Modifier::P, ndt(2023, 9, 29)),    // Friday
            (Modifier::ModP, ndt(2023, 9, 29)), // Friday
        ];
        for i in 0..4 {
            assert_eq!(
                cal.add_months(
                    &ndt(2023, 8, 31),
                    1,
                    &modi[i].0,
                    &RollDay::Unspecified {},
                    true
                ),
                modi[i].1
            );
        }
    }

    #[test]
    fn test_add_months_modifier_p() {
        let cal = get_calendar_by_name("bus").unwrap();
        let modi = vec![
            (Modifier::Act, ndt(2023, 7, 1)),  // Saturday
            (Modifier::F, ndt(2023, 7, 3)),    // Monday
            (Modifier::ModF, ndt(2023, 7, 3)), // Monday
            (Modifier::P, ndt(2023, 6, 30)),   // Friday
            (Modifier::ModP, ndt(2023, 7, 3)), // Monday
        ];
        for i in 0..4 {
            assert_eq!(
                cal.add_months(
                    &ndt(2023, 8, 1),
                    -1,
                    &modi[i].0,
                    &RollDay::Unspecified {},
                    true
                ),
                modi[i].1
            );
        }
    }

    #[test]
    fn test_cal_json() {
        let hols = vec![ndt(2015, 9, 8), ndt(2015, 9, 10)];
        let hcal = Cal::new(hols, vec![5, 6]);
        let js = hcal.to_json().unwrap();
        let hcal2 = Cal::from_json(&js).unwrap();
        assert_eq!(hcal, hcal2);
    }

    #[test]
    fn test_union_cal_json() {
        let hols = vec![ndt(2015, 9, 8), ndt(2015, 9, 10)];
        let settle = vec![ndt(2015, 9, 11)];
        let hcal = Cal::new(hols, vec![5, 6]);
        let scal = Cal::new(settle, vec![5, 6]);
        let ucal = UnionCal::new(vec![hcal], vec![scal].into());
        let js = ucal.to_json().unwrap();
        let ucal2 = UnionCal::from_json(&js).unwrap();
        assert_eq!(ucal, ucal2);
    }

    #[test]
    fn test_cross_equality() {
        let cal = fixture_hol_cal();
        let ucal = UnionCal::new(vec![cal.clone()], None);
        assert_eq!(cal, ucal);
        assert_eq!(ucal, cal);

        let ucals = UnionCal::new(vec![cal.clone()], vec![cal.clone()].into());
        assert_ne!(cal, ucals);
        assert_ne!(ucals, cal);

        let cal2 = fixture_hol_cal2();
        assert_ne!(cal2, ucal);
        assert_ne!(ucal, cal2);
    }
}
