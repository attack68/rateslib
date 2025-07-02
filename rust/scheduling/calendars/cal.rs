use chrono::prelude::*;
use chrono::Weekday;
use indexmap::set::IndexSet;
use pyo3::{pyclass};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::scheduling::{DateRoll, UnionCal, NamedCal, ndt, CalendarAdjustment};

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

impl CalendarAdjustment for Cal {}

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

impl PartialEq<NamedCal> for Cal {
    fn eq(&self, other: &NamedCal) -> bool {
        other.union_cal.eq(self)
    }
}


// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{Adjuster};

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
    fn test_calendar_adjust() {
        let cal = fixture_hol_cal();
        let result = cal.adjust(&ndt(2015, 9, 5), &Adjuster::Following {});
        assert_eq!(ndt(2015, 9, 8), result);
    }

    #[test]
    fn test_calendar_adjusts() {
        let cal = fixture_hol_cal();
        let result = cal.adjusts(
            &vec![ndt(2015, 9, 5), ndt(2015, 9, 6)],
            &Adjuster::Following {},
        );
        assert_eq!(vec![ndt(2015, 9, 8), ndt(2015, 9, 8)], result);
    }
}
