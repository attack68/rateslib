use chrono::prelude::*;
use chrono::Weekday;
use indexmap::set::IndexSet;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::scheduling::calendars::named::{get_holidays_by_name, get_weekmask_by_name};
use crate::scheduling::{ndt, CalendarAdjustment, DateRoll, NamedCal, UnionCal};

/// A basic business day calendar containing holidays.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cal {
    /// A vector of specific dates that are defined as **non-business** days.
    pub holidays: IndexSet<NaiveDateTime>,
    /// A vector of days in the week that are defined as **non-business** days. E.g. `[5, 6]` for Saturday and Sunday.
    pub week_mask: HashSet<Weekday>,
    // pub(crate) meta: Vec<String>,
}

impl Cal {
    /// Create a [`Cal`].
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Cal, ndt, DateRoll};
    /// let cal = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]); // With May Bank Holiday
    /// let spot = cal.add_bus_days(&ndt(2017, 4, 28), 2, true);
    /// # let spot = spot.unwrap();
    /// assert_eq!(ndt(2017, 5, 3), spot);
    /// ```
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

    /// Return a [`Cal`] specified by a pre-defined named identifier.
    ///
    /// For available 3-digit names see `named` module documentation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::scheduling::Cal;
    /// let ldn_cal = Cal::try_from_name("ldn").unwrap();
    /// ```
    pub fn try_from_name(name: &str) -> Result<Cal, PyErr> {
        Ok(Cal::new(
            get_holidays_by_name(name)?,
            get_weekmask_by_name(name)?,
            // get_rules_by_name(name)?
        ))
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
    use crate::scheduling::Adjuster;

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

    // Pre defined named calendars

    #[test]
    fn test_get_cal() {
        let result = Cal::try_from_name("bus").unwrap();
        let expected = Cal::new(vec![], vec![5, 6]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_all() {
        let cal = Cal::try_from_name("all").unwrap();
        assert!(cal.is_bus_day(
            &NaiveDateTime::parse_from_str("2024-11-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_nyc() {
        let cal = Cal::try_from_name("nyc").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-11-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_tgt() {
        let cal = Cal::try_from_name("tgt").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-05-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_ldn() {
        let cal = Cal::try_from_name("ldn").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-08-26 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_stk() {
        let cal = Cal::try_from_name("stk").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-06-06 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_osl() {
        let cal = Cal::try_from_name("osl").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-05-17 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_zur() {
        let cal = Cal::try_from_name("zur").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-08-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_tro() {
        let cal = Cal::try_from_name("tro").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-09-30 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_tyo() {
        let cal = Cal::try_from_name("tyo").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-1-3 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_fed() {
        let cal = Cal::try_from_name("fed").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2024-11-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_get_calendar_error() {
        match Cal::try_from_name("badname") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    #[test]
    fn test_syd() {
        let cal = Cal::try_from_name("syd").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2022-09-22 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_wlg() {
        let cal = Cal::try_from_name("wlg").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2034-07-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }

    #[test]
    fn test_mum() {
        let cal = Cal::try_from_name("mum").unwrap();
        assert!(cal.is_holiday(
            &NaiveDateTime::parse_from_str("2025-01-26 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        ));
    }
}
