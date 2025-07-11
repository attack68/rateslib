use chrono::prelude::*;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};

use crate::scheduling::{ndt, Cal, CalendarAdjustment, DateRoll};

/// A business day calendar which is the potential union of multiple calendars.
///
/// This calendar type allows the additional feature of also ensuring dates follow settlement
/// compliance as measured according to other calendars.
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
    /// A vector of [Cal] used to determine **business** days.
    pub calendars: Vec<Cal>,
    /// A vector of [Cal] used to determine **settleable** days.
    pub settlement_calendars: Option<Vec<Cal>>,
}

impl UnionCal {
    /// Create a new [UnionCal].
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Cal, UnionCal, ndt};
    /// let stk = Cal::new(vec![ndt(2025, 6, 20)], vec![5,6]);
    /// let fed = Cal::new(vec![ndt(2025, 6, 19)], vec![5,6]);
    /// let stk_pipe_fed = UnionCal::new(vec![stk], Some(vec![fed]));
    /// ```
    pub fn new(calendars: Vec<Cal>, settlement_calendars: Option<Vec<Cal>>) -> Self {
        UnionCal {
            calendars,
            settlement_calendars,
        }
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

impl CalendarAdjustment for UnionCal {}

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

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_hol_cal() -> Cal {
        let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
        Cal::new(hols, vec![5, 6])
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
