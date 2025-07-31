use chrono::prelude::*;
use pyo3::{FromPyObject, IntoPyObject};
use serde::{Deserialize, Serialize};
use std::convert::From;

use crate::scheduling::{Cal, CalendarAdjustment, DateRoll, NamedCal, UnionCal};

/// Create a `NaiveDateTime` with default null time.
///
/// Panics if date values are invalid.
pub fn ndt(year: i32, month: u32, day: u32) -> NaiveDateTime {
    NaiveDate::from_ymd_opt(year, month, day)
        .expect("`year`, `month` `day` are invalid.")
        .and_hms_opt(0, 0, 0)
        .unwrap()
}

/// Container for calendar types.
#[derive(Debug, Clone, PartialEq, FromPyObject, Serialize, Deserialize, IntoPyObject)]
pub enum Calendar {
    Cal(Cal),
    UnionCal(UnionCal),
    NamedCal(NamedCal),
}

impl From<Cal> for Calendar {
    fn from(item: Cal) -> Self {
        Calendar::Cal(item)
    }
}

impl From<UnionCal> for Calendar {
    fn from(item: UnionCal) -> Self {
        Calendar::UnionCal(item)
    }
}

impl From<NamedCal> for Calendar {
    fn from(item: NamedCal) -> Self {
        Calendar::NamedCal(item)
    }
}

impl DateRoll for Calendar {
    fn is_weekday(&self, date: &NaiveDateTime) -> bool {
        match self {
            Calendar::Cal(c) => c.is_weekday(date),
            Calendar::UnionCal(c) => c.is_weekday(date),
            Calendar::NamedCal(c) => c.is_weekday(date),
        }
    }

    fn is_holiday(&self, date: &NaiveDateTime) -> bool {
        match self {
            Calendar::Cal(c) => c.is_holiday(date),
            Calendar::UnionCal(c) => c.is_holiday(date),
            Calendar::NamedCal(c) => c.is_holiday(date),
        }
    }

    fn is_settlement(&self, date: &NaiveDateTime) -> bool {
        match self {
            Calendar::Cal(c) => c.is_settlement(date),
            Calendar::UnionCal(c) => c.is_settlement(date),
            Calendar::NamedCal(c) => c.is_settlement(date),
        }
    }
}

impl CalendarAdjustment for Calendar {}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

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
}
