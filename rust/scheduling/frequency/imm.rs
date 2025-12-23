#![allow(non_camel_case_types)]

use chrono::prelude::*;
use chrono::Months;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{Eq, PartialEq};

use crate::scheduling::ndt;

/// Specifier for IMM date definitions.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Imm {
    /// 3rd Wednesday of March, June, September and December.
    ///
    /// Commonly used by STIR futures in northern hemisphere.
    Wed3_HMUZ = 0,
    /// 3rd Wednesday of any calendar month.
    ///
    /// Commonly used by STIR futures in northern hemisphere.
    Wed3 = 1,
    /// 20th day of March, June, September and December.
    ///
    /// Commonly used by CDS.
    Day20_HMUZ = 2,
    /// 20th day of March and September.
    ///
    /// Commonly used by CDS.
    Day20_HU = 3,
    /// 20th day of June and December.
    ///
    /// Commonly used by CDS.
    Day20_MZ = 4,
    /// 20th day of any calendar month.
    Day20 = 5,
    /// 2nd Friday of March, June, September and December.
    ///
    /// Commonly used by ASX 90 day AUD bank bill futures.
    Fri2_HMUZ = 6,
    /// 2nd Friday of any calendar month.
    ///
    /// Commonly used by ASX 90 day AUD bank bill futures.
    Fri2 = 7,
    /// 1st Wednesday after the 9th of the month in March, June, September and December.
    ///
    /// Commonly used by ASX 90 day NZD bank bill futures.
    Wed1_Post9_HMUZ = 10,
    /// 1st Wednesday after the 9th of any calendar month.
    ///
    /// Commonly used by ASX 90 day NZD bank bill futures.
    Wed1_Post9 = 11,
    /// End of any calendar month
    Eom = 8,
    /// February Leap days
    Leap = 9,
}

impl Imm {
    /// Check whether a given date aligns with the IMM date definition.
    pub fn validate(&self, date: &NaiveDateTime) -> bool {
        let result = self.from_ym_opt(date.year(), date.month());
        match result {
            Ok(val) => *date == val,
            Err(_) => false,
        }
    }

    /// Get an IMM date with the appropriate definition from a given month and year.
    pub fn from_ym_opt(&self, year: i32, month: u32) -> Result<NaiveDateTime, PyErr> {
        match self {
            Imm::Wed3_HMUZ => {
                if month == 3 || month == 6 || month == 9 || month == 12 {
                    Imm::Wed3.from_ym_opt(year, month)
                } else {
                    Err(PyValueError::new_err("Must be month Mar, Jun, Sep or Dec."))
                }
            }
            Imm::Fri2_HMUZ => {
                if month == 3 || month == 6 || month == 9 || month == 12 {
                    Imm::Fri2.from_ym_opt(year, month)
                } else {
                    Err(PyValueError::new_err("Must be month Mar, Jun, Sep or Dec."))
                }
            }
            Imm::Wed1_Post9_HMUZ => {
                if month == 3 || month == 6 || month == 9 || month == 12 {
                    Imm::Wed1_Post9.from_ym_opt(year, month)
                } else {
                    Err(PyValueError::new_err("Must be month Mar, Jun, Sep or Dec."))
                }
            }
            Imm::Wed3 => {
                let w = ndt(year, month, 1).weekday() as u32;
                let r = if w <= 2 { 17 - w } else { 24 - w };
                Ok(ndt(year, month, r))
            }
            Imm::Fri2 => {
                let w = ndt(year, month, 1).weekday() as u32;
                let r = if w <= 4 { 12 - w } else { 19 - w };
                Ok(ndt(year, month, r))
            }
            Imm::Wed1_Post9 => {
                let w = ndt(year, month, 1).weekday() as u32;
                let r = if w <= 0 { 10 - w } else { 17 - w };
                Ok(ndt(year, month, r))
            }
            Imm::Day20_HMUZ => {
                if month == 3 || month == 6 || month == 9 || month == 12 {
                    Ok(ndt(year, month, 20))
                } else {
                    Err(PyValueError::new_err("Must be month Mar, Jun, Sep or Dec."))
                }
            }
            Imm::Day20_HU => {
                if month == 3 || month == 9 {
                    Ok(ndt(year, month, 20))
                } else {
                    Err(PyValueError::new_err("Must be month Mar, or Sep."))
                }
            }
            Imm::Day20_MZ => {
                if month == 6 || month == 12 {
                    Ok(ndt(year, month, 20))
                } else {
                    Err(PyValueError::new_err("Must be month Jun, or Dec."))
                }
            }
            Imm::Day20 => Ok(ndt(year, month, 20)),
            Imm::Eom => {
                let mut day = 31;
                let mut date = NaiveDate::from_ymd_opt(year, month, day);
                while date == None {
                    day = day - 1;
                    date = NaiveDate::from_ymd_opt(year, month, day);
                    if day == 0 {
                        return Err(PyValueError::new_err("`year` or `month` out of range."));
                    }
                }
                Ok(date.unwrap().and_hms_opt(0, 0, 0).unwrap())
            }
            Imm::Leap => {
                if month != 2 {
                    Err(PyValueError::new_err("Leap is only in `month`:2."))
                } else {
                    let d = NaiveDate::from_ymd_opt(year, 2, 29);
                    match d {
                        None => Err(PyValueError::new_err("No Leap in given `year`.")),
                        Some(val) => Ok(val.and_hms_opt(0, 0, 0).unwrap()),
                    }
                }
            }
        }
    }

    /// Get the IMM date that follows the given ``date``.
    pub fn next(&self, date: &NaiveDateTime) -> NaiveDateTime {
        let mut sample = *date;
        let mut result = self.from_ym_opt(date.year(), date.month());
        loop {
            match result {
                Ok(v) if v > *date => return v,
                _ => {
                    sample = sample + Months::new(1);
                    result = self.from_ym_opt(sample.year(), sample.month());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imm_date_determination() {
        let options: Vec<(Imm, NaiveDateTime, bool)> = vec![
            (Imm::Wed3_HMUZ, ndt(2000, 3, 15), true),
            (Imm::Wed3_HMUZ, ndt(2000, 3, 22), false),
            (Imm::Wed3_HMUZ, ndt(2000, 3, 8), false),
            (Imm::Wed3_HMUZ, ndt(2000, 2, 21), false),
            (Imm::Wed3, ndt(2024, 2, 21), true),
            (Imm::Wed3, ndt(2000, 3, 15), true),
            (Imm::Wed3, ndt(2025, 3, 19), true),
            (Imm::Wed3, ndt(2025, 3, 18), false),
            (Imm::Day20_HMUZ, ndt(2000, 2, 21), false),
            (Imm::Day20_HMUZ, ndt(2000, 2, 20), false),
            (Imm::Day20_HMUZ, ndt(2000, 3, 20), true),
            (Imm::Day20_HU, ndt(2000, 3, 20), true),
            (Imm::Day20_HU, ndt(2000, 6, 20), false),
            (Imm::Day20_MZ, ndt(2000, 3, 20), false),
            (Imm::Day20_MZ, ndt(2000, 6, 20), true),
            (Imm::Fri2, ndt(2024, 2, 9), true),
            (Imm::Fri2, ndt(2024, 12, 13), true),
            (Imm::Wed1_Post9, ndt(2025, 9, 10), true),
            (Imm::Wed1_Post9, ndt(2026, 9, 16), true),
        ];
        for option in options {
            assert_eq!(option.2, option.0.validate(&option.1));
        }
    }

    #[test]
    fn next_check() {
        let options: Vec<(Imm, NaiveDateTime, NaiveDateTime)> = vec![
            (Imm::Wed3_HMUZ, ndt(2024, 3, 20), ndt(2024, 6, 19)),
            (Imm::Wed3_HMUZ, ndt(2024, 3, 19), ndt(2024, 3, 20)),
            (Imm::Wed3, ndt(2024, 3, 21), ndt(2024, 4, 17)),
            (Imm::Day20_HU, ndt(2024, 3, 21), ndt(2024, 9, 20)),
            (Imm::Leap, ndt(2022, 1, 1), ndt(2024, 2, 29)),
        ];
        for option in options {
            assert_eq!(option.2, option.0.next(&option.1));
        }
    }

    #[test]
    fn test_is_eom() {
        assert_eq!(true, Imm::Eom.validate(&ndt(2025, 3, 31)));
        assert_eq!(false, Imm::Eom.validate(&ndt(2025, 3, 30)));
    }

    #[test]
    fn test_get_from() {
        assert_eq!(ndt(2022, 2, 28), Imm::Eom.from_ym_opt(2022, 2).unwrap());
        assert_eq!(ndt(2024, 2, 29), Imm::Eom.from_ym_opt(2024, 2).unwrap());
        assert_eq!(ndt(2022, 4, 30), Imm::Eom.from_ym_opt(2022, 4).unwrap());
        assert_eq!(ndt(2022, 3, 31), Imm::Eom.from_ym_opt(2022, 3).unwrap());
        assert_eq!(ndt(2024, 2, 29), Imm::Leap.from_ym_opt(2024, 2).unwrap());
        assert!(Imm::Leap.from_ym_opt(2022, 2).is_err());
    }
}
