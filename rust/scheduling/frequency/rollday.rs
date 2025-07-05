use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::PartialEq;

use crate::scheduling::ndt;

/// A roll day.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Debug, Copy, Clone, PartialEq)]
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

impl RollDay {
    /// Validate whether an unadjusted date is an allowed value under the [RollDay] definition.
    pub fn try_udate(&self, date: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let msg = "`date` does not align with given `roll`.".to_string();
        match self {
            RollDay::Unspecified {} => Ok(*date), // any date satisfies unspecified RollDay
            RollDay::Int { day: 31 } | RollDay::EoM {} => {
                if is_eom(date) {
                    Ok(*date)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::Int { day: 30 } => {
                if (is_eom(date) && date.day() < 30) || date.day() == 30 {
                    Ok(*date)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::Int { day: 29 } => {
                if (is_eom(date) && date.day() < 29) || date.day() == 29 {
                    Ok(*date)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::IMM {} => {
                if is_imm(date) {
                    Ok(*date)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::Int { day: value } => {
                if date.day() == *value {
                    Ok(*date)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::SoM {} => {
                if date.day() == 1 {
                    Ok(*date)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
        }
    }

    /// Add a given number of months to an unadjusted date under the [RollDay] definition.
    pub fn uadd(&self, udate: &NaiveDateTime, months: i32) -> NaiveDateTime {
        // refactor roll day
        let roll_ = match self {
            Self::Unspecified {} => Self::Int { day: udate.day() },
            _ => *self,
        };

        // convert months to a set of years and remainder months
        let mut yr_roll = (months.abs() / 12) * months.signum();
        let rem_months = months - yr_roll * 12;

        // determine the new month
        let mut new_month = i32::try_from(udate.month()).unwrap() + rem_months;
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
        get_roll(
            udate.year() + yr_roll,
            new_month.try_into().unwrap(),
            &roll_,
        )
        .unwrap()
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

/// Return an end of month date for given month and year.
pub fn get_eom(year: i32, month: u32) -> NaiveDateTime {
    let mut day = 31;
    let mut date = NaiveDate::from_ymd_opt(year, month, day);
    while date == None {
        day = day - 1;
        date = NaiveDate::from_ymd_opt(year, month, day);
    }
    date.unwrap().and_hms_opt(0, 0, 0).unwrap()
}

/// Test whether a given date is EoM.
pub fn is_eom(date: &NaiveDateTime) -> bool {
    let eom = get_eom(date.year(), date.month());
    *date == eom
}

/// Test whether a given date is an IMM (third Wednesday).
pub fn is_imm(date: &NaiveDateTime) -> bool {
    let imm = get_imm(date.year(), date.month());
    *date == imm
}

/// Test whether a given year is a leap year.
pub fn is_leap_year(year: i32) -> bool {
    NaiveDate::from_ymd_opt(year, 2, 29).is_some()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rollday_equality() {
        let rd1 = RollDay::IMM {};
        let rd2 = RollDay::IMM {};
        assert_eq!(rd1, rd2);

        let rd1 = RollDay::IMM {};
        let rd2 = RollDay::EoM {};
        assert_ne!(rd1, rd2);

        let rd1 = RollDay::Int { day: 20 };
        let rd2 = RollDay::Int { day: 20 };
        assert_eq!(rd1, rd2);

        let rd1 = RollDay::Int { day: 21 };
        let rd2 = RollDay::Int { day: 9 };
        assert_ne!(rd1, rd2);
    }

    #[test]
    fn test_is_imm() {
        assert_eq!(true, is_imm(&ndt(2025, 3, 19)));
        assert_eq!(false, is_imm(&ndt(2025, 3, 18)));
    }

    #[test]
    fn test_get_eom() {
        assert_eq!(ndt(2022, 2, 28), get_eom(2022, 2));
        assert_eq!(ndt(2024, 2, 29), get_eom(2024, 2));
        assert_eq!(ndt(2022, 4, 30), get_eom(2022, 4));
        assert_eq!(ndt(2022, 3, 31), get_eom(2022, 3));
    }

    #[test]
    fn test_is_eom() {
        assert_eq!(true, is_eom(&ndt(2025, 3, 31)));
        assert_eq!(false, is_eom(&ndt(2025, 3, 30)));
    }

    #[test]
    fn test_is_leap() {
        assert_eq!(true, is_leap_year(2024));
        assert_eq!(false, is_leap_year(2022));
    }

    #[test]
    fn test_rollday_try_udate() {
        let options: Vec<(RollDay, NaiveDateTime)> = vec![
            (RollDay::Int { day: 15 }, ndt(2000, 3, 15)),
            (RollDay::Int { day: 31 }, ndt(2000, 3, 31)),
            (RollDay::Int { day: 31 }, ndt(2022, 2, 28)),
            (RollDay::EoM {}, ndt(2000, 3, 31)),
            (RollDay::EoM {}, ndt(2022, 2, 28)),
            (RollDay::Int { day: 30 }, ndt(2024, 2, 29)),
            (RollDay::Int { day: 30 }, ndt(2024, 2, 29)),
            (RollDay::EoM {}, ndt(2024, 2, 29)),
            (RollDay::EoM {}, ndt(2024, 2, 29)),
            (RollDay::EoM {}, ndt(2024, 2, 29)),
        ];
        for option in options {
            assert_eq!(false, option.0.try_udate(&option.1).is_err());
        }
    }
}
