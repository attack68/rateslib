use chrono::prelude::*;
use indexmap::IndexSet;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{Eq, PartialEq};

use crate::scheduling::{Adjuster, Adjustment, Calendar, Imm};

/// A roll-day used with a [`Frequency::Months`](crate::scheduling::Frequency) variant.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum RollDay {
    /// A day of the month in [1, 31].
    Day(u32),
    /// The third Wednesday of any month (equivalent to [Imm::Wed3](crate::scheduling::Imm))
    IMM(),
}

impl RollDay {
    /// Get all possible [`RollDay`] variants implied from one or more unadjusted dates.
    ///
    /// # Notes
    /// Each date is analysed in turn. The order of [`RollDay`] construction for each date is:
    ///
    /// - Get the integer roll-day of the date.
    /// - Get additional end-of-month related integer roll-days for short calendar months if necessary.
    /// - Get non-numeric roll-days if date aligns with those, ordered by the underlying enum order.
    ///
    /// When multiple dates are checked the results for a subsequent date is added to the prior
    /// results under the [`IndexSet.intersection`] ordering rules.
    ///
    /// Any date will always return at least one [RollDay] and the first one will always be
    /// equivalent to an integer variant whose day equals the calendar day of the first date.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{RollDay, ndt};
    /// let result = RollDay::vec_from(&vec![ndt(2024, 2, 29), ndt(2024, 3, 20), ndt(2024, 3, 31)]);
    /// assert_eq!(result, vec![
    ///     RollDay::Day(29),
    ///     RollDay::Day(30),
    ///     RollDay::Day(31),
    ///     RollDay::Day(20),
    ///     RollDay::IMM(),
    /// ]);
    /// ```
    pub fn vec_from(udates: &Vec<NaiveDateTime>) -> Vec<Self> {
        let mut set: IndexSet<RollDay> = IndexSet::new();

        for udate in udates {
            // numeric first
            let mut v: Vec<Self> = vec![RollDay::Day(udate.day())];
            // EoM check
            if Imm::Eom.validate(udate) {
                let mut day = udate.day() + 1;
                while day < 32 {
                    v.push(RollDay::Day(day));
                    day = day + 1;
                }
            }
            // IMM check
            if Imm::Wed3.validate(udate) {
                v.push(RollDay::IMM())
            }
            // Intersect existing results
            set.append(&mut IndexSet::<RollDay>::from_iter(v));
        }
        set.into_iter().collect()
    }

    /// Validate whether an unadjusted date is an allowed value under the [`RollDay`] definition.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{RollDay, ndt};
    /// let date = RollDay::Day(31).try_udate(&ndt(2024, 2, 29));
    /// assert!(date.is_ok());
    ///
    /// let date = RollDay::IMM().try_udate(&ndt(2024, 1, 1));
    /// assert!(date.is_err());
    /// ```
    pub fn try_udate(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let msg = "`udate` does not align with given `RollDay`.".to_string();
        match self {
            RollDay::Day(31) => {
                if Imm::Eom.validate(udate) {
                    Ok(*udate)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::Day(30) => {
                if (Imm::Eom.validate(udate) && udate.day() < 30) || udate.day() == 30 {
                    Ok(*udate)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::Day(29) => {
                if (Imm::Eom.validate(udate) && udate.day() < 29) || udate.day() == 29 {
                    Ok(*udate)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::IMM() => {
                if Imm::Wed3.validate(udate) {
                    Ok(*udate)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
            RollDay::Day(value) => {
                if udate.day() == *value {
                    Ok(*udate)
                } else {
                    Err(PyValueError::new_err(msg))
                }
            }
        }
    }

    /// Add a given number of months to an unadjusted date under the [RollDay] definition.
    ///
    /// # Notes
    /// This method will also check the given `udate` using [RollDay::try_udate].
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{RollDay, ndt};
    /// let date = RollDay::IMM().try_uadd(&ndt(2024, 3, 20), 3);
    /// assert_eq!(ndt(2024, 6, 19), date.unwrap());
    ///
    /// let date = RollDay::Day(31).try_uadd(&ndt(2024, 3, 15), 3);
    /// assert!(date.is_err());
    /// ```
    pub fn try_uadd(&self, udate: &NaiveDateTime, months: i32) -> Result<NaiveDateTime, PyErr> {
        let _ = self.try_udate(udate)?;
        Ok(self.uadd(udate, months))
    }

    /// Add a given number of months to an unadjusted date under the [RollDay] definition.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{RollDay, ndt};
    /// let date = RollDay::Day(31).uadd(&ndt(2024, 3, 15), 3);
    /// assert_eq!(date, ndt(2024, 6, 30));
    /// ```
    pub fn uadd(&self, udate: &NaiveDateTime, months: i32) -> NaiveDateTime {
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
        self.try_from_ym(udate.year() + yr_roll, new_month.try_into().unwrap())
            .unwrap()
    }

    /// Return a specific date given the `month`, `year` that aligns with the [RollDay].
    ///     
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{RollDay, ndt};
    /// let date = RollDay::Day(31).try_from_ym(2024, 2);
    /// # let date = date.unwrap();
    /// assert_eq!(date, ndt(2024, 2, 29));
    /// ```
    pub fn try_from_ym(&self, year: i32, month: u32) -> Result<NaiveDateTime, PyErr> {
        match self {
            RollDay::Day(value) => Ok(get_roll_by_day(year, month, *value)),
            RollDay::IMM {} => Imm::Wed3.from_ym_opt(year, month),
        }
    }
}

/// Get unadjusted date alternatives for an associated adjusted date.
///
/// Note this only handles simple date rolling operations, and does not generalise to any
/// possible adjuster.
pub(crate) fn get_unadjusteds(
    date: &NaiveDateTime,
    adjuster: &Adjuster,
    calendar: &Calendar,
) -> Vec<NaiveDateTime> {
    let mut udates: Vec<NaiveDateTime> = vec![];

    // always return at least `date`
    udates.push(*date);

    // get the vector of reversals and filter out date
    let reversals: Vec<NaiveDateTime> = adjuster
        .reverse(date, calendar)
        .into_iter()
        .filter(|v| v != date)
        .collect();
    udates.extend(reversals);
    udates
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
    use crate::scheduling::{ndt, Cal};

    fn fixture_bus_cal() -> Calendar {
        Cal::try_from_name("bus").unwrap().into()
    }

    #[test]
    fn test_rollday_equality() {
        let rd1 = RollDay::IMM();
        let rd2 = RollDay::IMM();
        assert_eq!(rd1, rd2);

        let rd1 = RollDay::IMM();
        let rd2 = RollDay::Day(21);
        assert_ne!(rd1, rd2);

        let rd1 = RollDay::Day(20);
        let rd2 = RollDay::Day(20);
        assert_eq!(rd1, rd2);

        let rd1 = RollDay::Day(21);
        let rd2 = RollDay::Day(9);
        assert_ne!(rd1, rd2);
    }

    #[test]
    fn test_rollday_try_udate() {
        let options: Vec<(RollDay, NaiveDateTime)> = vec![
            (RollDay::Day(15), ndt(2000, 3, 15)),
            (RollDay::Day(31), ndt(2000, 3, 31)),
            (RollDay::Day(31), ndt(2022, 2, 28)),
            (RollDay::Day(30), ndt(2024, 2, 29)),
            (RollDay::Day(31), ndt(2024, 2, 29)),
        ];
        for option in options {
            assert_eq!(false, option.0.try_udate(&option.1).is_err());
        }
    }

    #[test]
    fn test_get_unadjusteds() {
        let options: Vec<(NaiveDateTime, Vec<NaiveDateTime>)> = vec![
            (ndt(2000, 2, 29), vec![ndt(2000, 2, 29)]),
            (
                ndt(2025, 11, 28),
                vec![ndt(2025, 11, 28), ndt(2025, 11, 29), ndt(2025, 11, 30)],
            ),
            (
                ndt(2025, 2, 3),
                vec![ndt(2025, 2, 3), ndt(2025, 2, 2), ndt(2025, 2, 1)],
            ),
        ];

        for option in options {
            let result = get_unadjusteds(
                &option.0,
                &Adjuster::ModifiedFollowing {},
                &fixture_bus_cal(),
            );

            assert_eq!(result, option.1);
        }
    }

    #[test]
    fn test_vec_from() {
        let options: Vec<(Vec<NaiveDateTime>, Vec<RollDay>)> = vec![
            (
                vec![ndt(2000, 2, 29)],
                vec![RollDay::Day(29), RollDay::Day(30), RollDay::Day(31)],
            ),
            (vec![ndt(2025, 11, 28)], vec![RollDay::Day(28)]),
            (
                vec![ndt(2025, 3, 19)],
                vec![RollDay::Day(19), RollDay::IMM {}],
            ),
            (vec![ndt(2025, 9, 15)], vec![RollDay::Day(15)]),
        ];

        for option in options {
            let result = RollDay::vec_from(&option.0);
            assert_eq!(result, option.1);
        }
    }

    #[test]
    fn test_vec_from_multiple() {
        let options: Vec<(Vec<NaiveDateTime>, Vec<RollDay>)> = vec![
            (
                vec![ndt(2000, 2, 29)],
                vec![RollDay::Day(29), RollDay::Day(30), RollDay::Day(31)],
            ),
            (
                vec![ndt(2025, 11, 28), ndt(2025, 11, 29), ndt(2025, 11, 30)],
                vec![
                    RollDay::Day(28),
                    RollDay::Day(29),
                    RollDay::Day(30),
                    RollDay::Day(31),
                ],
            ),
            (
                vec![ndt(2025, 3, 19)],
                vec![RollDay::Day(19), RollDay::IMM()],
            ),
            (
                vec![ndt(2025, 9, 15), ndt(2025, 9, 14), ndt(2025, 9, 13)],
                vec![RollDay::Day(15), RollDay::Day(14), RollDay::Day(13)],
            ),
        ];

        for option in options {
            let result = RollDay::vec_from(&option.0);
            assert_eq!(result, option.1);
        }
    }
}
