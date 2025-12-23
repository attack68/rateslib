use crate::scheduling::{ndt, Adjuster, Cal, Calendar, DateRoll, RollDay};
use chrono::prelude::*;
use chrono::Months;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};

/// Specifier for generating unadjusted scheduling periods.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Frequency {
    /// A set number of business days, defined by a [`Calendar`], which can only align with a
    /// business day as defined by that [`Calendar`].
    BusDays { number: i32, calendar: Calendar },
    /// A set number of calendar days, which can align with any unadjusted date. To achieve a
    /// `Weeks` variant use an appropriate `number` of days.
    CalDays { number: i32 },
    /// A set number of calendar months, with a potential [`RollDay`].
    /// To achieve a `Years` variant use an appropriate `number` of months.
    Months { number: i32, roll: Option<RollDay> },
    /// Only ever defining one single period, and which can align with any unadjusted date.
    Zero {},
}

/// Used to define periods of financial instrument schedules.
pub trait Scheduling {
    /// Validate if an unadjusted date aligns with the scheduling object.
    fn try_udate(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr>;

    /// Calculate the next unadjusted scheduling period date from a given `date`.
    ///
    /// <div class="warning">
    ///
    /// The input `date` is not checked to align with the scheduling object. This can lead to
    /// to optically unexpected results (see examples). If a check on the date is required use the
    /// [`try_unext`](Scheduling::try_unext) method instead.
    ///
    /// </div>
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Frequency, Scheduling, ndt, RollDay};
    /// let f = Frequency::Months{number:1, roll: Some(RollDay::Day(1))};
    /// let result = f.next(&ndt(2000, 1, 31));
    /// assert_eq!(ndt(2000, 2, 1), result);
    /// assert!(f.try_unext(&ndt(2000, 1, 31)).is_err());
    /// ```
    fn next(&self, date: &NaiveDateTime) -> NaiveDateTime;

    /// Calculate the previous unadjusted scheduling period date from a given `date`.
    ///
    /// <div class="warning">
    ///
    /// The input `date` is not checked to align with the scheduling object. This can lead to
    /// to optically unexpected results (see examples). If a check on the date is required use the
    /// [`try_uprevious`](Scheduling::try_uprevious) method instead.
    ///
    /// </div>
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Frequency, Scheduling, ndt, RollDay};
    /// let f = Frequency::Months{number:1, roll: Some(RollDay::Day(31))};
    /// let result = f.previous(&ndt(2000, 2, 1));
    /// assert_eq!(ndt(2000, 1, 31), result);
    /// assert!(f.try_uprevious(&ndt(2000, 2, 1)).is_err());
    /// ```
    fn previous(&self, date: &NaiveDateTime) -> NaiveDateTime;

    /// Return a vector of unadjusted regular scheduling dates if it exists.
    ///
    /// # Notes
    /// In many standard cases this will simply use the provided method
    /// [`try_uregular_from_unext`](Scheduling::try_uregular_from_unext), but allows for custom
    /// implementations when required.
    fn try_uregular(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
    ) -> Result<Vec<NaiveDateTime>, PyErr>;

    /// Calculate the next unadjusted scheduling period date from an unadjusted base date.
    ///     
    /// # Notes
    /// This method first checks that the `udate` is valid and returns an error if not.
    fn try_unext(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let _ = self.try_udate(udate)?;
        Ok(self.next(udate))
    }

    /// Calculate the previous unadjusted scheduling period date from an unadjusted base date.
    ///
    /// # Notes
    /// This method first checks that the `udate` is valid and returns an error if not.
    fn try_uprevious(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let _ = self.try_udate(udate)?;
        Ok(self.previous(udate))
    }

    /// Return a vector of unadjusted regular scheduling dates if it exists.
    ///
    /// # Notes
    /// This method begins with ``ueffective`` and repeatedly applies [`try_unext`](Scheduling::try_unext)
    /// to derive all appropriate dates until ``utermination``.
    fn try_uregular_from_unext(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
    ) -> Result<Vec<NaiveDateTime>, PyErr> {
        let mut v: Vec<NaiveDateTime> = vec![];
        let mut date = *ueffective;
        while date < *utermination {
            v.push(date);
            date = self.try_unext(&date)?;
        }
        if date == *utermination {
            v.push(*utermination);
            Ok(v)
        } else {
            Err(PyValueError::new_err(
                "Input dates to Frequency do not define a regular unadjusted schedule",
            ))
        }
    }

    /// Check if two given unadjusted dates define a **regular period** under a scheduling object.
    ///
    /// # Notes
    /// This method tests if [`try_uregular`](Scheduling::try_uregular) has exactly two dates.
    fn is_regular_period(&self, ueffective: &NaiveDateTime, utermination: &NaiveDateTime) -> bool {
        let s = self.try_uregular(ueffective, utermination);
        match s {
            Ok(v) => v.len() == 2,
            Err(_) => false,
        }
    }

    /// Check if two given unadjusted dates define a **short front stub period** under a scheduling object.
    ///
    /// # Notes
    /// This method tests if [`try_uprevious`](Scheduling::try_uprevious) is before `ueffective`.
    /// If dates are undeterminable this returns `false`.
    fn is_short_front_stub(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
    ) -> bool {
        let quasi = self.try_uprevious(utermination);
        match quasi {
            Ok(date) => date < *ueffective,
            Err(_) => false,
        }
    }

    /// Check if two given unadjusted dates define a **long front stub period** under a scheduling object.
    fn is_long_front_stub(&self, ueffective: &NaiveDateTime, utermination: &NaiveDateTime) -> bool {
        let quasi = self.try_uprevious(utermination);
        match quasi {
            Ok(date) if *ueffective < date => {
                let quasi_2 = self.try_uprevious(&date);
                match quasi_2 {
                    Ok(date) => date <= *ueffective, // for long stub equal to allowed
                    Err(_) => false,
                }
            }
            _ => false,
        }
    }

    /// Check if two given unadjusted dates define a **short back stub period** under a scheduling object.
    ///
    /// # Notes
    /// This method tests if [Scheduling::try_unext] is after `utermination`.
    /// If dates are undeterminable this returns `false`.
    fn is_short_back_stub(&self, ueffective: &NaiveDateTime, utermination: &NaiveDateTime) -> bool {
        let quasi = self.try_unext(ueffective);
        match quasi {
            Ok(date) => *utermination < date,
            Err(_) => false,
        }
    }

    /// Check if two given unadjusted dates define a **long back stub period** under a scheduling object.
    fn is_long_back_stub(&self, ueffective: &NaiveDateTime, utermination: &NaiveDateTime) -> bool {
        let quasi = self.try_unext(ueffective);
        match quasi {
            Ok(date) if date < *utermination => {
                let quasi_2 = self.try_unext(&date);
                match quasi_2 {
                    Ok(date) => *utermination <= date, // for long stub equal to allowed.
                    Err(_) => false,
                }
            }
            _ => false,
        }
    }

    /// Check if two given unadjusted dates define any **front stub** under a scheduling object.
    ///
    /// # Notes
    /// If dates are undeterminable this returns `false`.
    fn is_front_stub(&self, ueffective: &NaiveDateTime, utermination: &NaiveDateTime) -> bool {
        self.is_short_front_stub(ueffective, utermination)
            || self.is_long_front_stub(ueffective, utermination)
    }

    /// Check if two given unadjusted dates define any **back stub** under a scheduling object.
    ///
    /// # Notes
    /// If dates are undeterminable this returns `false`.
    fn is_back_stub(&self, ueffective: &NaiveDateTime, utermination: &NaiveDateTime) -> bool {
        self.is_short_back_stub(ueffective, utermination)
            || self.is_long_back_stub(ueffective, utermination)
    }

    /// Infer an unadjusted front stub date from unadjusted irregular schedule dates.
    ///
    /// # Notes
    /// If a regular schedule is defined then the result will hold `None` as no stub is required.
    /// If a stub can be inferred then it will be returned as `Some(date)`.
    /// An errors will be returned if the dates are too close together to infer stubs and do not
    /// define a regular period.
    fn try_infer_ufront_stub(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
        short: bool,
    ) -> Result<Option<NaiveDateTime>, PyErr> {
        let mut date = *utermination;
        while date > *ueffective {
            date = self.try_uprevious(&date)?;
        }
        if date == *ueffective {
            // defines a regular schedule and no stub is required.
            Ok(None)
        } else {
            if short {
                date = self.try_unext(&date)?;
            } else {
                date = self.try_unext(&date)?;
                date = self.try_unext(&date)?;
            }
            if date >= *utermination {
                // then the dates are too close together to define a stub
                Ok(None)
            } else {
                // return the valid stub date
                Ok(Some(date))
            }
        }
    }

    /// Infer an unadjusted back stub date from unadjusted irregular schedule dates.
    ///
    /// # Notes
    /// If a regular schedule is defined then the result will hold `None` as no stub is required.
    /// If a stub can be inferred then it will be returned as `Some(date)`.
    /// An errors will be returned if the dates are too close together to infer stubs and do not
    /// define a regular period.
    fn try_infer_uback_stub(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
        short: bool,
    ) -> Result<Option<NaiveDateTime>, PyErr> {
        let mut date = *ueffective;
        while date < *utermination {
            date = self.try_unext(&date)?;
        }
        if date == *utermination {
            // regular schedule so no stub required
            Ok(None)
        } else {
            if short {
                date = self.try_uprevious(&date)?;
            } else {
                date = self.try_uprevious(&date)?;
                date = self.try_uprevious(&date)?;
            }
            if date <= *ueffective {
                // dates are too close together to define a stub.
                Ok(None)
            } else {
                // return the valid stub
                Ok(Some(date))
            }
        }
    }

    /// Get the approximate number of coupons per annum.
    ///
    /// This will average the number coupons paid in 50 year period.
    fn periods_per_annum(&self) -> f64 {
        periods_per_annum(self)
    }
}

impl Frequency {
    /// Get a vector of possible, fully specified [`Frequency`] variants for a series of unadjusted dates.
    ///
    /// # Notes
    /// This method exists primarily to resolve cases when the [`RollDay`] on a
    /// [`Frequency::Months`](Frequency) variant is `None`, and there are multiple possibilities. In this case
    /// the method [`RollDay::vec_from`] is called internally.
    ///
    /// If the [`Frequency`] variant does not align with any of the provided unadjusted dates this
    /// will return an error.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Frequency, ndt, RollDay};
    /// // The RollDay is unspecified here
    /// let f = Frequency::Months{number: 3, roll: None};
    /// let result = f.try_vec_from(&vec![ndt(2024, 2, 29)]);
    /// assert_eq!(result.unwrap(), vec![
    ///     Frequency::Months{number: 3, roll: Some(RollDay::Day(29))},
    ///     Frequency::Months{number: 3, roll: Some(RollDay::Day(30))},
    ///     Frequency::Months{number: 3, roll: Some(RollDay::Day(31))},
    /// ]);
    /// ```
    pub fn try_vec_from(&self, udates: &Vec<NaiveDateTime>) -> Result<Vec<Frequency>, PyErr> {
        match self {
            Frequency::Months {
                number: n,
                roll: None,
            } => {
                // the RollDay is unspecified so get all possible RollDay variants
                Ok(RollDay::vec_from(udates)
                    .into_iter()
                    .map(|r| Frequency::Months {
                        number: *n,
                        roll: Some(r),
                    })
                    .collect())
            }
            _ => {
                // the Frequency is fully specified so return single element vector if
                // at least 1 udate is valid
                for udate in udates {
                    if self.try_udate(udate).is_ok() {
                        return Ok(vec![self.clone()]);
                    }
                }
                Err(PyValueError::new_err(
                    "The Frequency does not align with any of the `udates`.",
                ))
            }
        }
    }
}

impl Scheduling for Frequency {
    /// Validate if an unadjusted date aligns with the specified [Frequency] variant.
    ///
    /// # Notes
    /// This method will return error in one of two cases:
    /// - The `udate` does not align with the fully defined variant.
    /// - The variant is not fully defined (e.g. a [`Months`](Frequency) variant is missing
    ///   a [`RollDay`](RollDay)) and cannot make the determination.
    ///
    /// Therefore,
    /// - For a [CalDays](Frequency) variant or [Zero](Frequency) variant, any ``udate`` is valid.
    /// - For a [BusDays](Frequency) variant, ``udate`` must be a business day.
    /// - For a [Months](Frequency) variant, ``udate`` must align with the [RollDay]. If no [RollDay] is
    ///   specified an error will always be returned.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Frequency, RollDay, ndt, Scheduling};
    /// let result = Frequency::Months{number: 1, roll: Some(RollDay::IMM{})}.try_udate(&ndt(2025, 7, 16));
    /// assert!(result.is_ok());
    ///
    /// let result = Frequency::Months{number: 1, roll: None}.try_udate(&ndt(2025, 7, 16));
    /// assert!(result.is_err());
    /// ```
    fn try_udate(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        match self {
            Frequency::BusDays {
                number: _n,
                calendar: c,
            } => {
                if c.is_bus_day(udate) {
                    Ok(*udate)
                } else {
                    Err(PyValueError::new_err(
                        "`udate` is not a business day of the given calendar.",
                    ))
                }
            }
            Frequency::CalDays { number: _n } => Ok(*udate),
            Frequency::Months {
                number: _n,
                roll: r,
            } => match r {
                Some(r) => r.try_udate(udate),
                None => Err(PyValueError::new_err(
                    "`udate` cannot be validated since RollDay is None.",
                )),
            },
            Frequency::Zero {} => Ok(*udate),
        }
    }

    fn next(&self, date: &NaiveDateTime) -> NaiveDateTime {
        match self {
            Frequency::BusDays {
                number: n,
                calendar: c,
            } => c.lag_bus_days(date, *n, false),
            Frequency::CalDays { number: n } => {
                let cal = Cal::new(vec![], vec![]);
                cal.add_cal_days(date, *n, &Adjuster::Actual {})
            }
            Frequency::Months { number: n, roll: r } => match r {
                Some(r) => r.uadd(date, *n),
                None => RollDay::Day(date.day()).uadd(date, *n),
            },
            Frequency::Zero {} => ndt(9999, 1, 1),
        }
    }

    fn previous(&self, date: &NaiveDateTime) -> NaiveDateTime {
        match self {
            Frequency::BusDays {
                number: n,
                calendar: c,
            } => c.lag_bus_days(date, -(*n), false),
            Frequency::CalDays { number: n } => {
                let cal = Cal::new(vec![], vec![]);
                cal.add_cal_days(date, -(*n), &Adjuster::Actual {})
            }
            Frequency::Months { number: n, roll: r } => match r {
                Some(r) => r.uadd(date, -(*n)),
                None => RollDay::Day(date.day()).uadd(date, -(*n)),
            },
            Frequency::Zero {} => ndt(1500, 1, 1),
        }
    }

    fn try_uregular(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
    ) -> Result<Vec<NaiveDateTime>, PyErr> {
        match self {
            Frequency::Zero {} => Ok(vec![*ueffective, *utermination]),
            _ => self.try_uregular_from_unext(ueffective, utermination),
        }
    }

    fn periods_per_annum(&self) -> f64 {
        match self {
            Frequency::Zero {} => 0.01_f64,
            Frequency::Months { number: 1, roll: _ } => 12.0_f64,
            Frequency::Months { number: 2, roll: _ } => 6.0_f64,
            Frequency::Months { number: 3, roll: _ } => 4.0_f64,
            Frequency::Months { number: 4, roll: _ } => 3.0_f64,
            Frequency::Months { number: 6, roll: _ } => 2.0_f64,
            Frequency::Months {
                number: 12,
                roll: _,
            } => 1.0_f64,
            _ => periods_per_annum(self),
        }
    }
}

fn periods_per_annum<T: Scheduling + ?Sized>(obj: &T) -> f64 {
    let mut date = obj.next(&ndt(1999, 12, 31));
    if date > ndt(2049, 12, 31) {
        // then the next method has generated an unusually long period. return nominal value
        return 0.01_f64;
    }
    let estimated_end = date + Months::new(600);
    let mut counter = 0_f64;
    let count: f64;
    loop {
        counter += 1.0;
        let prev = date;
        date = obj.next(&prev);
        if date < prev {
            // Scheduling object is reversed so make a correction.
            date = obj.previous(&prev)
        }
        if date >= estimated_end {
            if (estimated_end - prev) < (date - estimated_end) {
                count = f64::max(1.0, counter - 1.0);
            } else {
                count = counter;
            }
            break;
        }
    }
    count / 50.0
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::ndt;

    #[test]
    fn test_try_udate() {
        let options: Vec<(Frequency, NaiveDateTime)> = vec![
            (
                Frequency::BusDays {
                    number: 4,
                    calendar: Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                },
                ndt(2025, 7, 11),
            ),
            (Frequency::CalDays { number: 4 }, ndt(2025, 7, 11)),
            (Frequency::Zero {}, ndt(2025, 7, 11)),
            (
                Frequency::Months {
                    number: 4,
                    roll: Some(RollDay::Day(11)),
                },
                ndt(2025, 7, 11),
            ),
        ];
        for option in options {
            let result = option.0.try_udate(&option.1).unwrap();
            assert_eq!(result, option.1);
        }
    }

    #[test]
    fn test_try_udate_err() {
        let options: Vec<(Frequency, NaiveDateTime)> = vec![
            (
                Frequency::BusDays {
                    number: 4,
                    calendar: Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                },
                ndt(2025, 7, 12),
            ),
            (
                Frequency::Months {
                    number: 4,
                    roll: None,
                },
                ndt(2025, 7, 12),
            ),
            (
                Frequency::Months {
                    number: 4,
                    roll: Some(RollDay::IMM {}),
                },
                ndt(2025, 7, 1),
            ),
        ];
        for option in options {
            assert!(option.0.try_udate(&option.1).is_err());
        }
    }

    #[test]
    fn test_is_regular_period_ok() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool)> = vec![
            (
                Frequency::CalDays { number: 5 },
                ndt(2000, 1, 1),
                ndt(2000, 1, 6),
                true,
            ),
            (
                Frequency::CalDays { number: 5 },
                ndt(2000, 1, 1),
                ndt(2000, 1, 5),
                false,
            ),
            (
                Frequency::Months {
                    number: 5,
                    roll: Some(RollDay::Day(1)),
                },
                ndt(2000, 1, 1),
                ndt(2000, 6, 1),
                true,
            ),
            (
                Frequency::Months {
                    number: 5,
                    roll: Some(RollDay::Day(1)),
                },
                ndt(2000, 1, 1),
                ndt(2000, 6, 5),
                false,
            ),
        ];

        for option in options {
            let result = option.0.is_regular_period(&option.1, &option.2);
            assert_eq!(result, option.3);
        }
    }

    #[test]
    fn test_is_short_front_stub() {
        assert_eq!(
            true,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_short_front_stub(&ndt(2000, 1, 1), &ndt(2000, 1, 20))
        );
        assert_eq!(
            false,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(1))
            }
            .is_short_front_stub(&ndt(2000, 1, 1), &ndt(2000, 2, 1))
        );
        assert_eq!(
            false,
            Frequency::Months {
                number: 1,
                roll: None
            }
            .is_short_front_stub(&ndt(2000, 1, 1), &ndt(2000, 1, 15))
        );
    }

    #[test]
    fn test_is_long_front_stub() {
        assert_eq!(
            // is a valid long stub
            true,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_front_stub(&ndt(2000, 1, 1), &ndt(2000, 2, 20))
        );
        assert_eq!(
            // is a valid 2-regular period long stub
            true,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_front_stub(&ndt(2000, 1, 20), &ndt(2000, 3, 20))
        );
        assert_eq!(
            // is too short
            false,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_front_stub(&ndt(2000, 1, 25), &ndt(2000, 2, 20))
        );
        assert_eq!(
            // is too long
            false,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_front_stub(&ndt(2000, 1, 15), &ndt(2000, 3, 20))
        );
    }

    #[test]
    fn test_is_long_back_stub() {
        assert_eq!(
            // is a valid long stub
            true,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_back_stub(&ndt(2000, 1, 20), &ndt(2000, 2, 28))
        );
        assert_eq!(
            // is a valid 2-regular period long stub
            true,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_back_stub(&ndt(2000, 1, 20), &ndt(2000, 3, 20))
        );
        assert_eq!(
            // is too short
            false,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_back_stub(&ndt(2000, 1, 20), &ndt(2000, 2, 10))
        );
        assert_eq!(
            // is too long
            false,
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::Day(20))
            }
            .is_long_front_stub(&ndt(2000, 1, 20), &ndt(2000, 3, 30))
        );
    }

    // #[test]
    // fn test_try_scheduling() {
    //     let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime)> = vec![
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: None,
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 8, 30),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 2,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 9, 30),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 3,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 10, 30),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 4,
    //                 roll: None,
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 11, 30),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 6,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2023, 1, 30),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 12,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2023, 7, 30),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::Day { day: 31 }),
    //             },
    //             ndt(2022, 6, 30),
    //             ndt(2022, 7, 31),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::IMM {}),
    //             },
    //             ndt(2022, 6, 15),
    //             ndt(2022, 7, 20),
    //         ),
    //         (
    //             Frequency::CalDays { number: 5 },
    //             ndt(2022, 6, 15),
    //             ndt(2022, 6, 20),
    //         ),
    //         (
    //             Frequency::CalDays { number: 14 },
    //             ndt(2022, 6, 15),
    //             ndt(2022, 6, 29),
    //         ),
    //         (
    //             Frequency::BusDays {
    //                 number: 5,
    //                 calendar: Calendar::Cal(Cal::new(vec![], vec![5, 6])),
    //             },
    //             ndt(2025, 6, 23),
    //             ndt(2025, 6, 30),
    //         ),
    //         (Frequency::Zero {}, ndt(1500, 1, 1), ndt(9999, 1, 1)),
    //     ];
    //     for option in options.iter() {
    //         assert_eq!(option.2, option.0.try_unext(&option.1).unwrap());
    //         assert_eq!(option.1, option.0.try_uprevious(&option.2).unwrap());
    //     }
    // }
    //
    #[test]
    fn test_get_uschedule_imm() {
        // test the example given in Coding Interest Rates
        let result = Frequency::Months {
            number: 1,
            roll: Some(RollDay::IMM {}),
        }
        .try_uregular(&ndt(2023, 3, 15), &ndt(2023, 9, 20))
        .unwrap();
        assert_eq!(
            result,
            vec![
                ndt(2023, 3, 15),
                ndt(2023, 4, 19),
                ndt(2023, 5, 17),
                ndt(2023, 6, 21),
                ndt(2023, 7, 19),
                ndt(2023, 8, 16),
                ndt(2023, 9, 20)
            ]
        );
    }
    //
    // #[test]
    // fn test_get_uschedule() {
    //     let result = Frequency::Months {
    //         number: 3,
    //         roll: Some(RollDay::Day { day: 1 }),
    //     }
    //     .try_uregular(&ndt(2000, 1, 1), &ndt(2001, 1, 1))
    //     .unwrap();
    //     assert_eq!(
    //         result,
    //         vec![
    //             ndt(2000, 1, 1),
    //             ndt(2000, 4, 1),
    //             ndt(2000, 7, 1),
    //             ndt(2000, 10, 1),
    //             ndt(2001, 1, 1)
    //         ]
    //     );
    // }

    // #[test]
    // fn test_infer_ufront() {
    //     let options: Vec<(
    //         Frequency,
    //         NaiveDateTime,
    //         NaiveDateTime,
    //         bool,
    //         Option<NaiveDateTime>,
    //     )> = vec![
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::Day { day: 15 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 10, 15),
    //             true,
    //             Some(ndt(2022, 8, 15)),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: None,
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 10, 15),
    //             false,
    //             Some(ndt(2022, 9, 15)),
    //         ),
    //     ];
    //
    //     for option in options.iter() {
    //         assert_eq!(
    //             option.4,
    //             option
    //                 .0
    //                 .try_infer_ufront_stub(&option.1, &option.2, option.3)
    //                 .unwrap()
    //         );
    //     }
    // }

    // #[test]
    // fn test_infer_ufront_err() {
    //     let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool)> = vec![
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::Day { day: 15 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 8, 15),
    //             true,
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: None,
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 9, 15),
    //             false,
    //         ),
    //         (
    //             Frequency::Zero {},
    //             ndt(2022, 7, 30),
    //             ndt(2022, 9, 15),
    //             false,
    //         ),
    //     ];
    //
    //     for option in options.iter() {
    //         let result = option
    //             .0
    //             .try_infer_ufront_stub(&option.1, &option.2, option.3)
    //             .is_err();
    //         assert_eq!(true, result);
    //     }
    // }

    // #[test]
    // fn test_infer_uback() {
    //     let options: Vec<(
    //         Frequency,
    //         NaiveDateTime,
    //         NaiveDateTime,
    //         bool,
    //         Option<NaiveDateTime>,
    //     )> = vec![
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 10, 15),
    //             true,
    //             Some(ndt(2022, 9, 30)),
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 10, 15),
    //             false,
    //             Some(ndt(2022, 8, 30)),
    //         ),
    //     ];
    //
    //     for option in options.iter() {
    //         assert_eq!(
    //             option.4,
    //             option
    //                 .0
    //                 .try_infer_uback_stub(&option.1, &option.2, option.3)
    //                 .unwrap()
    //         );
    //     }
    // }
    //
    // #[test]
    // fn test_infer_uback_err() {
    //     let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool)> = vec![
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 8, 15),
    //             true,
    //         ),
    //         (
    //             Frequency::Months {
    //                 number: 1,
    //                 roll: Some(RollDay::Day { day: 30 }),
    //             },
    //             ndt(2022, 7, 30),
    //             ndt(2022, 9, 15),
    //             false,
    //         ),
    //     ];
    //
    //     for option in options.iter() {
    //         let result = option
    //             .0
    //             .try_infer_uback_stub(&option.1, &option.2, option.3)
    //             .is_err();
    //         assert_eq!(true, result);
    //     }
    // }
    //
    #[test]
    fn test_try_vec_from() {
        let options: Vec<(Frequency, Vec<NaiveDateTime>, Vec<Frequency>)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: None,
                },
                vec![ndt(2022, 7, 30)],
                vec![Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(30)),
                }],
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: None,
                },
                vec![ndt(2022, 2, 28)],
                vec![
                    Frequency::Months {
                        number: 1,
                        roll: Some(RollDay::Day(28)),
                    },
                    Frequency::Months {
                        number: 1,
                        roll: Some(RollDay::Day(29)),
                    },
                    Frequency::Months {
                        number: 1,
                        roll: Some(RollDay::Day(30)),
                    },
                    Frequency::Months {
                        number: 1,
                        roll: Some(RollDay::Day(31)),
                    },
                ],
            ),
            (
                Frequency::CalDays { number: 1 },
                vec![ndt(2022, 2, 28)],
                vec![Frequency::CalDays { number: 1 }],
            ),
        ];

        for option in options.iter() {
            let result = option.0.try_vec_from(&option.1).unwrap();
            assert_eq!(option.2, result);
        }
    }

    #[test]
    fn test_try_vec_from_err() {
        let options: Vec<(Frequency, Vec<NaiveDateTime>)> = vec![(
            Frequency::Months {
                number: 1,
                roll: Some(RollDay::IMM {}),
            },
            vec![ndt(2022, 7, 30)],
        )];

        for option in options.iter() {
            assert_eq!(true, option.0.try_vec_from(&option.1).is_err());
        }
    }

    #[test]
    fn test_coupons_per_annum() {
        let options: Vec<(Frequency, f64)> = vec![
            (Frequency::CalDays { number: 365 }, 1.0),
            (Frequency::CalDays { number: 182 }, 2.0),
            (Frequency::CalDays { number: 183 }, 2.0),
            (Frequency::CalDays { number: 91 }, 4.02),
            (Frequency::CalDays { number: 28 }, 13.04),
            (Frequency::CalDays { number: 7 }, 52.18),
            (
                Frequency::BusDays {
                    number: 5,
                    calendar: Cal::new(vec![], vec![5, 6]).into(),
                },
                52.18,
            ),
            (
                Frequency::BusDays {
                    number: 63,
                    calendar: Cal::new(vec![], vec![5, 6]).into(),
                },
                4.14,
            ),
            (
                Frequency::BusDays {
                    number: 62,
                    calendar: Cal::new(vec![], vec![5, 6]).into(),
                },
                4.2,
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: None,
                },
                12.0,
            ),
            (
                Frequency::Months {
                    number: 2,
                    roll: None,
                },
                6.0,
            ),
            (
                Frequency::Months {
                    number: 3,
                    roll: None,
                },
                4.0,
            ),
            (
                Frequency::Months {
                    number: 4,
                    roll: None,
                },
                3.0,
            ),
            (
                Frequency::Months {
                    number: 6,
                    roll: None,
                },
                2.0,
            ),
            (
                Frequency::Months {
                    number: 9,
                    roll: None,
                },
                1.34,
            ),
            (
                Frequency::Months {
                    number: 12,
                    roll: None,
                },
                1.0,
            ),
            (
                Frequency::Months {
                    number: 24,
                    roll: None,
                },
                0.5,
            ),
            (
                Frequency::Months {
                    number: 3,
                    roll: Some(RollDay::IMM()),
                },
                4.0,
            ),
            (Frequency::Zero {}, 0.01),
        ];
        for option in options {
            let result = option.0.periods_per_annum();
            assert_eq!(result, option.1);
        }
    }
}
