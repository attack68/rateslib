use crate::scheduling::{ndt, Adjuster, Cal, Calendar, DateRoll, RollDay};
use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};

/// A frequency for generating unadjusted scheduling periods.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Debug, Clone, PartialEq)]
pub enum Frequency {
    /// A set number of business days, defined by a [Calendar], which can only align with a
    /// business day as defined by that [Calendar].
    BusDays { number: u32, calendar: Calendar },
    /// A set number of calendar days, which can align with any unadjusted date. To achieve a
    /// `Weeks` variant use an appropriate number of `CalDays`.
    CalDays { number: u32 },
    /// A set number of calendar months, with a defined [RollDay]. This will align with any
    /// unadjusted date if no [RollDay] is specified, otherwise it must align with the [RollDay].
    /// To achieve a `Years` variant use an appropriate number of `Months`.
    Months { number: u32, roll: Option<RollDay> },
    /// Only ever defining one single period, and which can align with any unadjusted date.
    Zero {},
}

/// Used to define periods of financial instrument schedules.
pub trait Scheduling {
    /// Calculate the next unadjusted scheduling period date from an unadjusted base date.
    fn try_unext(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr>;

    /// Calculate the previous unadjusted scheduling period date from an unadjusted base date.
    fn try_uprevious(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr>;

    /// Return a vector of unadjusted regular scheduling dates if it exists.
    ///
    /// # Notes
    /// In many standard cases this will simply use the provided method
    /// [Scheduling::try_uregular_from_unext], but allows for custom implementations when required.
    fn try_uregular(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
    ) -> Result<Vec<NaiveDateTime>, PyErr>;

    /// Return a vector of unadjusted regular scheduling dates if it exists.
    ///
    /// # Notes
    /// This method begins with ``ueffective`` and repeatedly applies [Scheduling::try_unext]
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
                Err(PyValueError::new_err(
                    "Dates are too close together to infer the desired stub",
                ))
            } else {
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
                Err(PyValueError::new_err(
                    "Dates are too close together to infer the desired stub",
                ))
            } else {
                Ok(Some(date))
            }
        }
    }
}

impl Frequency {
    /// Validate if an unadjusted date aligns with the specified [Frequency] variant.
    pub fn try_udate(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        match self {
            Frequency::BusDays {
                number: _n,
                calendar: c,
            } => {
                if c.is_bus_day(udate) {
                    Ok(*udate)
                } else {
                    Err(PyValueError::new_err(
                        "Date is not a business day of the given calendar.",
                    ))
                }
            }
            Frequency::CalDays { number: _n } => Ok(*udate),
            Frequency::Months {
                number: _n,
                roll: r,
            } => match r {
                Some(r) => r.try_udate(udate),
                None => Ok(*udate),
            },
            Frequency::Zero {} => Ok(*udate),
        }
    }

    /// Get a vector of possible, fully specified [Frequency] variants for a series of unadjusted dates.
    ///
    /// # Notes
    /// If the [Frequency] variant does not align with any of the provided unadjusted dates this
    /// will return an error. If any variants have optional parameters, e.g. the [RollDay] on a
    /// `Months` variant, this method will try to return all possible, populated variant options.
    /// For this specific case, the method [RollDay::vec_from] is used.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Frequency, ndt};
    /// // The RollDay is unspecified here
    /// let f = Frequency::Months{number: 3, roll: None};
    /// let result = f.try_vec_from(&vec![ndt(2024, 2, 29)]);
    /// // Vec<Frequency::Months{number: 3, roll: RollDay::Day{day: 29}},
    /// //     Frequency::Months{number: 3, roll: RollDay::Day{day: 30}},
    /// //     Frequency::Months{number: 3, roll: RollDay::Day{day: 31}}>
    /// ```
    pub fn try_vec_from(&self, udates: &Vec<NaiveDateTime>) -> Result<Vec<Frequency>, PyErr> {
        match self {
            Frequency::Months {
                number: n,
                roll: None,
            } => {
                // the roll is unspecified so get all possible RollDay variants
                Ok(RollDay::vec_from(udates)
                    .into_iter()
                    .map(|r| Frequency::Months {
                        number: *n,
                        roll: Some(r),
                    })
                    .collect())
            }
            _ => {
                for udate in udates {
                    if self.try_udate(udate).is_ok() {
                        return Ok(vec![self.clone()]);
                    }
                }
                return Err(PyValueError::new_err(
                    "The Frequency does not align with any of the `udates`.",
                ));
            }
        }
    }
}

impl Scheduling for Frequency {
    /// Calculate the next unadjusted scheduling period date from an unadjusted base date.
    ///
    /// # Notes
    /// This method will first call [Frequency::try_vec_from] to ensure that ``udate`` aligns
    /// with the [Frequency] and to populate any unknown, optional parameters (e.g. [RollDay]).
    /// The first returned [Frequency] from this vector will be used to define the period.
    ///
    /// A `Zero` variant will return `ndt(9999, 1, 1)`.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Frequency, ndt, Scheduling};
    /// // The RollDay is unspecified here
    /// let f = Frequency::Months{number: 3, roll: None};
    /// let date = f.try_unext(&vec![ndt(2024, 2, 29)]);
    /// assert_eq!(ndt(2024, 5, 29), date.unwrap());
    /// ```
    fn try_unext(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let f: Vec<Frequency> = self.try_vec_from(&vec![*udate])?;
        match &f[0] {
            Frequency::BusDays {
                number: n,
                calendar: c,
            } => c.add_bus_days(udate, *n as i32, false),
            Frequency::CalDays { number: n } => {
                let cal = Cal::new(vec![], vec![]);
                Ok(cal.add_cal_days(udate, *n as i32, &Adjuster::Actual {}))
            }
            Frequency::Months { number: n, roll: r } => match r {
                Some(r) => Ok(r.uadd(udate, *n as i32)),
                // try_vec_from cannot yield a Frequency::Months variant with no RollDay
                None => panic!["This line should be functionally unreachable - please report."],
            },
            Frequency::Zero {} => Ok(ndt(9999, 1, 1)),
        }
    }

    /// Calculate the previous unadjusted scheduling period date from an unadjusted base date.
    ///
    /// # Notes
    /// This method will first call [Frequency::try_vec_from] to ensure that ``udate`` aligns
    /// with the [Frequency] and to populate any unknown, optional parameters (e.g. [RollDay]).
    /// The first returned [Frequency] from this vector will be used to define the period.
    ///
    /// A `Zero` variant will return `ndt(9999, 1, 1)`.
    ///
    /// # Examples
    /// ```rust
    /// # use rateslib::scheduling::{Frequency, ndt, Scheduling};
    /// // The RollDay is unspecified here
    /// let f = Frequency::Months{number: 3, roll: None};
    /// let date = f.try_uprevious(&vec![ndt(2024, 2, 29)]);
    /// assert_eq!(ndt(2023, 11, 29), date.unwrap());
    /// ```
    fn try_uprevious(&self, udate: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let f: Vec<Frequency> = self.try_vec_from(&vec![*udate])?;
        match &f[0] {
            Frequency::BusDays {
                number: n,
                calendar: c,
            } => c.add_bus_days(udate, -(*n as i32), false),
            Frequency::CalDays { number: n } => {
                let cal = Cal::new(vec![], vec![]);
                Ok(cal.add_cal_days(udate, -(*n as i32), &Adjuster::Actual {}))
            }
            Frequency::Months { number: n, roll: r } => match r {
                Some(r) => Ok(r.uadd(udate, -(*n as i32))),
                // try_vec_from cannot yield a Frequency::Months variant with no RollDay
                None => panic!["This line should be functionally unreachable - please report."],
            },
            Frequency::Zero {} => Ok(ndt(1500, 1, 1)),
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
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::ndt;

    #[test]
    fn test_try_scheduling() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: None,
                },
                ndt(2022, 7, 30),
                ndt(2022, 8, 30),
            ),
            (
                Frequency::Months {
                    number: 2,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 9, 30),
            ),
            (
                Frequency::Months {
                    number: 3,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 30),
            ),
            (
                Frequency::Months {
                    number: 4,
                    roll: None,
                },
                ndt(2022, 7, 30),
                ndt(2022, 11, 30),
            ),
            (
                Frequency::Months {
                    number: 6,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2023, 1, 30),
            ),
            (
                Frequency::Months {
                    number: 12,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2023, 7, 30),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day { day: 31 }),
                },
                ndt(2022, 6, 30),
                ndt(2022, 7, 31),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::IMM {}),
                },
                ndt(2022, 6, 15),
                ndt(2022, 7, 20),
            ),
            (
                Frequency::CalDays { number: 5 },
                ndt(2022, 6, 15),
                ndt(2022, 6, 20),
            ),
            (
                Frequency::CalDays { number: 14 },
                ndt(2022, 6, 15),
                ndt(2022, 6, 29),
            ),
            (
                Frequency::BusDays {
                    number: 5,
                    calendar: Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                },
                ndt(2025, 6, 23),
                ndt(2025, 6, 30),
            ),
            (Frequency::Zero {}, ndt(1500, 1, 1), ndt(9999, 1, 1)),
        ];
        for option in options.iter() {
            assert_eq!(option.2, option.0.try_unext(&option.1).unwrap());
            assert_eq!(option.1, option.0.try_uprevious(&option.2).unwrap());
        }
    }

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

    #[test]
    fn test_get_uschedule() {
        let result = Frequency::Months {
            number: 3,
            roll: Some(RollDay::Day { day: 1 }),
        }
        .try_uregular(&ndt(2000, 1, 1), &ndt(2001, 1, 1))
        .unwrap();
        assert_eq!(
            result,
            vec![
                ndt(2000, 1, 1),
                ndt(2000, 4, 1),
                ndt(2000, 7, 1),
                ndt(2000, 10, 1),
                ndt(2001, 1, 1)
            ]
        );
    }

    #[test]
    fn test_infer_ufront() {
        let options: Vec<(
            Frequency,
            NaiveDateTime,
            NaiveDateTime,
            bool,
            Option<NaiveDateTime>,
        )> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day { day: 15 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                true,
                Some(ndt(2022, 8, 15)),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: None,
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                false,
                Some(ndt(2022, 9, 15)),
            ),
        ];

        for option in options.iter() {
            assert_eq!(
                option.4,
                option
                    .0
                    .try_infer_ufront_stub(&option.1, &option.2, option.3)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_infer_ufront_err() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day { day: 15 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 8, 15),
                true,
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: None,
                },
                ndt(2022, 7, 30),
                ndt(2022, 9, 15),
                false,
            ),
            (
                Frequency::Zero {},
                ndt(2022, 7, 30),
                ndt(2022, 9, 15),
                false,
            ),
        ];

        for option in options.iter() {
            let result = option
                .0
                .try_infer_ufront_stub(&option.1, &option.2, option.3)
                .is_err();
            assert_eq!(true, result);
        }
    }

    #[test]
    fn test_infer_uback() {
        let options: Vec<(
            Frequency,
            NaiveDateTime,
            NaiveDateTime,
            bool,
            Option<NaiveDateTime>,
        )> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                true,
                Some(ndt(2022, 9, 30)),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                false,
                Some(ndt(2022, 8, 30)),
            ),
        ];

        for option in options.iter() {
            assert_eq!(
                option.4,
                option
                    .0
                    .try_infer_uback_stub(&option.1, &option.2, option.3)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_infer_uback_err() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 8, 15),
                true,
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day { day: 30 }),
                },
                ndt(2022, 7, 30),
                ndt(2022, 9, 15),
                false,
            ),
        ];

        for option in options.iter() {
            let result = option
                .0
                .try_infer_uback_stub(&option.1, &option.2, option.3)
                .is_err();
            assert_eq!(true, result);
        }
    }

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
                    roll: Some(RollDay::Day { day: 30 }),
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
                        roll: Some(RollDay::Day { day: 28 }),
                    },
                    Frequency::Months {
                        number: 1,
                        roll: Some(RollDay::Day { day: 29 }),
                    },
                    Frequency::Months {
                        number: 1,
                        roll: Some(RollDay::Day { day: 30 }),
                    },
                    Frequency::Months {
                        number: 1,
                        roll: Some(RollDay::Day { day: 31 }),
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
    fn test_delete() {
        let f = Frequency::Months {
            number: 3,
            roll: None,
        };
        let result = f.try_vec_from(&vec![ndt(2024, 2, 29)]);
        println!("{:?}", result);
    }
}
