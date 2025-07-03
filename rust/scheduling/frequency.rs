use crate::scheduling::{ndt, Adjuster, Cal, Calendar, DateRoll, RollDay};
use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};

/// A frequency for generating unadjusted schedules and periods.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Debug, Clone, PartialEq)]
pub enum Frequency {
    /// A set number of business days, defined by a given calendar.
    BusDays { number: u32, calendar: Calendar },
    /// A set number of calendar days
    CalDays { number: u32 },
    /// A set number of calendar weeks.
    Weeks { number: u32 },
    /// A set number of calendar months, with a defined roll day.
    Months { number: u32, roll: RollDay },
    /// Only ever a single period
    Zero {},
}

/// Used to define periods of financial instrument schedules.
pub trait Scheduling {
    /// Calculate the next unadjusted period date in a schedule from an unadjusted effective date.
    fn try_next(&self, ueffective: &NaiveDateTime) -> Result<NaiveDateTime, PyErr>;

    /// Calculate the previous unadjusted period date in a schedule from an unadjusted effective date.
    fn try_previous(&self, ueffective: &NaiveDateTime) -> Result<NaiveDateTime, PyErr>;

    /// Return an unadjusted regular schedule if the given dates define such.
    fn try_uregular(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
    ) -> Result<Vec<NaiveDateTime>, PyErr> {
        let mut v: Vec<NaiveDateTime> = vec![];
        let mut date = *ueffective;
        while date < *utermination {
            v.push(date);
            date = self.try_next(&date)?;
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
    fn try_infer_front_stub(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
        short: bool,
    ) -> Result<NaiveDateTime, PyErr> {
        let mut date = *utermination;
        while date > *ueffective {
            date = self.try_previous(&date)?;
        }
        if date == *ueffective {
            Err(PyValueError::new_err("Input dates to Frequency define a regular unadjusted schedule, and do not require a stub date"))
        } else {
            if short {
                date = self.try_next(&date)?;
            } else {
                date = self.try_next(&date)?;
                date = self.try_next(&date)?;
            }
            if date >= *utermination {
                Err(PyValueError::new_err(
                    "Dates are too close together to infer the desired stub",
                ))
            } else {
                Ok(date)
            }
        }
    }

    /// Infer an unadjusted back stub date from unadjusted irregular schedule dates.
    fn try_infer_back_stub(
        &self,
        ueffective: &NaiveDateTime,
        utermination: &NaiveDateTime,
        short: bool,
    ) -> Result<NaiveDateTime, PyErr> {
        let mut date = *ueffective;
        while date < *utermination {
            date = self.try_next(&date)?;
        }
        if date == *utermination {
            Err(PyValueError::new_err("Input dates to Frequency define a regular unadjusted schedule, and do not require a stub date"))
        } else {
            if short {
                date = self.try_previous(&date)?;
            } else {
                date = self.try_previous(&date)?;
                date = self.try_previous(&date)?;
            }
            if date <= *ueffective {
                Err(PyValueError::new_err(
                    "Dates are too close together to infer the desired stub",
                ))
            } else {
                Ok(date)
            }
        }
    }
}

impl Frequency {
    /// Validate if a given date is an unadjusted date available to the scheduling object.
    pub fn try_udate(&self, date: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        match self {
            Frequency::BusDays {
                number: _n,
                calendar: c,
            } => {
                if c.is_bus_day(date) {
                    Ok(*date)
                } else {
                    Err(PyValueError::new_err(
                        "Date is not a business day of the given calendar.",
                    ))
                }
            }
            Frequency::CalDays { number: _n } => Ok(*date),
            Frequency::Weeks { number: _n } => Ok(*date),
            Frequency::Months {
                number: _n,
                roll: r,
            } => Ok(r.try_udate(date)?),
            Frequency::Zero {} => Ok(*date),
        }
    }
}

impl Scheduling for Frequency {
    fn try_next(&self, ueffective: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let _ = self.try_udate(ueffective)?;
        let cal = Cal::new(vec![], vec![]);
        match self {
            Frequency::BusDays {
                number: n,
                calendar: c,
            } => c.add_bus_days(ueffective, *n as i32, false),
            Frequency::CalDays { number: n } => {
                Ok(cal.add_cal_days(ueffective, *n as i32, &Adjuster::Actual {}))
            }
            Frequency::Weeks { number: n } => {
                Ok(cal.add_cal_days(ueffective, *n as i32 * 7, &Adjuster::Actual {}))
            }
            Frequency::Months { number: n, roll: r } => {
                Ok(cal.add_months(ueffective, *n as i32, &Adjuster::Actual {}, r))
            }
            Frequency::Zero {} => Ok(ndt(9999, 1, 1)),
        }
    }

    fn try_previous(&self, ueffective: &NaiveDateTime) -> Result<NaiveDateTime, PyErr> {
        let _ = self.try_udate(ueffective)?;
        let cal = Cal::new(vec![], vec![]);
        match self {
            Frequency::BusDays {
                number: n,
                calendar: c,
            } => c.add_bus_days(ueffective, -(*n as i32), false),
            Frequency::CalDays { number: n } => {
                Ok(cal.add_cal_days(ueffective, -(*n as i32), &Adjuster::Actual {}))
            }
            Frequency::Weeks { number: n } => {
                Ok(cal.add_cal_days(ueffective, -(*n as i32 * 7), &Adjuster::Actual {}))
            }
            Frequency::Months { number: n, roll: r } => {
                Ok(cal.add_months(ueffective, -(*n as i32), &Adjuster::Actual {}, r))
            }
            Frequency::Zero {} => Ok(ndt(1500, 1, 1)),
        }
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::ndt;

    #[test]
    fn test_get_next() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 8, 30),
            ),
            (
                Frequency::Months {
                    number: 2,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 9, 30),
            ),
            (
                Frequency::Months {
                    number: 3,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 30),
            ),
            (
                Frequency::Months {
                    number: 4,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 11, 30),
            ),
            (
                Frequency::Months {
                    number: 6,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2023, 1, 30),
            ),
            (
                Frequency::Months {
                    number: 12,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2023, 7, 30),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::EoM {},
                },
                ndt(2022, 6, 30),
                ndt(2022, 7, 31),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::IMM {},
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
                Frequency::Weeks { number: 2 },
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
            assert_eq!(option.2, option.0.try_next(&option.1).unwrap());
            assert_eq!(option.1, option.0.try_previous(&option.2).unwrap());
        }
    }

    #[test]
    fn test_get_uschedule_imm() {
        // test the example given in Coding Interest Rates
        let result = Frequency::Months {
            number: 1,
            roll: RollDay::IMM {},
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
            roll: RollDay::SoM {},
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
    fn test_infer_front() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool, NaiveDateTime)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                true,
                ndt(2022, 8, 15),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                false,
                ndt(2022, 9, 15),
            ),
        ];

        for option in options.iter() {
            assert_eq!(
                option.4,
                option
                    .0
                    .try_infer_front_stub(&option.1, &option.2, option.3)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_infer_front_err() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 8, 15),
                true,
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
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
                .try_infer_front_stub(&option.1, &option.2, option.3)
                .is_err();
            assert_eq!(true, result);
        }
    }

    #[test]
    fn test_infer_back() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool, NaiveDateTime)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                true,
                ndt(2022, 9, 30),
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 10, 15),
                false,
                ndt(2022, 8, 30),
            ),
        ];

        for option in options.iter() {
            assert_eq!(
                option.4,
                option
                    .0
                    .try_infer_back_stub(&option.1, &option.2, option.3)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_infer_back_err() {
        let options: Vec<(Frequency, NaiveDateTime, NaiveDateTime, bool)> = vec![
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 8, 15),
                true,
            ),
            (
                Frequency::Months {
                    number: 1,
                    roll: RollDay::Unspecified {},
                },
                ndt(2022, 7, 30),
                ndt(2022, 9, 15),
                false,
            ),
        ];

        for option in options.iter() {
            let result = option
                .0
                .try_infer_back_stub(&option.1, &option.2, option.3)
                .is_err();
            assert_eq!(true, result);
        }
    }
}
