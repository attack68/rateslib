use crate::scheduling::{
    get_unadjusteds, Adjuster, Adjustment, Calendar, Frequency, RollDay, Scheduling,
};
use chrono::prelude::*;
use itertools::{iproduct, Itertools};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};

/// A stub type indicator for date inference on one side of the schedule.
#[pyclass(module = "rateslib.rs", eq, eq_int)]
#[derive(Copy, Clone, PartialEq)]
pub enum StubInference {
    /// Short front stub inference.
    ShortFront,
    /// Long front stub inference.
    LongFront,
    /// Short back stub inference.
    ShortBack,
    /// Long back stub inference.
    LongBack,
}

/// A generic financial schedule with regular contiguous periods and, possibly, stubs.
///
/// # Notes
/// - A **regular** schedule has a [Frequency] that perfectly divides its ``ueffective`` and
///   ``utermination`` dates, and has no stub dates.
/// - An **irregular** schedule has a ``ufront_stub`` and/or ``uback_stub`` dates defining periods
///   at the boundary of the schedule which are not a standard length of time defined by the
///   [Frequency]. However, a regular schedule must exist between those interior dates.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    /// The unadjusted start date of the schedule.
    pub ueffective: NaiveDateTime,
    /// The unadjusted end date of the schedule.
    pub utermination: NaiveDateTime,
    /// The scheduling [Frequency] for regular periods.
    pub frequency: Frequency,
    /// The optional, unadjusted front stub date.
    pub ufront_stub: Option<NaiveDateTime>,
    /// The optional, unadjusted back stub date.
    pub uback_stub: Option<NaiveDateTime>,
    /// The [Calendar] for accrual and payment date adjustment.
    pub calendar: Calendar,
    /// The [Adjuster] to adjust the unadjusted schedule dates to adjusted period accrual dates.
    pub accrual_adjuster: Adjuster,
    /// The [Adjuster] to adjust the adjusted schedule dates to period payment dates.
    pub payment_adjuster: Adjuster,

    /// The vector of unadjusted period accrual dates.
    pub uschedule: Vec<NaiveDateTime>,
    /// The vector of adjusted period accrual dates.
    pub aschedule: Vec<NaiveDateTime>,
    /// The vector of payment dates associated with the adjusted accrual dates.
    pub pschedule: Vec<NaiveDateTime>,
}

impl Schedule {
    /// Create a [Schedule] from defined unadjusted dates and a [Frequency].
    ///
    /// # Notes
    ///
    /// An unadjusted regular schedule, that aligns with [Frequency], must be defined between
    /// the relevant dates. If not an error is returned. The [Frequency] must be well defined -
    /// optional parameters, such as a missing [RollDay] will result in errors.
    ///
    /// This method uses [Scheduling::try_uregular](crate::scheduling::Scheduling::try_uregular)
    /// to ascertain if the provided dates define a regular schedule or not.
    fn try_new_uschedule_defined(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
    ) -> Result<Self, PyErr> {
        // evaluate the Options and get the start and end of regular schedule component
        let (regular_start, regular_end) =
            match_interior_dates(&ueffective, &ufront_stub, &uback_stub, &utermination);

        // test for single period
        let test_end = frequency.try_unext(&regular_start)?;
        let uregular: Vec<NaiveDateTime>;
        if test_end > regular_end && (ufront_stub.is_none() || uback_stub.is_none()) {
            // then dont have two stub periods and the regular component is a single short period
            uregular = vec![regular_start, regular_end];
        } else {
            // test if the determined regular period is actually a regular period under Frequency
            uregular = frequency.try_uregular(&regular_start, &regular_end)?;
        }

        let uschedule = composite_uschedule(
            &ueffective,
            &utermination,
            &ufront_stub,
            &uback_stub,
            &uregular,
        );
        let aschedule: Vec<NaiveDateTime> = uschedule
            .iter()
            .map(|dt| accrual_adjuster.adjust(&dt, &calendar))
            .collect();
        let pschedule = aschedule
            .iter()
            .map(|dt| payment_adjuster.adjust(&dt, &calendar))
            .collect();

        // eliminate dead stubs - those whose effective or termination, when adjusted,
        // give the same date as the adjusted front stub or back stub
        if !aschedule.iter().all_unique() {
            return Err(PyValueError::new_err("Unadjusted dates provided to the schedule overlap or adjust to the same values.\nUsually this a result of badly specified short stub dates."));
        }

        Ok(Self {
            ueffective,
            utermination,
            frequency,
            ufront_stub,
            uback_stub,
            calendar: calendar.clone(),
            accrual_adjuster,
            payment_adjuster,
            uschedule,
            aschedule,
            pschedule,
        })
    }

    /// Create a [Schedule] from unadjusted dates.
    ///
    /// # Notes
    ///
    /// An unadjusted regular schedule, that aligns with [Frequency], must be defined between
    /// the relevant dates. If not an error is returned.
    ///
    /// This method uses [Scheduling::try_uregular](crate::scheduling::Scheduling::try_uregular)
    /// to ascertain if the provided dates define a regular schedule or not.
    pub fn try_new_uschedule(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        eom: bool,
    ) -> Result<Self, PyErr> {
        // evaluate the Options and get the start and end of regular schedule component
        let (regular_start, regular_end) =
            match_interior_dates(&ueffective, &ufront_stub, &uback_stub, &utermination);

        // get all possible Frequency variants. this will often only be 1 element
        let frequencies =
            frequency.try_vec_from_intersection(&vec![regular_start], &vec![regular_end])?;

        // find all possible schedules that are valid for frequencies
        let uschedules: Vec<Schedule> = frequencies
            .into_iter()
            .map(|f| {
                Schedule::try_new_uschedule_defined(
                    ueffective,
                    utermination,
                    f,
                    ufront_stub,
                    uback_stub,
                    calendar.clone(),
                    accrual_adjuster,
                    payment_adjuster,
                )
            })
            .filter_map(|s| s.ok())
            .collect();

        // error if no valid schedules were found
        if uschedules.len() == 0 {
            return Err(PyValueError::new_err(
                "No valid Schedules could be created with given udate combinations and Frequency.",
            ));
        }

        // filter the found schedules. if `eom` then prefer the first schedule with RollDay::Day(31)
        // else prefer the first found schedule.
        let non_eom_s = uschedules[0].clone();
        let s: Schedule = if eom {
            let temp_s: Vec<Schedule> = uschedules
                .into_iter()
                .filter(|s| {
                    matches!(
                        s.frequency,
                        Frequency::Months {
                            number: _,
                            roll: Some(RollDay::Day(31))
                        }
                    )
                })
                .collect();
            if temp_s.len() >= 1 {
                temp_s[0].clone()
            } else {
                non_eom_s
            }
        } else {
            non_eom_s
        };
        Ok(s)
    }

    /// Create an [Schedule] from unadjusted dates with specified [StubInference].
    ///
    /// # Notes
    ///
    /// If ``stub_inference`` is `None` then this method will revert to [Schedule::try_new_uschedule].
    /// If ``stub_inference`` is given but it conflicts with an explicit ``stub`` date given then
    /// an error will be returned.
    /// If ``stub_inference`` is given but a ``stub`` date is not required then a valid [Schedule]
    /// is returned without an inferred stub.
    pub fn try_new_uschedule_inferred(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        eom: bool,
        stub_inference: Option<StubInference>,
    ) -> Result<Self, PyErr> {
        // validate inference is not blocked by user defined values.
        let _ = validate_stub_dates_and_inference(&ufront_stub, &uback_stub, &stub_inference)?;

        let stubs: (Option<NaiveDateTime>, Option<NaiveDateTime>);
        if stub_inference.is_none() {
            stubs = (ufront_stub, uback_stub);
        } else {
            let (interior_start, interior_end) =
                match_interior_dates(&ueffective, &ufront_stub, &uback_stub, &utermination);
            stubs = match stub_inference.unwrap() {
                StubInference::ShortFront => (
                    frequency.try_infer_ufront_stub(&interior_start, &interior_end, true)?,
                    uback_stub,
                ),
                StubInference::LongFront => (
                    frequency.try_infer_ufront_stub(&interior_start, &interior_end, false)?,
                    uback_stub,
                ),
                StubInference::ShortBack => (
                    ufront_stub,
                    frequency.try_infer_uback_stub(&interior_start, &interior_end, true)?,
                ),
                StubInference::LongBack => (
                    ufront_stub,
                    frequency.try_infer_uback_stub(&interior_start, &interior_end, false)?,
                ),
            }
        }
        Self::try_new_uschedule(
            ueffective,
            utermination,
            frequency,
            stubs.0,
            stubs.1,
            calendar,
            accrual_adjuster,
            payment_adjuster,
            eom,
        )
    }

    /// Generate an [Schedule] from possibly adjusted dates, with stub inference.
    ///
    /// # Notes
    ///
    /// An unadjusted regular schedule, that aligns with [Frequency], must be able to be defined
    /// between the relevant dates. If not an error is returned.
    pub fn try_new_schedule_inferred(
        effective: NaiveDateTime,
        termination: NaiveDateTime,
        frequency: Frequency,
        front_stub: Option<NaiveDateTime>,
        back_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        eom: bool,
        stub_inference: Option<StubInference>,
    ) -> Result<Schedule, PyErr> {
        // find all unadjusted combinations. only adjust the boundaries of the regular component.
        let dates: (
            Vec<NaiveDateTime>,
            Vec<Option<NaiveDateTime>>,
            Vec<Option<NaiveDateTime>>,
            Vec<NaiveDateTime>,
        ) = match (front_stub, back_stub) {
            (None, None) => (
                get_unadjusteds(&effective, &accrual_adjuster, &calendar),
                vec![None],
                vec![None],
                get_unadjusteds(&termination, &accrual_adjuster, &calendar),
            ),
            (Some(d), None) => (
                vec![effective],
                get_unadjusteds(&d, &accrual_adjuster, &calendar)
                    .into_iter()
                    .map(Some)
                    .collect(),
                vec![None],
                get_unadjusteds(&termination, &accrual_adjuster, &calendar),
            ),
            (None, Some(d)) => (
                get_unadjusteds(&effective, &accrual_adjuster, &calendar),
                vec![None],
                get_unadjusteds(&d, &accrual_adjuster, &calendar)
                    .into_iter()
                    .map(Some)
                    .collect(),
                vec![termination],
            ),
            (Some(d), Some(d2)) => (
                vec![effective],
                get_unadjusteds(&d, &accrual_adjuster, &calendar)
                    .into_iter()
                    .map(Some)
                    .collect(),
                get_unadjusteds(&d2, &accrual_adjuster, &calendar)
                    .into_iter()
                    .map(Some)
                    .collect(),
                vec![termination],
            ),
        };

        let combinations = iproduct!(dates.0, dates.1, dates.2, dates.3);
        let schedules: Vec<Schedule> = combinations
            .into_iter()
            .map(|(e, fs, bs, t)| {
                Schedule::try_new_uschedule_inferred(
                    e,
                    t,
                    frequency.clone(),
                    fs,
                    bs,
                    calendar.clone(),
                    accrual_adjuster,
                    payment_adjuster,
                    eom,
                    stub_inference,
                )
            })
            .filter_map(|s| s.ok())
            .collect();

        if schedules.len() == 0 {
            Err(PyValueError::new_err(
                "A Schedule could not be generated from the parameter combinations.",
            ))
        } else {
            Ok(schedules[0].clone())
        }
    }
}

fn match_interior_dates(
    ueffective: &NaiveDateTime,
    ufront_stub: &Option<NaiveDateTime>,
    uback_stub: &Option<NaiveDateTime>,
    utermination: &NaiveDateTime,
) -> (NaiveDateTime, NaiveDateTime) {
    match (ufront_stub, uback_stub) {
        (None, None) => (*ueffective, *utermination),
        (Some(v), None) => (*v, *utermination),
        (None, Some(v)) => (*ueffective, *v),
        (Some(v), Some(w)) => (*v, *w),
    }
}

/// Validate provided stubs do not conflict with the required [StubInference]
fn validate_stub_dates_and_inference(
    ufront_stub: &Option<NaiveDateTime>,
    uback_stub: &Option<NaiveDateTime>,
    stub_inference: &Option<StubInference>,
) -> Result<(), PyErr> {
    match (ufront_stub, uback_stub, stub_inference) {
        (Some(_v), Some(_w), Some(_f)) => Err(PyValueError::new_err(
            "Cannot infer stubs if they are explicitly given.",
        )),
        (Some(_v), None, Some(val))
            if matches!(val, StubInference::ShortFront | StubInference::LongFront) =>
        {
            Err(PyValueError::new_err(
                "Cannot infer stubs if they are explicitly given.",
            ))
        }
        (None, Some(_w), Some(val))
            if matches!(val, StubInference::ShortBack | StubInference::LongBack) =>
        {
            Err(PyValueError::new_err(
                "Cannot infer stubs if they are explicitly given.",
            ))
        }
        _ => Ok(()),
    }
}

/// Get unadjusted schedule dates assuming all inputs are correct and pre-validated.
fn composite_uschedule(
    ueffective: &NaiveDateTime,
    utermination: &NaiveDateTime,
    ufront_stub: &Option<NaiveDateTime>,
    uback_stub: &Option<NaiveDateTime>,
    regular_uschedule: &Vec<NaiveDateTime>,
) -> Vec<NaiveDateTime> {
    let mut uschedule: Vec<NaiveDateTime> = vec![];
    match (*ufront_stub, *uback_stub) {
        (None, None) => {
            uschedule.extend(regular_uschedule);
        }
        (Some(_v), None) => {
            uschedule.push(*ueffective);
            uschedule.extend(regular_uschedule);
        }
        (None, Some(_v)) => {
            uschedule.extend(regular_uschedule);
            uschedule.push(*utermination);
        }
        (Some(_v), Some(_w)) => {
            uschedule.push(*ueffective);
            uschedule.extend(regular_uschedule);
            uschedule.push(*utermination);
        }
    }
    uschedule
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt, Cal, RollDay};

    #[test]
    fn test_new_uschedule_defined_err() {
        // test that None RollDay produces errors even for a well defined schedule
        let result = Schedule::try_new_uschedule_defined(
            ndt(2000, 1, 1),
            ndt(2001, 1, 1),
            Frequency::Months {
                number: 6,
                roll: None,
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::Actual {},
            Adjuster::Actual {},
        );
        assert!(result.is_err())
    }

    #[test]
    fn test_new_uschedule_defined_ok() {
        // test that the above test works for valid RollDay
        let result = Schedule::try_new_uschedule_defined(
            ndt(2000, 1, 1),
            ndt(2001, 1, 1),
            Frequency::Months {
                number: 6,
                roll: Some(RollDay::Day(1)),
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::Actual {},
            Adjuster::Actual {},
        );
        assert_eq!(
            result.unwrap(),
            Schedule {
                ueffective: ndt(2000, 1, 1),
                utermination: ndt(2001, 1, 1),
                frequency: Frequency::Months {
                    number: 6,
                    roll: Some(RollDay::Day(1))
                },
                ufront_stub: None,
                uback_stub: None,
                calendar: Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                accrual_adjuster: Adjuster::Actual {},
                payment_adjuster: Adjuster::Actual {},
                uschedule: vec![ndt(2000, 1, 1), ndt(2000, 7, 1), ndt(2001, 1, 1)],
                aschedule: vec![ndt(2000, 1, 1), ndt(2000, 7, 1), ndt(2001, 1, 1)],
                pschedule: vec![ndt(2000, 1, 1), ndt(2000, 7, 1), ndt(2001, 1, 1)],
            }
        )
    }

    #[test]
    fn test_try_new_uschedule_dead_stubs() {
        let s = Schedule::try_new_uschedule_defined(
            ndt(2023, 1, 1),
            ndt(2024, 1, 2),
            Frequency::Months {
                number: 6,
                roll: Some(RollDay::Day(2)),
            },
            Some(ndt(2023, 1, 2)),
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
        );
        assert!(s.is_err()); // 1st Jan is adjusted to 2nd Jan aligning with front stub

        let s = Schedule::try_new_uschedule_defined(
            ndt(2022, 1, 1),
            ndt(2023, 1, 2),
            Frequency::Months {
                number: 6,
                roll: Some(RollDay::Day(1)),
            },
            None,
            Some(ndt(2023, 1, 1)),
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
        );
        assert!(s.is_err()); // 1st Jan is adjusted to 2nd Jan aligning with front stub
    }

    #[test]
    fn test_try_new_uschedule_defined_single_short_period() {
        let s = Schedule::try_new_uschedule_defined(
            ndt(2023, 1, 1),
            ndt(2023, 4, 1),
            Frequency::Months {
                number: 6,
                roll: Some(RollDay::Day(1)),
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
        )
        .unwrap();
        assert_eq!(s.uschedule, vec![ndt(2023, 1, 1), ndt(2023, 4, 1)]);
    }

    #[test]
    fn test_try_new_uschedule_short_period() {
        let s = Schedule::try_new_uschedule(
            ndt(2022, 7, 1),
            ndt(2022, 10, 1),
            Frequency::Months {
                number: 12,
                roll: None,
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
            true,
        )
        .unwrap();
        assert_eq!(s.uschedule, vec![ndt(2022, 7, 1), ndt(2022, 10, 1)]);
    }

    #[test]
    fn test_try_new_uschedule() {
        let s = Schedule::try_new_uschedule(
            ndt(2000, 1, 1),
            ndt(2000, 12, 15),
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(15)),
            },
            Some(ndt(2000, 3, 15)),
            None,
            Calendar::Cal(Cal::new(vec![], vec![])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
            true,
        )
        .unwrap();
        let uschedule = vec![
            ndt(2000, 1, 1),
            ndt(2000, 3, 15),
            ndt(2000, 6, 15),
            ndt(2000, 9, 15),
            ndt(2000, 12, 15),
        ];
        let pschedule = vec![
            ndt(2000, 1, 2),
            ndt(2000, 3, 16),
            ndt(2000, 6, 16),
            ndt(2000, 9, 16),
            ndt(2000, 12, 16),
        ];
        assert_eq!(uschedule, s.uschedule);
        assert_eq!(pschedule, s.pschedule);
    }

    #[test]
    fn test_try_new_uschedule_eom_parameter_selection() {
        let s = Schedule::try_new_uschedule(
            ndt(2024, 2, 29),
            ndt(2024, 11, 30),
            Frequency::Months {
                number: 3,
                roll: None,
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
            true,
        )
        .unwrap();
        assert_eq!(
            s.frequency,
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(31))
            }
        );

        let s = Schedule::try_new_uschedule(
            ndt(2024, 2, 29),
            ndt(2024, 11, 30),
            Frequency::Months {
                number: 3,
                roll: None,
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
            false,
        )
        .unwrap();
        assert_eq!(
            s.frequency,
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(30))
            }
        );

        let s = Schedule::try_new_uschedule(
            ndt(2024, 2, 29),
            ndt(2024, 11, 29),
            Frequency::Months {
                number: 3,
                roll: None,
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
            true,
        )
        .unwrap();
        assert_eq!(
            s.frequency,
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(29))
            }
        );
    }

    #[test]
    fn test_try_new_uschedule_inferred_fails() {
        // fails because stub dates are given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_uschedule_inferred(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                Some(ndt(2000, 1, 10)),
                Some(ndt(2000, 1, 16)),
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle { number: 1 },
                true,
                Some(StubInference::ShortBack)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_uschedule_inferred(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                None,
                Some(ndt(2000, 1, 16)),
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle { number: 1 },
                true,
                Some(StubInference::ShortBack)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_uschedule_inferred(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                None,
                Some(ndt(2000, 1, 16)),
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle { number: 1 },
                true,
                Some(StubInference::LongBack)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_uschedule_inferred(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                Some(ndt(2000, 1, 16)),
                None,
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle { number: 1 },
                true,
                Some(StubInference::ShortFront)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_uschedule_inferred(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                Some(ndt(2000, 1, 16)),
                None,
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle { number: 1 },
                true,
                Some(StubInference::LongFront)
            )
            .is_err()
        );
    }
}
