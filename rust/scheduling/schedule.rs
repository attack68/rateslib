use crate::scheduling::{
    get_unadjusteds, Adjuster, Adjustment, Calendar, Frequency, RollDay, Scheduling,
};
use chrono::prelude::*;
use itertools::iproduct;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};

/// Specifier used by [`Schedule::try_new_inferred`] to instruct its inference logic.
#[pyclass(module = "rateslib.rs", eq, eq_int)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StubInference {
    /// Short front stub inference.
    ShortFront = 0,
    /// Long front stub inference.
    LongFront = 1,
    /// Short back stub inference.
    ShortBack = 2,
    /// Long back stub inference.
    LongBack = 3,
}

/// A generic financial schedule with regular contiguous periods and, possibly, stubs.
///
/// # Notes
/// - A **regular** schedule has a [`Frequency`] that perfectly divides its ``ueffective`` and
///   ``utermination`` dates, and has no stub dates.
/// - An **irregular** schedule has a ``ufront_stub`` and/or ``uback_stub`` dates defining periods
///   at the boundary of the schedule which are not a standard length of time defined by the
///   [`Frequency`]. However, a regular schedule must exist between those interior dates.
#[pyclass(module = "rateslib.rs", eq)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "ScheduleDataModel")]
pub struct Schedule {
    /// The unadjusted start date of the schedule.
    pub ueffective: NaiveDateTime,
    /// The unadjusted end date of the schedule.
    pub utermination: NaiveDateTime,
    /// The scheduling [`Frequency`] for regular periods.
    pub frequency: Frequency,
    /// The optional, unadjusted front stub date.
    pub ufront_stub: Option<NaiveDateTime>,
    /// The optional, unadjusted back stub date.
    pub uback_stub: Option<NaiveDateTime>,
    /// The [`Calendar`] for accrual and payment date adjustment.
    pub calendar: Calendar,
    /// The [`Adjuster`] to adjust the unadjusted schedule dates to adjusted period accrual dates.
    pub accrual_adjuster: Adjuster,
    /// The [`Adjuster`] to adjust the accrual schedule dates to period payment dates.
    pub payment_adjuster: Adjuster,
    /// An additional [`Adjuster`] to adjust the accrual schedule dates to some other period payment or fixing dates.
    ///
    /// This is often used as a notional exchange lag, which for XCS, for example, differs to a regular coupon lag.
    pub payment_adjuster2: Adjuster,
    /// An additional [`Adjuster`] to adjust the accrual schedule dates to some other period payment or fixing dates.
    ///
    /// If *None* is set to match ``payment_adjuster``.
    pub payment_adjuster3: Option<Adjuster>,
    /// The vector of unadjusted period accrual dates.
    #[serde(skip)]
    pub uschedule: Vec<NaiveDateTime>,
    /// The vector of adjusted period accrual dates.
    #[serde(skip)]
    pub aschedule: Vec<NaiveDateTime>,
    /// The vector of payment dates associated with the adjusted accrual dates.
    #[serde(skip)]
    pub pschedule: Vec<NaiveDateTime>,
    /// An additional vector of payment dates associated with the adjusted accrual dates.
    #[serde(skip)]
    pub pschedule2: Vec<NaiveDateTime>,
    /// An additional vector of payment dates associated with the adjusted accrual dates.
    #[serde(skip)]
    pub pschedule3: Vec<NaiveDateTime>,
}

#[derive(Deserialize)]
struct ScheduleDataModel {
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    frequency: Frequency,
    ufront_stub: Option<NaiveDateTime>,
    uback_stub: Option<NaiveDateTime>,
    calendar: Calendar,
    accrual_adjuster: Adjuster,
    payment_adjuster: Adjuster,
    payment_adjuster2: Adjuster,
    payment_adjuster3: Option<Adjuster>,
}

impl std::convert::From<ScheduleDataModel> for Schedule {
    fn from(model: ScheduleDataModel) -> Self {
        Self::try_new_defined(
            model.ueffective,
            model.utermination,
            model.frequency,
            model.ufront_stub,
            model.uback_stub,
            model.calendar,
            model.accrual_adjuster,
            model.payment_adjuster,
            model.payment_adjuster2,
            model.payment_adjuster3,
        )
        .expect("Data model for `Schedule` is corrupt or invalid.")
    }
}

/// Check that right is greater than left if both Some, and that they do not create a 'dead stub'.
fn validate_individual_dates(
    left: &Option<NaiveDateTime>,
    right: &Option<NaiveDateTime>,
    accrual_adjuster: &Adjuster,
    calendar: &Calendar,
) -> Result<(), PyErr> {
    match (left, right) {
        (Some(_left), Some(_right)) => {}
        _ => return Ok(()),
    }
    if left >= right {
        return Err(PyValueError::new_err(
            "Dates are invalid since they are repeated.",
        ));
    }
    if accrual_adjuster.adjust(&left.unwrap(), calendar)
        >= accrual_adjuster.adjust(&right.unwrap(), calendar)
    {
        return Err(PyValueError::new_err(
            "Dates define dead stubs and are invalid",
        ));
    }
    Ok(())
}

/// Ensure dates are ordered and that they do not define 'dead stubs', which are created when
/// two scheduling dates are adjusted under some [Adjuster] and result in the same date.
fn validate_date_ordering(
    ueffective: &NaiveDateTime,
    ufront_stub: &Option<NaiveDateTime>,
    uback_stub: &Option<NaiveDateTime>,
    utermination: &NaiveDateTime,
    accrual_adjuster: &Adjuster,
    calendar: &Calendar,
) -> Result<(), PyErr> {
    let _ = validate_individual_dates(&Some(*ueffective), ufront_stub, accrual_adjuster, calendar)?;
    let _ = validate_individual_dates(&Some(*ueffective), uback_stub, accrual_adjuster, calendar)?;
    let _ = validate_individual_dates(
        &Some(*ueffective),
        &Some(*utermination),
        accrual_adjuster,
        calendar,
    )?;
    // front and back stub dates can be equal if the schedule is defined only by two stubs
    // let _ = validate_individual_dates(ufront_stub, uback_stub, accrual_adjuster, calendar)?;
    let _ = validate_individual_dates(
        ufront_stub,
        &Some(*utermination),
        accrual_adjuster,
        calendar,
    )?;
    let _ =
        validate_individual_dates(uback_stub, &Some(*utermination), accrual_adjuster, calendar)?;
    Ok(())
}

/// Ensure that two dates can define a proper stub period, either short or long, front or back.
fn validate_is_stub(
    left: &NaiveDateTime,
    right: &NaiveDateTime,
    frequency: &Frequency,
    front: bool,
) -> Result<(), PyErr> {
    if front {
        if frequency.is_front_stub(left, right) {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Dates intended to define a front stub do not permit a valid stub period.",
            ))
        }
    } else {
        if frequency.is_back_stub(left, right) {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Dates intended to define a back stub do not permit a valid stub period.",
            ))
        }
    }
}

impl Schedule {
    /// Create a [`Schedule`] from well defined unadjusted dates and a [`Frequency`].
    ///
    /// # Notes
    /// If provided arguments do not define a valid schedule pattern then an error is returned.
    ///
    /// # Examples
    /// This is a valid schedule with a long back stub and regular monthly periods.
    /// ```rust
    /// # use rateslib::scheduling::{Schedule, ndt, Frequency, Adjuster, Calendar, Cal, RollDay};
    /// let s = Schedule::try_new_defined(
    ///     ndt(2024, 1, 3), ndt(2024, 4, 15),                  // ueffective, utermination
    ///     Frequency::Months{number:1, roll: Some(RollDay::Day(3))}, // frequency
    ///     None, Some(ndt(2024, 3, 3)),                        // ufront_stub, uback_stub
    ///     Cal::new(vec![], vec![5,6]).into(),                 // calendar
    ///     Adjuster::ModifiedFollowing{},                      // accrual_adjuster
    ///     Adjuster::BusDaysLagSettle(3),                      // payment_adjuster
    ///     Adjuster::Actual{},                                 // payment_adjuster2
    ///     None,                                               // payment_adjuster3
    /// );
    /// # let s = s.unwrap();
    /// assert_eq!(s.uschedule, vec![ndt(2024, 1, 3), ndt(2024, 2, 3), ndt(2024, 3, 3), ndt(2024, 4, 15)]);
    /// assert_eq!(s.aschedule, vec![ndt(2024, 1, 3), ndt(2024, 2, 5), ndt(2024, 3, 4), ndt(2024, 4, 15)]);
    /// assert_eq!(s.pschedule, vec![ndt(2024, 1, 8), ndt(2024, 2, 8), ndt(2024, 3, 7), ndt(2024, 4, 18)]);
    /// ```
    /// This is not a valid schedule since there are no defined stubs and the dates do not align
    /// with the [RollDay].
    /// ```rust
    /// # use rateslib::scheduling::{Schedule, ndt, Frequency, Adjuster, Calendar, Cal, RollDay};
    /// let s = Schedule::try_new_defined(
    ///     ndt(2024, 1, 6), ndt(2024, 4, 6),                  // ueffective, utermination
    ///     Frequency::Months{number:1, roll: Some(RollDay::Day(3))}, // frequency
    ///     None, None,                                         // ufront_stub, uback_stub
    ///     Cal::new(vec![], vec![5,6]).into(),                 // calendar
    ///     Adjuster::ModifiedFollowing{},                      // accrual_adjuster
    ///     Adjuster::BusDaysLagSettle(3),                      // payment_adjuster
    ///     Adjuster::Actual{},                                 // payment_adjuster2
    ///     None,                                               // payment_adjuster3
    /// );
    /// assert!(s.is_err());
    /// ```
    pub fn try_new_defined(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        payment_adjuster2: Adjuster,
        payment_adjuster3: Option<Adjuster>,
    ) -> Result<Self, PyErr> {
        // validate date ordering
        let _ = validate_date_ordering(
            &ueffective,
            &ufront_stub,
            &uback_stub,
            &utermination,
            &accrual_adjuster,
            &calendar,
        )?;

        let uschedule: Vec<NaiveDateTime>;

        match (ufront_stub, uback_stub) {
            (None, None) => {
                // then schedule is defined only by ueffective and utermination
                let uregular = frequency.try_uregular(&ueffective, &utermination);
                if uregular.is_ok() {
                    // case 1) schedule must be a regular schedule
                    uschedule = uregular.unwrap();
                } else if frequency.is_front_stub(&ueffective, &utermination)
                    || frequency.is_back_stub(&ueffective, &utermination)
                {
                    //case 2) schedule must be a single period stub
                    uschedule = vec![ueffective, utermination];
                } else {
                    return Err(PyValueError::new_err("`ueffective`, `utermination` and `frequency` do not define a regular schedule or a single period stub."));
                }
            }
            (Some(regular_start), None) => {
                // case 3) with a front stub
                let uregular = frequency.try_uregular(&regular_start, &utermination)?;
                let _ = validate_is_stub(&ueffective, &regular_start, &frequency, true)?;
                uschedule = composite_uschedule(
                    &ueffective,
                    &utermination,
                    &ufront_stub,
                    &uback_stub,
                    &uregular,
                );
            }
            (None, Some(regular_end)) => {
                // case 3) with a back stub
                let uregular = frequency.try_uregular(&ueffective, &regular_end)?;
                let _ = validate_is_stub(&regular_end, &utermination, &frequency, false)?;
                uschedule = composite_uschedule(
                    &ueffective,
                    &utermination,
                    &ufront_stub,
                    &uback_stub,
                    &uregular,
                );
            }
            (Some(regular_start), Some(regular_end)) => {
                let _ = validate_is_stub(&ueffective, &regular_start, &frequency, true)?;
                let _ = validate_is_stub(&regular_end, &utermination, &frequency, false)?;
                if regular_start == regular_end {
                    // is only possible when stubs are both given and are equal, due to date validation
                    // case 4) schedule must be two stubs
                    uschedule = vec![ueffective, regular_start, utermination];
                } else {
                    // case 5) some regular component with stubs at both ends
                    let uregular = frequency.try_uregular(&regular_start, &regular_end)?;
                    uschedule = composite_uschedule(
                        &ueffective,
                        &utermination,
                        &ufront_stub,
                        &uback_stub,
                        &uregular,
                    );
                }
            }
        }

        let aschedule: Vec<NaiveDateTime> = accrual_adjuster.adjusts(&uschedule, &calendar);
        let pschedule = payment_adjuster.adjusts(&aschedule, &calendar);
        let pschedule2 = payment_adjuster2.adjusts(&aschedule, &calendar);
        let pschedule3 = match payment_adjuster3 {
            None => pschedule.clone(),
            Some(adjuster) => adjuster.adjusts(&aschedule, &calendar),
        };

        Ok(Self {
            ueffective,
            utermination,
            frequency,
            ufront_stub,
            uback_stub,
            calendar: calendar.clone(),
            accrual_adjuster,
            payment_adjuster,
            payment_adjuster2,
            payment_adjuster3,
            uschedule,
            aschedule,
            pschedule,
            pschedule2,
            pschedule3,
        })
    }

    /// Create a [`Schedule`] from unadjusted dates with specified [`StubInference`].
    ///
    /// # Notes
    /// This method introduces the ``stub_inference`` argument.
    /// If it is given as `None` then this method will revert to [Schedule::try_new_uschedule].
    /// If ``stub_inference`` is given but it conflicts with an explicit ``stub`` date given then
    /// an error will be returned.
    /// If ``stub_inference`` is given but a ``stub`` date is not required then a valid [Schedule]
    /// is returned without an inferred stub.
    fn try_new_infer_stub(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        payment_adjuster2: Adjuster,
        payment_adjuster3: Option<Adjuster>,
        stub_inference: Option<StubInference>,
    ) -> Result<Self, PyErr> {
        // evaluate if schedule is valid as defined without stub inference
        let temp_schedule = Schedule::try_new_defined(
            ueffective,
            utermination,
            frequency.clone(),
            ufront_stub,
            uback_stub,
            calendar.clone(),
            accrual_adjuster,
            payment_adjuster,
            payment_adjuster2,
            payment_adjuster3,
        );

        // validate inference is not blocked by user defined values.
        let _ = validate_stub_dates_and_inference(&ufront_stub, &uback_stub, &stub_inference)?;

        let stubs: (Option<NaiveDateTime>, Option<NaiveDateTime>);
        if stub_inference.is_none() {
            return temp_schedule;
        } else {
            let (interior_start, interior_end) =
                match_interior_dates(&ueffective, &ufront_stub, &uback_stub, &utermination);
            stubs = match stub_inference.unwrap() {
                StubInference::ShortFront => {
                    if temp_schedule.is_ok() {
                        let test_schedule = temp_schedule.unwrap();
                        if frequency.is_short_front_stub(
                            &test_schedule.uschedule[0],
                            &test_schedule.uschedule[1],
                        ) {
                            return Ok(test_schedule);
                        } // already has a short front stub
                    }
                    (
                        frequency.try_infer_ufront_stub(&interior_start, &interior_end, true)?,
                        uback_stub,
                    )
                }
                StubInference::LongFront => {
                    if temp_schedule.is_ok() {
                        let test_schedule = temp_schedule.unwrap();
                        if frequency.is_long_front_stub(
                            &test_schedule.uschedule[0],
                            &test_schedule.uschedule[1],
                        ) {
                            return Ok(test_schedule);
                        } // already has a long front stub
                    }
                    (
                        frequency.try_infer_ufront_stub(&interior_start, &interior_end, false)?,
                        uback_stub,
                    )
                }
                StubInference::ShortBack => {
                    if temp_schedule.is_ok() {
                        let test_schedule = temp_schedule.unwrap();
                        let n = test_schedule.uschedule.len();
                        if frequency.is_short_back_stub(
                            &test_schedule.uschedule[n - 1],
                            &test_schedule.uschedule[n - 2],
                        ) {
                            return Ok(test_schedule);
                        } // already has a short back stub
                    }
                    (
                        ufront_stub,
                        frequency.try_infer_uback_stub(&interior_start, &interior_end, true)?,
                    )
                }
                StubInference::LongBack => {
                    if temp_schedule.is_ok() {
                        let test_schedule = temp_schedule.unwrap();
                        let n = test_schedule.uschedule.len();
                        if frequency.is_short_back_stub(
                            &test_schedule.uschedule[n - 1],
                            &test_schedule.uschedule[n - 2],
                        ) {
                            return Ok(test_schedule);
                        } // already has a long back stub
                    }
                    (
                        ufront_stub,
                        frequency.try_infer_uback_stub(&interior_start, &interior_end, false)?,
                    )
                }
            }
        }
        Self::try_new_defined(
            ueffective,
            utermination,
            frequency,
            stubs.0,
            stubs.1,
            calendar,
            accrual_adjuster,
            payment_adjuster,
            payment_adjuster2,
            payment_adjuster3,
        )
    }

    /// Create a [`Schedule`] from unadjusted dates.
    ///
    /// # Notes
    ///
    /// An unadjusted regular schedule, that aligns with [Frequency], must be defined between
    /// the relevant dates. If not an error is returned.
    ///
    /// This method uses [Scheduling::try_uregular](crate::scheduling::Scheduling::try_uregular)
    /// to ascertain if the provided dates define a regular schedule or not.
    fn try_new_uschedule_infer_frequency(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        payment_adjuster2: Adjuster,
        payment_adjuster3: Option<Adjuster>,
        eom: bool,
        stub_inference: Option<StubInference>,
    ) -> Result<Self, PyErr> {
        // evaluate the Options and get the start and end of regular schedule component
        let (regular_start, regular_end) =
            match_interior_dates(&ueffective, &ufront_stub, &uback_stub, &utermination);

        // get all possible Frequency variants. this will often only be 1 element
        let frequencies = frequency.try_vec_from(&vec![regular_start, regular_end])?;

        // find all possible schedules that are valid for frequencies
        let uschedules: Vec<Schedule> = frequencies
            .into_iter()
            .filter_map(|f| {
                Schedule::try_new_infer_stub(
                    ueffective,
                    utermination,
                    f,
                    ufront_stub,
                    uback_stub,
                    calendar.clone(),
                    accrual_adjuster,
                    payment_adjuster,
                    payment_adjuster2,
                    payment_adjuster3,
                    stub_inference,
                )
                .ok()
            })
            .collect();

        // error if no valid schedules were found
        if uschedules.len() == 0 {
            return Err(PyValueError::new_err(
                "No valid Schedules could be created with given `udates` combinations and `frequency`.",
            ));
        }

        // filter regular schedules
        let regulars: Vec<Schedule> = uschedules
            .iter()
            .cloned()
            .filter(|schedule| schedule.is_regular())
            .collect();
        if regulars.len() != 0 {
            Ok(filter_schedules_by_eom(regulars, eom))
        } else {
            Ok(filter_schedules_by_eom(uschedules, eom))
        }
    }

    /// Create a [`Schedule`] using inference if some of the parameters are not well defined.
    ///
    /// # Notes
    /// If all parameters are well defined and dates are definitively known in their unadjusted
    /// forms then the [`try_new_defined`](Schedule::try_new_defined) method
    /// should be used instead.
    ///
    /// This method provides the additional features below:
    /// - **Unadjusted date inference**: if *adjusted* dates are given then a neighbourhood of
    ///   dates will be sub-sampled to determine
    ///   any possibilities for *unadjusted* dates defined by the `accrual_adjuster` and `calendar`.
    ///   Only the dates at either side of the regular schedule component are explored. Stub date
    ///   boundaries are used as provided.
    /// - **Frequency inference**: any [`Frequency`](crate::scheduling::Frequency) that contains
    ///   optional elements, e.g. no [`RollDay`],
    ///   will be explored for all possible alternatives that results in the most likely schedule,
    ///   guided by the `eom` parameter.
    /// - **Stub date inference**: one-sided stub date inference can be attempted guided by
    ///   the `stub_inference` parameter.
    pub fn try_new_inferred(
        effective: NaiveDateTime,
        termination: NaiveDateTime,
        frequency: Frequency,
        front_stub: Option<NaiveDateTime>,
        back_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        payment_adjuster2: Adjuster,
        payment_adjuster3: Option<Adjuster>,
        eom: bool,
        stub_inference: Option<StubInference>,
    ) -> Result<Schedule, PyErr> {
        // perform a preliminary check to determine if a given stub date actually falls under some
        // regular schedule. This is common when a list of bonds have 'first coupon' dates that
        // may or may not be official stub dates.
        if front_stub.is_none() && back_stub.is_none() {
            // then do nothing in this pre-check
        } else {
            let dates: (Vec<NaiveDateTime>, Vec<NaiveDateTime>) = (
                get_unadjusteds(&effective, &accrual_adjuster, &calendar),
                get_unadjusteds(&termination, &accrual_adjuster, &calendar),
            );
            let combinations = iproduct!(dates.0, dates.1);
            let schedules: Vec<Schedule> = combinations
                .into_iter()
                .filter_map(|(e, t)| {
                    Schedule::try_new_uschedule_infer_frequency(
                        e,
                        t,
                        frequency.clone(),
                        None,
                        None,
                        calendar.clone(),
                        accrual_adjuster,
                        payment_adjuster,
                        payment_adjuster2,
                        payment_adjuster3,
                        eom,
                        stub_inference,
                    )
                    .ok()
                })
                .filter(|schedule| schedule.is_regular())
                .filter(|s| {
                    front_stub.is_none()
                        || (front_stub.is_some()
                            && (front_stub.unwrap() == s.aschedule[1]
                                || front_stub.unwrap() == s.uschedule[1]))
                })
                .filter(|s| {
                    back_stub.is_none()
                        || (back_stub.is_some()
                            && (back_stub.unwrap() == s.aschedule[s.aschedule.len() - 2]
                                || back_stub.unwrap() == s.uschedule[s.uschedule.len() - 2]))
                })
                .collect();
            if schedules.len() == 0 {
                // do nothing because the pre-check has failed: moved to usual construction
            } else {
                // filter regular schedules
                return Ok(filter_schedules_by_eom(schedules, eom));
            }
        }

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
            .filter_map(|(e, fs, bs, t)| {
                Schedule::try_new_uschedule_infer_frequency(
                    e,
                    t,
                    frequency.clone(),
                    fs,
                    bs,
                    calendar.clone(),
                    accrual_adjuster,
                    payment_adjuster,
                    payment_adjuster2,
                    payment_adjuster3,
                    eom,
                    stub_inference,
                )
                .ok()
            })
            .collect();

        if schedules.len() == 0 {
            Err(PyValueError::new_err(
                "A Schedule could not be generated from the parameter combinations.",
            ))
        } else {
            // filter regular schedules
            let regulars: Vec<Schedule> = schedules
                .iter()
                .cloned()
                .filter(|schedule| schedule.is_regular())
                .collect();
            if regulars.len() != 0 {
                Ok(filter_schedules_by_eom(regulars, eom))
            } else {
                Ok(filter_schedules_by_eom(schedules, eom))
            }
        }
    }

    /// Check if a [`Schedule`] contains only regular periods, and no stub periods.
    pub fn is_regular(&self) -> bool {
        let ucheck = self
            .frequency
            .try_uregular(&self.ueffective, &self.utermination);
        if ucheck.is_ok() {
            ucheck.unwrap() == self.uschedule
        } else {
            false
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

fn filter_schedules_by_eom(uschedules: Vec<Schedule>, eom: bool) -> Schedule {
    // filter the found schedules. if `eom` then prefer the first schedule with RollDay::Day(31)
    // else prefer the first found schedule.
    let original = uschedules[0].clone();

    if !eom {
        // just return the first schedule
        original
    } else {
        // scan for an eom possibility
        let possibles: Vec<Schedule> = uschedules
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
        if possibles.len() >= 1 {
            possibles[0].clone()
        } else {
            original
        }
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt, Cal, RollDay};

    #[test]
    fn test_new_uschedule_defined_cases_1_and_2() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, Vec<NaiveDateTime>)> = vec![
            (
                ndt(2000, 1, 1), // regular schedule
                ndt(2000, 3, 1),
                vec![ndt(2000, 1, 1), ndt(2000, 2, 1), ndt(2000, 3, 1)],
            ),
            (
                ndt(2000, 1, 1), // short single period sub
                ndt(2000, 1, 20),
                vec![ndt(2000, 1, 1), ndt(2000, 1, 20)],
            ),
            (
                ndt(2000, 1, 1), // long single period stub
                ndt(2000, 2, 15),
                vec![ndt(2000, 1, 1), ndt(2000, 2, 15)],
            ),
        ];
        for option in options {
            let result = Schedule::try_new_defined(
                option.0,
                option.1,
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(1)),
                },
                None,
                None,
                Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                Adjuster::Following {},
                Adjuster::Following {},
                Adjuster::Following {},
                None,
            );
            assert_eq!(result.unwrap().uschedule, option.2);
        }
    }

    #[test]
    fn test_new_uschedule_defined_cases_1_and_2_err() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, Frequency)> = vec![
            (
                ndt(2000, 1, 1), // regular schedule is not defined stub too long
                ndt(2000, 3, 15),
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(1)),
                },
            ),
            (
                ndt(2000, 1, 1), // undefined RollDay
                ndt(2000, 3, 1),
                Frequency::Months {
                    number: 1,
                    roll: None,
                },
            ),
        ];
        for option in options {
            let result = Schedule::try_new_defined(
                option.0,
                option.1,
                option.2,
                None,
                None,
                Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                Adjuster::Following {},
                Adjuster::Following {},
                Adjuster::Following {},
                None,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_new_uschedule_defined_cases_4() {
        let options: Vec<(
            NaiveDateTime,
            NaiveDateTime,
            NaiveDateTime,
            Vec<NaiveDateTime>,
        )> = vec![
            (
                ndt(2000, 1, 1), // Short then Short
                ndt(2000, 1, 15),
                ndt(2000, 2, 10),
                vec![ndt(2000, 1, 1), ndt(2000, 1, 15), ndt(2000, 2, 10)],
            ),
            (
                ndt(2000, 1, 1), // Short then Long
                ndt(2000, 1, 15),
                ndt(2000, 2, 25),
                vec![ndt(2000, 1, 1), ndt(2000, 1, 15), ndt(2000, 2, 25)],
            ),
            (
                ndt(2000, 1, 1), // Long then Short
                ndt(2000, 2, 15),
                ndt(2000, 2, 25),
                vec![ndt(2000, 1, 1), ndt(2000, 2, 15), ndt(2000, 2, 25)],
            ),
            (
                ndt(2000, 1, 1), // Long then Long
                ndt(2000, 2, 15),
                ndt(2000, 3, 20),
                vec![ndt(2000, 1, 1), ndt(2000, 2, 15), ndt(2000, 3, 20)],
            ),
        ];
        for option in options {
            let result = Schedule::try_new_defined(
                option.0,
                option.2,
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(15)),
                }, // Zero also works, as does CalDays(30)
                Some(option.1),
                Some(option.1),
                Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                Adjuster::Following {},
                Adjuster::Following {},
                Adjuster::Following {},
                None,
            );
            assert_eq!(result.unwrap().uschedule, option.3);
        }
    }

    #[test]
    fn test_new_uschedule_defined_cases_3() {
        let options: Vec<(
            NaiveDateTime,
            Option<NaiveDateTime>,
            Option<NaiveDateTime>,
            NaiveDateTime,
            Vec<NaiveDateTime>,
        )> = vec![
            (
                ndt(2000, 1, 1), // Short then Regular
                Some(ndt(2000, 1, 15)),
                None,
                ndt(2000, 3, 15),
                vec![
                    ndt(2000, 1, 1),
                    ndt(2000, 1, 15),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                ],
            ),
            (
                ndt(2000, 1, 1), // Long then Regular
                Some(ndt(2000, 2, 15)),
                None,
                ndt(2000, 4, 15),
                vec![
                    ndt(2000, 1, 1),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 4, 15),
                ],
            ),
            (
                ndt(2000, 1, 15), // Regular then Short
                None,
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 10),
                vec![
                    ndt(2000, 1, 15),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 4, 10),
                ],
            ),
            (
                ndt(2000, 1, 15), // Regular then Long
                None,
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 25),
                vec![
                    ndt(2000, 1, 15),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 4, 25),
                ],
            ),
            (
                ndt(2000, 1, 15), // Regular then 2 -period Long
                None,
                Some(ndt(2000, 3, 15)),
                ndt(2000, 5, 15),
                vec![
                    ndt(2000, 1, 15),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 5, 15),
                ],
            ),
        ];
        for option in options {
            let result = Schedule::try_new_defined(
                option.0,
                option.3,
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(15)),
                }, // Zero also works
                option.1,
                option.2,
                Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                Adjuster::Following {},
                Adjuster::Following {},
                Adjuster::Following {},
                None,
            );
            assert_eq!(result.unwrap().uschedule, option.4);
        }
    }

    #[test]
    fn test_new_uschedule_defined_cases_3_err() {
        let options: Vec<(
            NaiveDateTime,
            Option<NaiveDateTime>,
            Option<NaiveDateTime>,
            NaiveDateTime,
        )> = vec![
            (
                ndt(2000, 1, 1), // Short then Regular misaligned
                Some(ndt(2000, 1, 15)),
                None,
                ndt(2000, 3, 16),
            ),
            (
                ndt(2000, 1, 1), // Front Stub is too long
                Some(ndt(2000, 5, 15)),
                None,
                ndt(2000, 7, 15),
            ),
            (
                ndt(2000, 1, 13), // Regular misaligned then Short
                None,
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 10),
            ),
            (
                ndt(2000, 1, 15), // Back Stub is too long
                None,
                Some(ndt(2000, 3, 15)),
                ndt(2000, 7, 25),
            ),
            (
                ndt(2000, 1, 15), // Short stub cannot be a regular period
                None,
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 15),
            ),
        ];
        for option in options {
            let result = Schedule::try_new_defined(
                option.0,
                option.3,
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(15)),
                }, // Zero also works
                option.1,
                option.2,
                Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                Adjuster::Following {},
                Adjuster::Following {},
                Adjuster::Following {},
                None,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_new_uschedule_defined_cases_5() {
        let options: Vec<(
            NaiveDateTime,
            Option<NaiveDateTime>,
            Option<NaiveDateTime>,
            NaiveDateTime,
            Vec<NaiveDateTime>,
        )> = vec![
            (
                ndt(2000, 1, 1), // Short Short
                Some(ndt(2000, 1, 15)),
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 10),
                vec![
                    ndt(2000, 1, 1),
                    ndt(2000, 1, 15),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 4, 10),
                ],
            ),
            (
                ndt(2000, 1, 1), // Short Long
                Some(ndt(2000, 1, 15)),
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 25),
                vec![
                    ndt(2000, 1, 1),
                    ndt(2000, 1, 15),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 4, 25),
                ],
            ),
            (
                ndt(2000, 1, 1), // Long Long
                Some(ndt(2000, 2, 15)),
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 25),
                vec![
                    ndt(2000, 1, 1),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 4, 25),
                ],
            ),
            (
                ndt(2000, 1, 1), // Long Short
                Some(ndt(2000, 2, 15)),
                Some(ndt(2000, 3, 15)),
                ndt(2000, 4, 10),
                vec![
                    ndt(2000, 1, 1),
                    ndt(2000, 2, 15),
                    ndt(2000, 3, 15),
                    ndt(2000, 4, 10),
                ],
            ),
        ];
        for option in options {
            let result = Schedule::try_new_defined(
                option.0,
                option.3,
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(15)),
                }, // Zero also works
                option.1,
                option.2,
                Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                Adjuster::Following {},
                Adjuster::Following {},
                Adjuster::Following {},
                None,
            );
            assert_eq!(result.unwrap().uschedule, option.4);
        }
    }

    #[test]
    fn test_new_uschedule_defined_cases_5_err() {
        let options: Vec<(
            NaiveDateTime,
            Option<NaiveDateTime>,
            Option<NaiveDateTime>,
            NaiveDateTime,
        )> = vec![(
            ndt(2000, 1, 1), // Regular is misaligned
            Some(ndt(2000, 1, 15)),
            Some(ndt(2000, 3, 16)),
            ndt(2000, 4, 10),
        )];
        for option in options {
            let result = Schedule::try_new_defined(
                option.0,
                option.3,
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(15)),
                }, // Zero also works
                option.1,
                option.2,
                Calendar::Cal(Cal::new(vec![], vec![5, 6])),
                Adjuster::Following {},
                Adjuster::Following {},
                Adjuster::Following {},
                None,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_new_uschedule_defined_err() {
        // test that None RollDay produces errors even for a well defined schedule
        let result = Schedule::try_new_defined(
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
            Adjuster::Following {},
            None,
        );
        assert!(result.is_err())
    }

    #[test]
    fn test_try_new_uschedule_dead_stubs() {
        let s = Schedule::try_new_defined(
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
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
        );
        assert!(s.is_err()); // 1st Jan is adjusted to 2nd Jan aligning with front stub

        let s = Schedule::try_new_defined(
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
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
        );
        assert!(s.is_err()); // 1st Jan is adjusted to 2nd Jan aligning with front stub
    }

    #[test]
    fn test_try_new_uschedule_eom_parameter_selection() {
        let s = Schedule::try_new_uschedule_infer_frequency(
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
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
            true,
            None,
        )
        .unwrap();
        assert_eq!(
            s.frequency,
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(31))
            }
        );

        let s = Schedule::try_new_uschedule_infer_frequency(
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
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(
            s.frequency,
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(30))
            }
        );

        let s = Schedule::try_new_uschedule_infer_frequency(
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
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
            true,
            None,
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
            Schedule::try_new_infer_stub(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                Some(ndt(2000, 1, 10)),
                Some(ndt(2000, 1, 16)),
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle(1),
                Adjuster::Following {},
                None,
                Some(StubInference::ShortBack)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_infer_stub(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                None,
                Some(ndt(2000, 1, 16)),
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle(1),
                Adjuster::Following {},
                None,
                Some(StubInference::ShortBack)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_infer_stub(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                None,
                Some(ndt(2000, 1, 16)),
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle(1),
                Adjuster::Following {},
                None,
                Some(StubInference::LongBack)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_infer_stub(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                Some(ndt(2000, 1, 16)),
                None,
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle(1),
                Adjuster::Following {},
                None,
                Some(StubInference::ShortFront)
            )
            .is_err()
        );

        // fails because stub date is given as well as an inference enum
        assert_eq!(
            true,
            Schedule::try_new_infer_stub(
                ndt(2000, 1, 1),
                ndt(2000, 2, 1),
                Frequency::CalDays { number: 100 },
                Some(ndt(2000, 1, 16)),
                None,
                Calendar::Cal(Cal::new(vec![], vec![])),
                Adjuster::ModifiedFollowing {},
                Adjuster::BusDaysLagSettle(1),
                Adjuster::Following {},
                None,
                Some(StubInference::LongFront)
            )
            .is_err()
        );
    }

    #[test]
    fn test_try_new_schedule_short_period() {
        // test infer stub works when no stub is required for single period stub case
        let s = Schedule::try_new_uschedule_infer_frequency(
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
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
            true,
            Some(StubInference::ShortFront),
        )
        .expect("short period");
        assert_eq!(s.uschedule, vec![ndt(2022, 7, 1), ndt(2022, 10, 1)]);
    }

    #[test]
    fn test_try_new_schedule_infer_frequency_imm() {
        // test IMM frequency is inferred
        let s = Schedule::try_new_uschedule_infer_frequency(
            ndt(2025, 3, 19),
            ndt(2025, 9, 17),
            Frequency::Months {
                number: 3,
                roll: None,
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
            true,
            None,
        )
        .expect("short period");
        assert_eq!(
            s.frequency,
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::IMM())
            }
        );
    }

    #[test]
    fn test_is_regular() {
        let s = Schedule::try_new_uschedule_infer_frequency(
            ndt(2025, 3, 19),
            ndt(2025, 9, 19),
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(19)),
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
            true,
            None,
        )
        .expect("regular");
        assert!(s.is_regular());

        let s = Schedule::try_new_uschedule_infer_frequency(
            ndt(2025, 3, 19),
            ndt(2025, 9, 25),
            Frequency::Months {
                number: 3,
                roll: Some(RollDay::Day(19)),
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle(1),
            Adjuster::Following {},
            None,
            true,
            Some(StubInference::ShortBack),
        )
        .expect("regular");
        assert!(!s.is_regular());
    }

    #[test]
    fn test_front_stub_inference() {
        let s = Schedule::try_new_inferred(
            ndt(2022, 1, 1),
            ndt(2022, 6, 1),
            Frequency::Months {
                number: 3,
                roll: None,
            },
            None,
            None,
            Calendar::Cal(Cal::new(vec![], vec![])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle(2),
            Adjuster::Following {},
            None,
            false,
            Some(StubInference::ShortFront),
        )
        .expect("schedule is valid");
        assert_eq!(
            s.uschedule,
            vec![ndt(2022, 1, 1), ndt(2022, 3, 1), ndt(2022, 6, 1)]
        );
    }

    #[test]
    fn test_inference_allows_stubs_when_they_are_regular() {
        let s = Schedule::try_new_inferred(
            ndt(2025, 1, 15),
            ndt(2025, 4, 15),
            Frequency::Months {
                number: 1,
                roll: None,
            },
            None,
            Some(ndt(2025, 3, 15)),
            Calendar::Cal(Cal::new(vec![], vec![5, 6])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle(2),
            Adjuster::Following {},
            None,
            false,
            Some(StubInference::ShortFront),
        )
        .expect("schedule is valid");
        assert_eq!(
            s.uschedule,
            vec![
                ndt(2025, 1, 15),
                ndt(2025, 2, 15),
                ndt(2025, 3, 15),
                ndt(2025, 4, 15)
            ]
        );
    }
}
