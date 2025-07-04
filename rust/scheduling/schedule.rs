use crate::scheduling::{Adjuster, Adjustment, Calendar, Frequency, Scheduling};
use chrono::prelude::*;
// use chrono::Days;
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
    /// Generate an unadjusted schedule with stub inference.
    ///
    /// # Notes
    ///
    /// If ``stub_inference`` is `None` then this method will revert to [Schedule::try_new_uschedule].
    /// If ``stub_inference`` is given but it conflicts with an explicit ``stub`` date given then
    /// and error will be returned.
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
        stub_inference: Option<StubInference>,
    ) -> Result<Self, PyErr> {
        // method will conditionally branch to non-inferred method if no inference enum.
        if stub_inference.is_none() {
            return Self::try_new_uschedule(
                ueffective,
                utermination,
                frequency,
                ufront_stub,
                uback_stub,
                calendar,
                accrual_adjuster,
                payment_adjuster,
            );
        }

        // validate inference is not blocked by user defined values.
        let _ = validate_stub_dates_and_inference(&ufront_stub, &uback_stub, &stub_inference)?;

        let (interior_start, interior_end) =
            match_interior_dates(&ueffective, ufront_stub, uback_stub, &utermination);
        if frequency
            .try_uregular(&interior_start, &interior_end)
            .is_ok()
        {
            // no inference is required
            return Self::try_new_uschedule(
                ueffective,
                utermination,
                frequency,
                ufront_stub,
                uback_stub,
                calendar,
                accrual_adjuster,
                payment_adjuster,
            );
        } else {
            let (ufront_stub_, uback_stub_) = match stub_inference.unwrap() {
                StubInference::ShortFront => (
                    Some(frequency.try_infer_ufront_stub(&interior_start, &interior_end, true)?),
                    uback_stub,
                ),
                StubInference::LongFront => (
                    Some(frequency.try_infer_ufront_stub(&interior_start, &interior_end, false)?),
                    uback_stub,
                ),
                StubInference::ShortBack => (
                    ufront_stub,
                    Some(frequency.try_infer_uback_stub(&interior_start, &interior_end, true)?),
                ),
                StubInference::LongBack => (
                    ufront_stub,
                    Some(frequency.try_infer_uback_stub(&interior_start, &interior_end, false)?),
                ),
            };
            return Self::try_new_uschedule(
                ueffective,
                utermination,
                frequency,
                ufront_stub_,
                uback_stub_,
                calendar,
                accrual_adjuster,
                payment_adjuster,
            );
        }
    }

    //     /// Generate a [Schedule] from possibly adjusted dates.
    //     ///
    //     pub fn try_new_schedule(
    //         effective: NaiveDateTime,
    //         termination: NaiveDateTime,
    //         frequency: Frequency,
    //         ufront_stub: Option<NaiveDateTime>,
    //         uback_stub: Option<NaiveDateTime>,
    //         calendar: Calendar,
    //         accrual_adjuster: Adjuster,
    //         payment_adjuster: Adjuster,
    //     ) -> Result<Self, PyErr> {
    //         let ueffectives = get_unadjusteds(&effective, &accrual_adjuster, &calendar);
    //         let uterminations = get_unadjusteds(&termination, &accrual_adjuster, &calendar);
    //
    //         Err(PyValueError::new_err("Not Yet Implemented"))
    //     }

    /// Generate an unadjusted schedule.
    ///
    /// # Notes
    ///
    /// An unadjusted regular schedule, that aligns with [Frequency], must be defined between
    /// the relevant dates. If not an error is returned.
    pub fn try_new_uschedule(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
    ) -> Result<Self, PyErr> {
        let (regular_start, regular_end) =
            match_interior_dates(&ueffective, ufront_stub, uback_stub, &utermination);

        // test if the determined regular period is actually a regular period under Frequency
        let uregular = frequency.try_uregular(&regular_start, &regular_end)?;
        let uschedule =
            composite_uschedule(ueffective, utermination, ufront_stub, uback_stub, uregular);
        let aschedule: Vec<NaiveDateTime> = uschedule
            .iter()
            .map(|dt| accrual_adjuster.adjust(&dt, &calendar))
            .collect();
        let pschedule = aschedule
            .iter()
            .map(|dt| payment_adjuster.adjust(&dt, &calendar))
            .collect();
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
}

// /// Get unadjusted date alternatives for an associated adjusted date.
// fn get_unadjusteds(date: &NaiveDateTime, adjuster: &Adjuster, calendar: &Calendar) -> Vec<NaiveDateTime> {
//     let mut udates: Vec<NaiveDateTime> = vec![];
//     let mut udate = *date;
//     while adjuster.adjust(&udate, calendar) == *date {
//         udates.push(udate);
//         udate = udate - Days::new(1);
//     }
//     udate = *date + Days::new(1);
//     while adjuster.adjust(&udate, calendar) == *date {
//         udates.push(udate);
//         udate = udate + Days::new(1);
//     }
//     udates
// }

fn match_interior_dates(
    ueffective: &NaiveDateTime,
    ufront_stub: Option<NaiveDateTime>,
    uback_stub: Option<NaiveDateTime>,
    utermination: &NaiveDateTime,
) -> (NaiveDateTime, NaiveDateTime) {
    match (ufront_stub, uback_stub) {
        (None, None) => (*ueffective, *utermination),
        (Some(v), None) => (v, *utermination),
        (None, Some(v)) => (*ueffective, v),
        (Some(v), Some(w)) => (v, w),
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
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    ufront_stub: Option<NaiveDateTime>,
    uback_stub: Option<NaiveDateTime>,
    regular_uschedule: Vec<NaiveDateTime>,
) -> Vec<NaiveDateTime> {
    let mut uschedule: Vec<NaiveDateTime> = vec![];
    match (ufront_stub, uback_stub) {
        (None, None) => {
            uschedule.extend(regular_uschedule);
        }
        (Some(_v), None) => {
            uschedule.push(ueffective);
            uschedule.extend(regular_uschedule);
        }
        (None, Some(_v)) => {
            uschedule.extend(regular_uschedule);
            uschedule.push(utermination);
        }
        (Some(_v), Some(_w)) => {
            uschedule.push(ueffective);
            uschedule.extend(regular_uschedule);
            uschedule.push(utermination);
        }
    }
    uschedule
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt, Cal, RollDay};

    //     fn fixture_hol_cal() -> Cal {
    //         let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
    //         Cal::new(hols, vec![5, 6])
    //     }

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
                Some(StubInference::LongFront)
            )
            .is_err()
        );
    }

    #[test]
    fn test_try_new_uschedule() {
        let s = Schedule::try_new_uschedule(
            ndt(2000, 1, 1),
            ndt(2000, 12, 15),
            Frequency::Months {
                number: 3,
                roll: RollDay::Unspecified {},
            },
            Some(ndt(2000, 3, 15)),
            None,
            Calendar::Cal(Cal::new(vec![], vec![])),
            Adjuster::ModifiedFollowing {},
            Adjuster::BusDaysLagSettle { number: 1 },
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
}
