use chrono::prelude::*;
use chrono::Months;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

use crate::scheduling::{
    ndt, Adjuster, Adjustment, Calendar, DateRoll, Frequency, Imm, RollDay, Scheduling,
};

/// Specifier for day count conventions
#[pyclass(module = "rateslib.rs", eq, eq_int)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum Convention {
    /// Actual days in period divided by 365.
    Act365F = 0,
    /// Actual days in period divided by 360.
    Act360 = 1,
    /// 30 days in month and 360 days in year with month end modification rules.
    ///
    /// - Start day is *min(30, start day)*.
    /// - End day is *min(30, end day)* if start day is 30.
    Thirty360 = 2,
    /// 30 days in month and 360 days in year with month end modification rules.
    ///
    /// - Start day is *min(30, start day)* or 30 if start day is EoM February and roll is EoM.
    /// - End day is *min(30, end day)* if start day is 30, or 30 if start and end are EoM February
    ///   and roll is EoM.
    ThirtyU360 = 3,
    /// 30 days in month and 360 days in year with month end modification rules.
    ///
    /// - Start day is *min(30, start day)*.
    /// - End day is *min(30, end day)*.
    ThirtyE360 = 4,
    /// 30 days in month and 360 days in year with month end modification rules.
    ///
    /// - Start day is *min(30, start day)* or 30 if start day is EoM February.
    /// - End day is *min(30, end day)* or 30 if end day is EoM February and not *Leg* termination.
    ThirtyE360ISDA = 5,
    /// Number of whole years plus fractional end period according to 'Act365F'.
    YearsAct365F = 6,
    /// Number of whole years plus fractional end period according to 'Act360'.
    YearsAct360 = 7,
    /// Number of whole years plus fractional counting months difference divided by 12.
    YearsMonths = 8,
    /// Return 1.0 for any period.
    One = 9,
    /// Actual days divided by actual days with leap year modification rules.
    ActActISDA = 10,
    /// Day count based on [Frequency] definition.
    ActActICMA = 11,
    /// Number of business days in period divided by 252.
    Bus252 = 12,
    /// ActActICMA falling back to Act365F in stub periods.
    ActActICMA_Stub_Act365F = 13,
}

impl Convention {
    pub fn dcf(
        &self,
        start: &NaiveDateTime,
        end: &NaiveDateTime,
        termination: Option<&NaiveDateTime>,
        frequency: Option<&Frequency>,
        stub: Option<bool>,
        calendar: Option<&Calendar>,
        adjuster: Option<&Adjuster>,
    ) -> Result<f64, PyErr> {
        match self {
            Convention::Act360 => Ok(dcf_act_numeric(360.0, start, end)),
            Convention::Act365F => Ok(dcf_act_numeric(365.0, start, end)),
            Convention::YearsAct365F => Ok(dcf_years_and_act_numeric(365.0, start, end)),
            Convention::YearsAct360 => Ok(dcf_years_and_act_numeric(360.0, start, end)),
            Convention::YearsMonths => Ok(dcf_years_and_months(start, end)),
            Convention::Thirty360 => Ok(dcf_30360(start, end)),
            Convention::ThirtyU360 => dcf_30u360(start, end, frequency),
            Convention::ThirtyE360 => Ok(dcf_30e360(start, end)),
            Convention::ThirtyE360ISDA => dcf_30e360_isda(start, end, termination),
            Convention::One => Ok(1.0),
            Convention::ActActISDA => Ok(dcf_act_isda(start, end)),
            Convention::ActActICMA => {
                if frequency.is_none() {
                    Err(PyValueError::new_err(
                        "`frequency` must be supplied for 'ActActICMA' type convention.",
                    ))
                } else if stub.is_none() {
                    Err(PyValueError::new_err(
                        "`stub` must be supplied for 'ActActICMA' type convention.",
                    ))
                } else {
                    dcf_act_icma(
                        start,
                        end,
                        termination,
                        frequency.unwrap(),
                        stub.unwrap(),
                        calendar,
                        adjuster,
                    )
                }
            }
            Convention::Bus252 => {
                if calendar.is_none() {
                    Err(PyValueError::new_err(
                        "`calendar` must be supplied for 'Bus252' type convention.",
                    ))
                } else {
                    Ok(dcf_bus252(start, end, calendar.unwrap()))
                }
            }
            Convention::ActActICMA_Stub_Act365F => {
                if frequency.is_none() {
                    Err(PyValueError::new_err(
                        "`frequency` must be supplied for 'ActActICMA' type convention.",
                    ))
                } else if stub.is_none() {
                    Err(PyValueError::new_err(
                        "`stub` must be supplied for 'ActActICMA' type convention.",
                    ))
                } else {
                    dcf_act_icma_stub_365f(
                        start,
                        end,
                        termination,
                        frequency.unwrap(),
                        stub.unwrap(),
                        calendar,
                        adjuster,
                    )
                }
            }
        }
    }
}

fn dcf_act_numeric(denominator: f64, start: &NaiveDateTime, end: &NaiveDateTime) -> f64 {
    (*end - *start).num_days() as f64 / denominator
}

fn dcf_years_and_act_numeric(denominator: f64, start: &NaiveDateTime, end: &NaiveDateTime) -> f64 {
    if *end <= (*start + Months::new(12)) {
        dcf_act_numeric(denominator, start, end)
    } else {
        let intermediate = RollDay::Day(start.day())
            .try_from_ym(end.year(), start.month())
            .expect("Dates are out of bounds");
        if intermediate <= *end {
            let years: f64 = (end.year() - start.year()) as f64;
            years + dcf_act_numeric(denominator, &intermediate, end)
        } else {
            let years: f64 = (end.year() - start.year()) as f64 - 1.0;
            years + dcf_act_numeric(denominator, &(intermediate - Months::new(12)), end)
        }
    }
}

fn dcf_years_and_months(start: &NaiveDateTime, end: &NaiveDateTime) -> f64 {
    let start_ = ndt(start.year(), start.month(), 1);
    let end_ = ndt(end.year(), end.month(), 1);
    let mut count_date = ndt(end.year(), start.month(), 1);
    if count_date > end_ {
        count_date = count_date - Months::new(12)
    };
    let years = count_date.year() - start_.year();
    let mut counter = 0;
    while count_date < end_ {
        count_date = count_date + Months::new(1);
        counter += 1;
    }
    years as f64 + counter as f64 / 12.0
}

/// Normal 30360 without any adjustments
fn dcf_30360_unadjusted(ys: i32, ms: u32, ds: u32, ye: i32, me: u32, de: u32) -> f64 {
    (ye - ys) as f64 + (me as f64 - ms as f64) / 12.0 + (de as f64 - ds as f64) / 360.0
}

/// Return DCF under 30360 convention.
///
/// - start.day is adjusted to min(30, start.day)
/// - end.day is adjusted to min(30, end.day) only if start.day is 30.
/// - calculation proceeds as normal
fn dcf_30360(start: &NaiveDateTime, end: &NaiveDateTime) -> f64 {
    let ds = u32::min(30_u32, start.day());
    let de = if ds == 30 {
        u32::min(30_u32, end.day())
    } else {
        end.day()
    };
    dcf_30360_unadjusted(start.year(), start.month(), ds, end.year(), end.month(), de)
}

/// Return DCF under 30e360 convention.
///
/// - start.day is adjusted to min(30, start.day)
/// - end.day is adjusted to min(30, end.day)
/// - calculation proceeds as normal
fn dcf_30e360(start: &NaiveDateTime, end: &NaiveDateTime) -> f64 {
    let ds = u32::min(30_u32, start.day());
    let de = u32::min(30_u32, end.day());
    dcf_30360_unadjusted(start.year(), start.month(), ds, end.year(), end.month(), de)
}

/// Return DCF under 30u360 convention.
///
/// - start.day is 30 if roll is EoM and start is last day in February.
/// - end.day is 30 if roll is EoM and start and end are both last days of February.
/// - start.day is 30 if start.day is 31.
/// - end.day is 30 if end.day is 31 and start.day is 30.
///
/// # Notes
/// `frequency` is only evaluated to determine a [RollDay] if start is end of February.
fn dcf_30u360(
    start: &NaiveDateTime,
    end: &NaiveDateTime,
    frequency: Option<&Frequency>,
) -> Result<f64, PyErr> {
    let mut ds = start.day();
    let mut de = end.day();

    // handle February EoM rolls adjustment
    if Imm::Eom.validate(start) && start.month() == 2 {
        let roll: RollDay = match frequency {
            Some(Frequency::Months {
                number: _,
                roll: Some(r),
            }) => *r,
            _ => {
                return Err(PyValueError::new_err(
                    "`frequency` must be provided or has no `roll`. A roll-day must be supplied for '30u360' convention to detect February EoM rolls.\n`start` is detected as end of February, otherwise use '30360' which will leave this date unadjusted.",
                ));
            }
        };
        if roll == RollDay::Day(31) {
            ds = 30;
            if Imm::Eom.validate(end) && end.month() == 2 {
                de = 30;
            }
        }
    }

    // perform regular 30360 adjustments
    ds = u32::min(30_u32, ds);
    if de == 31 && ds == 30 {
        de = 30;
    }
    Ok(dcf_30360_unadjusted(
        start.year(),
        start.month(),
        ds,
        end.year(),
        end.month(),
        de,
    ))
}

/// Return DCF under 30e360ISDA convention.
///
/// - start.day is 30 if start.day is 31 or start.day is end of February.
/// - end.day is 30 if end.day is 31 or end.day is end of February and not the termination date.
fn dcf_30e360_isda(
    start: &NaiveDateTime,
    end: &NaiveDateTime,
    termination: Option<&NaiveDateTime>,
) -> Result<f64, PyErr> {
    let mut ds = u32::min(30_u32, start.day());

    //handle February EoM adjustments
    if Imm::Eom.validate(start) && start.month() == 2 {
        ds = 30;
    }
    let mut de = u32::min(30_u32, end.day());
    if Imm::Eom.validate(end) && end.month() == 2 {
        if termination.is_none() {
            return Err(PyValueError::new_err(
                "`termination` must be provided for '30e360ISDA' convention to detect end of February.\n`end` is detected as end of February, otherwise use '30e360' which will leave this date unadjusted.",
            ));
        } else if *end != *(termination.unwrap()) {
            de = 30;
        }
    }

    Ok(dcf_30360_unadjusted(
        start.year(),
        start.month(),
        ds,
        end.year(),
        end.month(),
        de,
    ))
}

fn dcf_act_isda(start: &NaiveDateTime, end: &NaiveDateTime) -> f64 {
    if start == end {
        return 0.0;
    };

    let is_start_leap = NaiveDate::from_ymd_opt(start.year(), 2, 29).is_some();
    let is_end_leap = NaiveDate::from_ymd_opt(end.year(), 2, 29).is_some();

    let year_1_diff = if is_start_leap { 366.0 } else { 365.0 };
    let year_2_diff = if is_end_leap { 366.0 } else { 365.0 };

    let mut total_sum: f64 = (end.year() - start.year()) as f64 - 1.0;
    total_sum += (ndt(start.year() + 1, 1, 1) - *start).num_days() as f64 / year_1_diff;
    total_sum += (*end - ndt(end.year(), 1, 1)).num_days() as f64 / year_2_diff;
    total_sum
}

fn dcf_act_icma(
    start: &NaiveDateTime,
    end: &NaiveDateTime,
    termination: Option<&NaiveDateTime>,
    frequency: &Frequency,
    stub: bool,
    calendar: Option<&Calendar>,
    adjuster: Option<&Adjuster>,
) -> Result<f64, PyErr> {
    if !stub {
        Ok(1.0 / frequency.periods_per_annum())
    } else {
        if termination.is_none() || adjuster.is_none() || calendar.is_none() {
            return Err(PyValueError::new_err(
                "Stub periods under ActActICMA require `termination`, `adjuster` and `calendar` arguments to determine appropriate fractions."
            ));
        }
        let is_back_stub = end == termination.unwrap();
        let mut fraction = -1.0;
        if is_back_stub {
            let mut qe0 = *start;
            let mut qe1 = *start;
            while *end > qe1 {
                fraction += 1.0;
                qe0 = qe1;
                qe1 = (*(adjuster.unwrap())).adjust(&frequency.next(&qe0), calendar.unwrap());
            }
            fraction =
                fraction + ((*end - qe0).num_days() as f64) / ((qe1 - qe0).num_days() as f64);
            Ok(fraction / frequency.periods_per_annum())
        } else {
            let mut qs0 = *end;
            let mut qs1 = *end;
            while *start < qs1 {
                fraction += 1.0;
                qs0 = qs1;
                qs1 = (*(adjuster.unwrap())).adjust(&frequency.previous(&qs0), calendar.unwrap());
            }
            fraction =
                fraction + ((qs0 - *start).num_days() as f64) / ((qs0 - qs1).num_days() as f64);
            Ok(fraction / frequency.periods_per_annum())
        }
    }
}

fn dcf_act_icma_stub_365f(
    start: &NaiveDateTime,
    end: &NaiveDateTime,
    termination: Option<&NaiveDateTime>,
    frequency: &Frequency,
    stub: bool,
    calendar: Option<&Calendar>,
    adjuster: Option<&Adjuster>,
) -> Result<f64, PyErr> {
    if !stub {
        Ok(1.0 / frequency.periods_per_annum())
    } else {
        let ppa = frequency.periods_per_annum();
        if termination.is_none() || adjuster.is_none() || calendar.is_none() {
            return Err(PyValueError::new_err(
                "Stub periods under ActActICMA require `termination`, `adjuster` and `calendar` arguments to determine appropriate fractions."
            ));
        }
        let is_back_stub = end == termination.unwrap();
        let mut fraction = -1.0;
        if is_back_stub {
            let mut qe0 = *start;
            let mut qe1 = *start;
            while *end > qe1 {
                fraction += 1.0;
                qe0 = qe1;
                qe1 = (*(adjuster.unwrap())).adjust(&frequency.next(&qe0), calendar.unwrap());
            }
            fraction = fraction + ppa * (*end - qe0).num_days() as f64 / 365.0;
            Ok(fraction / ppa)
        } else {
            let mut qs0 = *end;
            let mut qs1 = *end;
            while *start < qs1 {
                fraction += 1.0;
                qs0 = qs1;
                qs1 = (*(adjuster.unwrap())).adjust(&frequency.previous(&qs0), calendar.unwrap());
            }
            fraction = fraction + ppa * (qs0 - *start).num_days() as f64 / 365.0;
            Ok(fraction / ppa)
        }
    }
}

fn dcf_bus252(start: &NaiveDateTime, end: &NaiveDateTime, calendar: &Calendar) -> f64 {
    if end < start {
        panic!("Given end is greater than start");
    } else if start == end {
        return 0.0;
    }
    let start_bd = Adjuster::Following {}.adjust(start, calendar);
    let end_bd = Adjuster::Previous {}.adjust(end, calendar);
    let subtract = if end_bd == *end { -1.0 } else { 0.0 };
    if start_bd == end_bd {
        if start_bd > *start && end_bd < *end {
            //then logically there is one b.d. between the non-business start and non-business end
            1.0 / 252.0
        } else if end_bd < *end {
            // then the business start is permitted to the calculation until the non-business end
            1.0 / 252.0
        } else {
            // start_bd > start
            // then the business end is not permitted to have occurred and non-business start
            // does not count
            0.0
        }
    } else if start_bd > end_bd {
        // there are no business days in between start and end
        0.0
    } else {
        (calendar.bus_date_range(&start_bd, &end_bd).unwrap().len() as f64 + subtract) / 252.0
    }
}

#[pyfunction]
pub(crate) fn _get_convention_str(convention: Convention) -> String {
    match convention {
        Convention::Act365F => "Act365F".to_string(),
        Convention::Act360 => "Act360".to_string(),
        Convention::YearsAct365F => "Act365F+".to_string(),
        Convention::YearsAct360 => "Act360+".to_string(),
        Convention::YearsMonths => "1+".to_string(),
        Convention::Thirty360 => "30360".to_string(),
        Convention::ThirtyU360 => "30u360".to_string(),
        Convention::ThirtyE360 => "30e360".to_string(),
        Convention::ThirtyE360ISDA => "30e360ISDA".to_string(),
        Convention::ActActISDA => "ActActISDA".to_string(),
        Convention::ActActICMA => "ActActICMA".to_string(),
        Convention::One => "1".to_string(),
        Convention::Bus252 => "Bus252".to_string(),
        Convention::ActActICMA_Stub_Act365F => "ActActICMA_Stub_Act365F".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{ndt, Cal};

    #[test]
    fn test_act_numeric() {
        let result = dcf_act_numeric(10.0, &ndt(2000, 1, 1), &ndt(2000, 1, 21));
        assert_eq!(result, 2.0)
    }

    #[test]
    fn test_act_plus() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, f64)> = vec![
            (ndt(2000, 1, 1), ndt(2002, 1, 21), 2.0 + 20.0 / 365.0),
            (ndt(2000, 12, 31), ndt(2002, 1, 1), 1.0 + 1.0 / 365.0),
            (ndt(2000, 12, 31), ndt(2002, 12, 31), 2.0),
            (ndt(2024, 2, 29), ndt(2025, 2, 28), 1.0),
            (ndt(2000, 12, 15), ndt(2003, 1, 15), 2.0 + 31.0 / 365.0),
        ];
        for option in options {
            let result = dcf_years_and_act_numeric(365.0, &option.0, &option.1);
            assert_eq!(result, option.2)
        }
    }

    #[test]
    fn test_30360() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, f64)> = vec![
            (ndt(2000, 1, 1), ndt(2000, 1, 21), 20.0 / 360.0),
            (
                ndt(2000, 1, 1),
                ndt(2001, 3, 21),
                1.0 + 2.0 / 12.0 + 20.0 / 360.0,
            ),
        ];
        for option in options {
            let result = dcf_30360(&option.0, &option.1);
            assert_eq!(result, option.2)
        }
    }

    #[test]
    fn test_30u360() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, Frequency, f64)> = vec![
            (
                ndt(2000, 1, 1),
                ndt(2000, 1, 21),
                Frequency::Months {
                    number: 1,
                    roll: Some(RollDay::Day(1)),
                },
                20.0 / 360.0,
            ),
            (
                ndt(2000, 1, 1),
                ndt(2001, 3, 21),
                Frequency::CalDays { number: 20 },
                1.0 + 2.0 / 12.0 + 20.0 / 360.0,
            ),
            (
                ndt(2024, 2, 29),
                ndt(2025, 2, 28),
                Frequency::Months {
                    number: 12,
                    roll: Some(RollDay::Day(29)),
                },
                1.0 - 1.0 / 360.0,
            ),
            (
                ndt(2024, 2, 29),
                ndt(2025, 2, 28),
                Frequency::Months {
                    number: 12,
                    roll: Some(RollDay::Day(31)),
                },
                1.0,
            ),
        ];
        for option in options {
            let result = dcf_30u360(&option.0, &option.1, Some(&option.2)).unwrap();
            assert_eq!(result, option.3);
        }
    }

    #[test]
    fn test_years_and_months() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, f64)> = vec![
            (ndt(2000, 1, 1), ndt(2000, 1, 21), 0.0),
            (ndt(2000, 1, 1), ndt(2001, 3, 21), 1.0 + 2.0 / 12.0),
            (ndt(2024, 2, 29), ndt(2025, 2, 28), 1.0),
            (ndt(2024, 2, 29), ndt(2025, 2, 28), 1.0),
            (ndt(2000, 12, 29), ndt(2025, 1, 12), 24.0 + 1.0 / 12.0),
        ];
        for option in options {
            let result = dcf_years_and_months(&option.0, &option.1);
            assert_eq!(result, option.2)
        }
    }

    #[test]
    fn test_actacticma() {
        let options: Vec<(NaiveDateTime, NaiveDateTime, Frequency, f64)> = vec![
            (
                ndt(1999, 2, 1),
                ndt(1999, 7, 1),
                Frequency::Months {
                    number: 12,
                    roll: None,
                },
                150.0 / 365.0,
            ),
            (
                ndt(2002, 8, 15),
                ndt(2003, 7, 15),
                Frequency::Months {
                    number: 6,
                    roll: None,
                },
                0.5 + 153.0 / 368.0,
            ),
        ];
        for option in options {
            let result = dcf_act_icma(
                &option.0,
                &option.1,
                Some(&ndt(2099, 1, 1)),
                &option.2,
                true,
                Some(&Cal::new(vec![], vec![]).into()),
                Some(&Adjuster::Actual {}),
            )
            .unwrap();
            assert_eq!(result, option.3)
        }
    }
}
