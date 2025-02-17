
use crate::calendars::{
    CalType,
    RollDay,
    Modifier,
};
use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};

/// A schedule frequency.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone)]
pub enum Frequency {
    /// Periods every month.
    Monthly,
    /// Periods every two months.
    BiMonthly,
    /// Periods every three months.
    Quarterly,
    /// Periods every four months.
    TriAnnually,
    /// Periods every six months.
    SemiAnnually,
    /// Periods every twelve months.
    Annually,
    /// Only every a single period.
    Zero,
}

impl Frequency {
    pub fn months(&self) -> u32 {
        match self {
            Frequency::Monthly => 1_u32,
            Frequency::BiMonthly => 2_u32,
            Frequency::Quarterly => 3_u32,
            Frequency::TriAnnually => 4_u32,
            Frequency::SemiAnnually => 6_u32,
            Frequency::Annually => 12_u32,
            Frequency::Zero => 120000_u32  // 10,000 years.
        }

    }
}

/// A stub type indicator for date inference.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone)]
pub enum Stub {
    /// Short front stub inference.
    ShortFront,
    /// Long front stub inference.
    LongFront,
    /// Short back stub inference.
    ShortBack,
    /// Long back stub inference.
    LongBack,
}

pub struct Schedule {
    ueffective: NaiveDateTime,
    utermination: NaiveDateTime,
    frequency: Frequency,
    front_stub: Option<NaiveDateTime>,
    back_stub: Option<NaiveDateTime>,
    roll: RollDay,
    modifier: Modifier,
    calendar: CalType,
    payment_lag: i8,

    // created data objects
    uschedule: Vec<NaiveDateTime>,
    aschedule: Vec<NaiveDateTime>,
    pschedule: Vec<NaiveDateTime>
}

// impl Schedule {
//     pub fn try_new(
//         effective: NaiveDateTime,
//         termination: NaiveDateTime,
//         frequency: Frequency,
//         stub: Stub,
//         front_stub: Option<NaiveDateTime>,
//         back_stub: Option<NaiveDateTime>,
//         roll: Option<RollDay>,
//         eom: bool,
//         modifier: Modifier,
//         calendar: CalType,
//         payment_lag: i8,
//     ) -> Result<Self, PyErr> {
//         OK()
//     }
// }

fn _is_divisible_months(start: &NaiveDateTime, end: &NaiveDateTime, frequency: &Frequency) -> bool {
    let months = end.month() - start.month();
    return (months % frequency.months()) == 0_u32
}



// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;

//     fn fixture_hol_cal() -> Cal {
//         let hols = vec![ndt(2015, 9, 5), ndt(2015, 9, 7)]; // Saturday and Monday
//         Cal::new(hols, vec![5, 6])
//     }

    #[test]
    fn test_is_divisible_months() {
        // test true
        let s1 = ndt(2022, 2, 2);
        let e1 = ndt(2022, 6, 6);
        assert_eq!(true, _is_divisible_months(&s1, &e1, &Frequency::BiMonthly));

        // test false
        let s2 = ndt(2022, 2, 2);
        let e2 = ndt(2022, 9, 6);
        assert_eq!(false, _is_divisible_months(&s2, &e2, &Frequency::Quarterly));
    }
}