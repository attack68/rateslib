use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use serde::{Deserialize, Serialize};

use crate::scheduling::{get_calendar_by_name, Cal, CalendarAdjustment, DateRoll, UnionCal};

/// A wrapper for a UnionCal struct specified by a string representation.
///
/// This struct is designed for use when serialization of a calendar as part of an another composite
/// struct seeks to be related to named calendar combinations and not an inefficient list of dates.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "NamedCalDataModel")]
pub struct NamedCal {
    pub(crate) name: String,
    #[serde(skip)]
    pub(crate) union_cal: UnionCal,
}

#[derive(Deserialize)]
struct NamedCalDataModel {
    name: String,
}

impl std::convert::From<NamedCalDataModel> for NamedCal {
    fn from(model: NamedCalDataModel) -> Self {
        Self::try_new(&model.name).expect("NamedCal data model contains bad data.")
    }
}

impl NamedCal {
    /// Create a new named calendar.
    ///
    /// `name` must be a string that contains named calendars separated by commas, additionally
    /// separating business day calendars with associated settlement calendars by a pipe. A valid
    /// example input is "tgt,ldn|fed".
    pub fn try_new(name: &str) -> Result<Self, PyErr> {
        let name_ = name.to_lowercase();
        let parts: Vec<&str> = name_.split("|").collect();
        if parts.len() > 2 {
            Err(PyValueError::new_err(
                "Cannot use more than one pipe ('|') operator in `name`.",
            ))
        } else if parts.len() == 1 {
            let cals: Vec<Cal> = parse_cals(parts[0])?;
            Ok(Self {
                name: name_,
                union_cal: UnionCal {
                    calendars: cals,
                    settlement_calendars: None,
                },
            })
        } else {
            let cals: Vec<Cal> = parse_cals(parts[0])?;
            let settle_cals: Vec<Cal> = parse_cals(parts[1])?;
            Ok(Self {
                name: name_,
                union_cal: UnionCal {
                    calendars: cals,
                    settlement_calendars: Some(settle_cals),
                },
            })
        }
    }
}

impl DateRoll for NamedCal {
    fn is_weekday(&self, date: &NaiveDateTime) -> bool {
        self.union_cal.is_weekday(date)
    }

    fn is_holiday(&self, date: &NaiveDateTime) -> bool {
        self.union_cal.is_holiday(date)
    }

    fn is_settlement(&self, date: &NaiveDateTime) -> bool {
        self.union_cal.is_settlement(date)
    }
}

impl CalendarAdjustment for NamedCal {}

fn parse_cals(name: &str) -> Result<Vec<Cal>, PyErr> {
    let mut cals: Vec<Cal> = Vec::new();
    for cal in name.split(",") {
        cals.push(get_calendar_by_name(cal)?)
    }
    Ok(cals)
}

impl<T> PartialEq<T> for NamedCal
where
    T: DateRoll,
{
    fn eq(&self, other: &T) -> bool {
        self.union_cal.eq(other)
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::ndt;

    #[test]
    fn test_named_cal() {
        let ncal = NamedCal::try_new("tgt,nyc").unwrap();

        assert!(ncal.is_non_bus_day(&ndt(1970, 2, 16))); // NYC holiday
        assert!(ncal.is_non_bus_day(&ndt(1970, 5, 1))); // TGT holiday
        assert!(ncal.is_bus_day(&ndt(1970, 2, 17)));
    }

    #[test]
    fn test_named_cal_pipe() {
        let ncal = NamedCal::try_new("tgt,nyc|ldn").unwrap();

        assert!(ncal.is_non_bus_day(&ndt(1970, 2, 16))); // NYC holiday
        assert!(ncal.is_non_bus_day(&ndt(1970, 5, 1))); // TGT holiday
        assert!(ncal.is_bus_day(&ndt(1970, 2, 17)));

        assert!(!ncal.is_settlement(&ndt(1970, 5, 4))); // LDN holiday
        assert!(ncal.is_settlement(&ndt(1970, 5, 1))); // not LDN holiday
    }

    #[test]
    fn test_named_cal_error() {
        let ncal = NamedCal::try_new("tgt,nyc|ldn|");
        assert!(ncal.is_err());

        let ncal = NamedCal::try_new("");
        assert!(ncal.is_err());
    }
}
