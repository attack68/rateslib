use pyo3::{pyclass, pyfunction};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum Convention {
    /// Return 1.0 for any period.
    One,
    /// Number of months between dates divided by 12.
    OnePlus,
    /// Actual days in period divided by 365.
    Act365F,
    /// Number of whole years plus actual days in fractional periods divided by 365.
    Act365FPlus,
    /// Actual days in period divided by 360.
    Act360,
    ThirtyE360,
    Thirty360,
    Thirty360ISDA,
    ActActISDA,
    ActActICMA,
    Bus252,
}

#[pyfunction]
pub(crate) fn _get_convention_str(convention: Convention) -> String {
    match convention {
        Convention::Act365F => "Act365F".to_string(),
        Convention::Act365FPlus => "Act365F+".to_string(),
        Convention::Act360 => "Act360".to_string(),
        Convention::Thirty360 => "30360".to_string(),
        Convention::ThirtyE360 => "30e360".to_string(),
        Convention::Thirty360ISDA => "30e360ISDA".to_string(),
        Convention::ActActISDA => "ActActISDA".to_string(),
        Convention::ActActICMA => "ActActICMA".to_string(),
        Convention::One => "1".to_string(),
        Convention::OnePlus => "1+".to_string(),
        Convention::Bus252 => "Bus252".to_string(),
    }
}
