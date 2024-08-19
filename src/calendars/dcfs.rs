use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;
use pyo3::pyclass;

#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
