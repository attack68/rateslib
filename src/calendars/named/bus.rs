//! Define the parameters for a generic Western business day calendar without any specific holidays.

pub const WEEKMASK: &'static [u8] = &[5, 6];  // Saturday and Sunday weekend
pub const HOLIDAYS: &'static [&str] = &[];  // no specific holidays
