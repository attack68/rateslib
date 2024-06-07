//! Define a generic Western business weekday calendar without any specific holidays.

pub const WEEKMASK: &'static [u8] = &[5, 6];  // Saturday and Sunday weekend
pub const RULES: &'static [&str] = &[];
pub const HOLIDAYS: &'static [&str] = &[];  // no specific holidays
