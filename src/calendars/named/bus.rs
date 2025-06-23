//! Define a generic Western business weekday calendar without any specific holidays.

pub const WEEKMASK: &[u8] = &[5, 6]; // Saturday and Sunday weekend
pub const RULES: &[&str] = &[];
pub const HOLIDAYS: &[&str] = &[]; // no specific holidays
