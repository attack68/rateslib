// SPDX-License-Identifier: LicenseRef-Rateslib-Dual
//
// Copyright (c) 2026 Siffrorna Technology Limited
// This code cannot be used or copied externally
//
// Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
// Source-available, not open source.
//
// See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
// and/or contact info (at) rateslib (dot) com
////////////////////////////////////////////////////////////////////////////////////////////////////

mod adjuster;
mod cal;
mod calendar;
mod dateroll;
mod manager;
mod named;
mod named_cal;
mod union_cal;

pub use crate::scheduling::calendars::{
    adjuster::{Adjuster, Adjustment, CalendarAdjustment},
    cal::Cal,
    calendar::{ndt, Calendar},
    dateroll::DateRoll,
    manager::CalendarManager,
    named_cal::NamedCal,
    union_cal::UnionCal,
};

pub(crate) use crate::scheduling::calendars::named_cal::CalWrapper;

macro_rules! impl_date_roll_partial_eq {
    ($t1:ty, $t2:ty) => {
        // Implement T1 == T2
        impl PartialEq<$t2> for $t1 {
            fn eq(&self, other: &$t2) -> bool {
                let c = self
                    .cal_date_range(&ndt(1970, 1, 1), &ndt(2200, 12, 31))
                    .unwrap();
                c.iter().all(|d| {
                    self.is_bus_day(d) == other.is_bus_day(d)
                        && self.is_settlement(d) == other.is_settlement(d)
                })
            }
        }
    };
}

// Usage: Just list the pairs you want to support
impl_date_roll_partial_eq!(Cal, UnionCal);
impl_date_roll_partial_eq!(Cal, NamedCal);
impl_date_roll_partial_eq!(UnionCal, Cal);
impl_date_roll_partial_eq!(UnionCal, UnionCal);
impl_date_roll_partial_eq!(UnionCal, NamedCal);
impl_date_roll_partial_eq!(NamedCal, Cal);
impl_date_roll_partial_eq!(NamedCal, UnionCal);
impl_date_roll_partial_eq!(NamedCal, NamedCal);
