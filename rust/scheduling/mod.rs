//! Create business day calendars and perform financial date manipulation.
//!
//! ### Basic usage
//!
//! The `Cal` struct allows the construction of a single business day calendar, e.g.
//! a particular currency calendar. The below constructs two separate calendars,
//! one for some London holidays and
//! one for some Tokyo holidays in 2017.
//!
//! ```rust
//! # use rateslib::calendars::{Cal, ndt};
//! // UK Monday 1st May Bank Holiday
//! let ldn = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! // Japan Constitution Memorial Day, Greenery Day, Children's Day
//! let tky = Cal::new(vec![ndt(2017, 5, 3), ndt(2017, 5, 4), ndt(2017, 5, 5)], vec![5, 6]);
//! ```
//! These calendars are used to manipulate dates e.g.
//!
//! ```rust
//! # use rateslib::calendars::{Cal, ndt, DateRoll};
//! # let ldn = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! let date = ndt(2017, 4, 28);  // Friday 28th April 2017
//! let spot = ldn.add_bus_days(&date, 2, true).unwrap();
//! // Wednesday 3rd May 2017, observing the holiday.
//! assert_eq!(spot, ndt(2017, 5, 3));
//! ```
//!
//! ### Combination usage
//!
//! For use with multi-currency products calendars often need to be combined.
//!
//! ```rust
//! # use rateslib::calendars::{UnionCal, Cal, ndt, DateRoll};
//! # let ldn = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! # let tky = Cal::new(vec![ndt(2017, 5, 3), ndt(2017, 5, 4), ndt(2017, 5, 5)], vec![5, 6]);
//! # let date = ndt(2017, 4, 28);  // Friday 28th April 2017
//! let ldn_tky = UnionCal::new(vec![ldn, tky], None);
//! let spot = ldn_tky.add_bus_days(&date, 2, true).unwrap();
//! // Monday 8th May 2017, observing all holidays.
//! assert_eq!(spot, ndt(2017, 5, 8));
//! ```
//!
//! Particularly when adjusting for FX transaction calendars the non-USD calendars may be used
//! for date determination but the US calendar is used to validate eligible settlement.
//! This is also a union of calendars but it is enforced via the `settlement_calendars` field.
//!
//! ```rust
//! # use rateslib::calendars::{UnionCal, Cal, ndt};
//! let tgt = Cal::new(vec![], vec![5, 6]);
//! let nyc = Cal::new(vec![ndt(2023, 6, 19)], vec![5, 6]);  // Juneteenth Holiday
//! let tgt__nyc = UnionCal::new(vec![tgt], vec![nyc].into());
//! ```
//!
//! The spot (T+2) date as measured from Friday 16th June 2023 ignores the US calendar for date
//! determination and allows Tuesday 20th June 2023 since the US holiday is on the Monday.
//!
//! ```rust
//! # use rateslib::calendars::{UnionCal, Cal, ndt, DateRoll};
//! # let tgt = Cal::new(vec![], vec![5, 6]);
//! # let nyc = Cal::new(vec![ndt(2023, 6, 19)], vec![5, 6]);  // Juneteenth Holiday
//! # let tgt__nyc = UnionCal::new(vec![tgt], vec![nyc].into());
//! let date = ndt(2023, 6, 16);  // Friday 16th June 2023
//! let spot = tgt__nyc.add_bus_days(&date, 2, true).unwrap();
//! // Tuesday 20th June 2023, ignoring the US holiday on Monday.
//! assert_eq!(spot, ndt(2023, 6, 20));
//! ```
//!
//! On the other hand as measured from Thursday 15th June 2023 the spot cannot be on the Monday
//! when `settlement` is enforced over the US calendar.
//!
//! ```rust
//! # use rateslib::calendars::{UnionCal, Cal, ndt, DateRoll};
//! # let tgt = Cal::new(vec![], vec![5, 6]);
//! # let nyc = Cal::new(vec![ndt(2023, 6, 19)], vec![5, 6]);  // Juneteenth Holiday
//! # let tgt__nyc = UnionCal::new(vec![tgt], vec![nyc].into());
//! let date = ndt(2023, 6, 15);  // Thursday 15th June 2023
//! let spot = tgt__nyc.add_bus_days(&date, 2, true).unwrap();
//! // Tuesday 20th June 2023, enforcing no settlement on US holiday.
//! assert_eq!(spot, ndt(2023, 6, 20));
//! ```
//!
//! If `settlement` is not enforced spot can be set as the Monday for this calendar, since it is
//! not a European holiday.
//!
//! ```rust
//! # use rateslib::calendars::{UnionCal, Cal, ndt, DateRoll};
//! # let tgt = Cal::new(vec![], vec![5, 6]);
//! # let nyc = Cal::new(vec![ndt(2023, 6, 19)], vec![5, 6]);  // Juneteenth Holiday
//! # let tgt__nyc = UnionCal::new(vec![tgt], vec![nyc].into());
//! # let date = ndt(2023, 6, 15);  // Thursday 15th June 2023
//! let spot = tgt__nyc.add_bus_days(&date, 2, false).unwrap();
//! // Monday 19th June 2023, ignoring the US holiday settlement requirement.
//! assert_eq!(spot, ndt(2023, 6, 19));
//! ```

mod adjuster;
mod calendars;
mod dcfs;
mod enums;
mod frequency;
mod rollday;
mod schedule;

mod serde;

pub(crate) mod py;

pub use crate::scheduling::{
    adjuster::{Adjuster, Adjustment, CalendarAdjustment},
    calendars::{Cal, UnionCal, NamedCal, Calendar, ndt, get_calendar_by_name, DateRoll},
    dcfs::Convention,
    frequency::Frequency,
    rollday::{get_eom, get_imm, get_roll, is_eom, is_imm, is_leap_year, RollDay},
};

pub(crate) use crate::scheduling::{
    dcfs::_get_convention_str
};
