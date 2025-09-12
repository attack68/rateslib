//! Create a business day [`Calendar`], instrument [`Schedule`] and perform financial date manipulation.
//!
//! The purpose of this module is to provide objects which are capable of replicating all of the
//! complexities of financial instrument specification, including examples such as;
//! - FX spot determination including all of the various currency pair rules.
//! - Business day calendar combination for multi-currency derivatives.
//! - Standard schedule generation including all of the accrual and payment [`Adjuster`] rules, like
//!   *modified following*, CDS's unadjusted last period etc.
//! - Inference for stub dates and monthly [`RollDay`] when utilising a UI which extends to users
//!   being allowed to supply unknown or ambiguous parameters.
//!
//! # Calendars and Date Adjustment
//!
//! ## Calendars
//!
//! *Rateslib* provides three calendar types: [`Cal`], [`UnionCal`] and [`NamedCal`] and the container
//! enum [`Calendar`]. These are based on simple holiday and weekend specification and union rules
//! for combinations. Some common calendars are implemented directly by name, and can be combined
//! with string parsing syntax.
//!
//! All calendars implement the [`DateRoll`] trait which provide simple date adjustment, which
//! *rateslib* calls **rolling**. This involves moving forward or backward from non-business days
//! (or non-settleable days) to specific **business days** or **settleable business days**.
//!
//! ### Example
//! This example creates a business day calendar defining Saturday and Sunday weekends and a
//! specific holiday (the Early May UK Bank Holiday). It uses a date rolling method to
//! manipulate Saturday 29th April 2017 under the *'following'* and *'modified following'* rules.
//! ```rust
//! # use rateslib::scheduling::{Cal, ndt, DateRoll};
//! let cal = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! assert_eq!(ndt(2017, 5, 2), cal.roll_forward_bus_day(&ndt(2017, 4, 29)));
//! assert_eq!(ndt(2017, 4, 28), cal.roll_mod_forward_bus_day(&ndt(2017, 4, 29)));
//! ```
//!
//! ## Date Adjustment
//!
//! Date adjustment allows for a more complicated set of rules than simple date rolling.
//! The [`Adjuster`] is an enum which defines the implementation of all of these rules and may
//! be extended in the future if more rules are required for more complex instruments. It
//! implements the [`Adjustment`] trait requiring some object capable of performing [`DateRoll`] to
//! define the operations.
//!
//! All [`Calendar`] types implement the [`CalendarAdjustment`] trait which permits date
//! adjustment when an [`Adjuster`] is cross-provided.
//!
//! ### Example
//! This example performs the complex rule of adjusting a given date forward by 5 calendar days
//! and then rolling that result forward to the next settleable business day.
//! ```rust
//! # use rateslib::scheduling::{Cal, ndt, Adjuster, CalendarAdjustment};
//! # let cal = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! let adjuster = Adjuster::CalDaysLagSettle(5);
//! assert_eq!(ndt(2017, 5, 2), cal.adjust(&ndt(2017, 4, 27), &adjuster));
//! assert_eq!(ndt(2017, 5, 2), cal.adjust(&ndt(2017, 4, 24), &adjuster));
//! ```
//!
//! # Schedules
//!
//! A [`Schedule`] is an ordered and patterned array of periods and dates.
//!
//! All [`Schedule`] objects in *rateslib* are centered about the definition of their [`Frequency`],
//! which is an enum describing a regular period of time. Certain [`Frequency`] variants have
//! additional information to fully parametrise them. For example a [`Frequency::BusDays`](Frequency) variant
//! requires a [`Calendar`] to define its valid days, and a [`Frequency::Months`](Frequency) variant requires
//! a [`RollDay`] to define the day in the month that separates its periods.
//!
//! The [`Frequency`] implements the [`Scheduling`] trait which allows periods and stubs to be
//! defined, alluding to the documented definition of **regular** and **irregular** schedules as
//! well as permitting the pattern of periods that can form a valid [`Schedule`].
//!
//! ### Example
//! This example creates a new [`Schedule`] by inferring that it can be constructed as a **regular schedule**
//! (one without stubs) if the [`RollDay`] is asserted to be the [`RollDay::IMM`](RollDay) variant.
//! Without an *IMM* roll-day this schedule would be irregular with a short front stub.
//! ```rust
//! # use rateslib::scheduling::{Cal, ndt, Adjuster, Frequency, Schedule, RollDay, StubInference, Calendar};
//! # let cal = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! let schedule = Schedule::try_new_inferred(
//!    ndt(2024, 3, 20),                        // effective
//!    ndt(2025, 9, 17),                        // termination
//!    Frequency::Months{number:3, roll: None}, // frequency
//!    None,                                    // front_stub
//!    None,                                    // back_stub
//!    Calendar::Cal(cal),                      // calendar
//!    Adjuster::ModifiedFollowing{},           // accrual_adjuster
//!    Adjuster::BusDaysLagSettle(2),           // payment_adjuster
//!    Adjuster::Actual{},                      // payment_adjuster2
//!    None,                                    // payment_adjuster3
//!    false,                                   // eom
//!    Some(StubInference::ShortFront),         // stub_inference
//! );
//! # let schedule = schedule.unwrap();
//! assert_eq!(schedule.frequency, Frequency::Months{number:3, roll: Some(RollDay::IMM())});
//! assert!(schedule.is_regular());
//! ```
//! The next example creates a new [`Schedule`] by inferring that its `termination` is an adjusted
//! end-of-month date, and therefore its [`RollDay`] is asserted to be the [`RollDay::Day(31)`](RollDay)
//! variant, and its `utermination` is therefore 30th November and it infers a `ufront_stub` correctly
//! as 31st May 2025.
//! ```rust
//! # use rateslib::scheduling::{Cal, ndt, Adjuster, Frequency, Schedule, RollDay, StubInference, Calendar};
//! # let cal = Cal::new(vec![ndt(2017, 5, 1)], vec![5, 6]);
//! let schedule = Schedule::try_new_inferred(
//!    ndt(2025, 4, 15),                        // effective
//!    ndt(2025, 11, 28),                       // termination
//!    Frequency::Months{number:3, roll: None}, // frequency
//!    None,                                    // front_stub
//!    None,                                    // back_stub
//!    Calendar::Cal(cal),                      // calendar
//!    Adjuster::ModifiedFollowing{},           // accrual_adjuster
//!    Adjuster::BusDaysLagSettle(2),           // payment_adjuster
//!    Adjuster::Actual{},                      // payment_adjuster2
//!    None,                                    // payment_adjuster3
//!    true,                                    // eom
//!    Some(StubInference::ShortFront),         // stub_inference
//! );
//! # let schedule = schedule.unwrap();
//! assert_eq!(schedule.frequency, Frequency::Months{number:3, roll: Some(RollDay::Day(31))});
//! assert_eq!(schedule.utermination, ndt(2025, 11, 30));
//! assert_eq!(schedule.ufront_stub, Some(ndt(2025, 5, 31)));
//! ```

mod calendars;
mod convention;
mod frequency;
mod schedule;

mod serde;

pub(crate) mod py;

pub use crate::scheduling::{
    calendars::{
        ndt, Adjuster, Adjustment, Cal, Calendar, CalendarAdjustment, DateRoll, NamedCal, UnionCal,
    },
    convention::Convention,
    frequency::{Frequency, Imm, RollDay, Scheduling},
    schedule::{Schedule, StubInference},
};
pub(crate) use crate::scheduling::{frequency::get_unadjusteds, py::PyAdjuster};
