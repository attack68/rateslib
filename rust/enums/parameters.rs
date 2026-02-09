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

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Specifier for date adjustment rules.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum FloatFixingMethod {
    /// RFR periods are settled with cashflow dates determined (separately as part of a Schedule) with a lag.
    RFRPaymentDelay {},
    /// RFR fixings and associated DCFs use values taken from 'n' business days prior.
    RFRObservationShift(i32),
    /// The final 'n' RFR fixings' values are taken as the most recent published value.
    RFRLockout(i32),
    /// RFR fixings use values taken from 'n' business days prior (no DCF shift).
    RFRLookback(i32),
    /// Uses arithmetic averaging instead compounding on the RFRPaymentDelay method.
    RFRPaymentDelayAverage {},
    /// Uses arithmetic averaging instead compounding on the RFRObservationShift method.
    RFRObservationShiftAverage(i32),
    /// Uses arithmetic averaging instead compounding on the RFRLockout method.
    RFRLockoutAverage(i32),
    /// Uses arithmetic averaging instead compounding on the RFRLookback method.
    RFRLookbackAverage(i32),
    /// Uses a tenor IBOR type rate calculation with the fixing lagged by 'n' business days.
    IBOR(i32),
}

impl FloatFixingMethod {
    /// Return a fixing lag parameter associated with the variant.
    pub fn method_param(&self) -> i32 {
        match self {
            FloatFixingMethod::RFRPaymentDelay {}
            | FloatFixingMethod::RFRPaymentDelayAverage {} => 0_i32,
            FloatFixingMethod::RFRObservationShift(param)
            | FloatFixingMethod::RFRObservationShiftAverage(param)
            | FloatFixingMethod::RFRLookback(param)
            | FloatFixingMethod::RFRLookbackAverage(param)
            | FloatFixingMethod::RFRLockout(param)
            | FloatFixingMethod::RFRLockoutAverage(param)
            | FloatFixingMethod::IBOR(param) => *param,
        }
    }
}

/// Enumerable type for index base determination on each Period in a Leg.
#[pyclass(module = "rateslib.rs", eq, eq_int, hash, frozen, from_py_object)]
#[derive(Debug, Hash, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum LegIndexBase {
    /// Set the index base on every period as the initial base date of the Leg.
    Initial = 0,
    /// Set the index base date of each period successively as the reference value for the
    // previous period.
    PeriodOnPeriod = 1,
}
