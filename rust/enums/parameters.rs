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
