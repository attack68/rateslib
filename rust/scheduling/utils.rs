use crate::calendars::RollDay;
use crate::scheduling::enums::{Frequency, RollDayCategory, ValidateSchedule};
use chrono::prelude::*;

/// Return if a given year is a leap year.
pub(crate) fn is_leap_year(year: i32) -> bool {
    NaiveDate::from_ymd_opt(year, 2, 29).is_some()
}

/// Infer a RollDay from given dates of a regular schedule.
pub(crate) fn get_unadjusted_rollday(
    ueffective: &NaiveDateTime,
    utermination: &NaiveDateTime,
    eom: bool,
) -> Option<RollDay> {
    if ueffective.day() < 28 || utermination.day() < 28 {
        // if day < 28 both days must match since only evaluating numeric day matching (not IMM)
        if ueffective.day() == utermination.day() {
            Some(RollDay::Int {
                day: ueffective.day(),
            })
        } else {
            None
        }
    } else {
        // will only evaluate using EoM categories sicen day >= 28 in both cases
        let e_cat = RollDayCategory::from_date(ueffective);
        let t_cat = RollDayCategory::from_date(utermination);
        e_cat.get_rollday_from_eom_categories(&t_cat, eom)
    }
}

// pub(crate) fn check_unadjusted_regular_schedule(
//     ueffective: &NaiveDateTime,
//     utermination: &NaiveDateTime,
//     frequency: &Frequency,
//     eom: bool,
//     roll: Option<RollDay>,
// ) -> ValidateSchedule {
//     if !frequency.is_divisible(ueffective, utermination) {
//         ValidateSchedule::Invalid {
//             error: "Months date separation not aligned with frequency.".to_string(),
//         }
//     } else {
//     }
// }

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;

    #[test]
    fn test_is_leap() {
        assert_eq!(true, is_leap_year(2024));
        assert_eq!(false, is_leap_year(2022));
    }

    #[test]
    fn test_get_unadjusted_rollday() {

    }
}
