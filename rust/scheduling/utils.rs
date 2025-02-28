use crate::calendars::RollDay;
use crate::scheduling::enums::{Frequency, RollDayCategory, ValidateSchedule};
use chrono::prelude::*;

/// Infer a RollDay from given dates of a regular schedule.
///
/// Days before month end will only be valid if they match by day.
/// Month end options are controlled by the cases and the ``eom`` parameter.
/// If any date is 31 and the other date is EoM then '31' is returned.
/// If both dates are EoM but neither is 31 and ``eom`` is True then 'EoM' is returned.
/// If both dates are EoM but neither is 31 and ``eom`` is False then max(day) is returned.
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
//         let mut roll_;
//         match roll {
//             None => {
//                 let rollday = get_unadjusted_rollday(&ueffective, &utermination, eom);
//                 match rollday {
//                     None => return ValidateSchedule::Invalid {error: "Roll day could not be inferred from given dates.".to_string() }
//                     Some(v) =>
//                 }
//             }
//         }
//     }
// }

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;

    #[test]
    fn test_get_unadjusted_rollday() {
        // take two dates and ensure that the inferred roll is accurate give the input
        let options: Vec<(NaiveDateTime, NaiveDateTime, bool, RollDay)> = vec![
            (
                ndt(2000, 1, 15),
                ndt(2000, 2, 15),
                false,
                RollDay::Int { day: 15 },
            ),
            (
                ndt(2022, 2, 28),
                ndt(2022, 9, 30),
                false,
                RollDay::Int { day: 30 },
            ),
            (ndt(2022, 2, 28), ndt(2022, 9, 30), true, RollDay::EoM {}),
            (
                ndt(2024, 2, 29),
                ndt(2022, 9, 30),
                false,
                RollDay::Int { day: 30 },
            ),
            (ndt(2024, 2, 29), ndt(2022, 9, 30), true, RollDay::EoM {}),
            (ndt(2024, 4, 30), ndt(2022, 9, 30), true, RollDay::EoM {}),
            (
                ndt(2024, 4, 30),
                ndt(2022, 9, 30),
                false,
                RollDay::Int { day: 30 },
            ),
            (
                ndt(2024, 5, 31),
                ndt(2022, 9, 30),
                false,
                RollDay::Int { day: 31 },
            ),
            (
                ndt(2024, 5, 31),
                ndt(2022, 9, 30),
                true,
                RollDay::Int { day: 31 },
            ),
        ];
        for option in options.iter() {
            assert_eq!(
                option.3,
                get_unadjusted_rollday(&option.0, &option.1, option.2).unwrap()
            );
        }
    }

    #[test]
    fn test_get_unadjusted_rollday_invalid() {
        // take two dates and ensure that the inferred roll is accurate give the input
        let options: Vec<(NaiveDateTime, NaiveDateTime, bool)> = vec![
            (ndt(2024, 2, 28), ndt(2022, 9, 30), false), // is leap
            (ndt(2024, 2, 28), ndt(2022, 9, 30), true),  // is leap
            (ndt(2024, 2, 10), ndt(2022, 9, 15), false), // day unaligned
            (ndt(2024, 4, 29), ndt(2022, 9, 30), true),  // day unaligned
        ];
        for option in options.iter() {
            assert_eq!(None, get_unadjusted_rollday(&option.0, &option.1, option.2));
        }
    }
}
