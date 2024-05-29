pub mod bus;
pub mod nyc;

use crate::calendars::calendar::HolCal;
use std::collections::HashMap;
use chrono::NaiveDateTime;

fn get_weekmask_by_name(name: &str) -> Vec<u8> {
    let hmap: HashMap<&str, &[u8]> = HashMap::from([
        ("bus", bus::WEEKMASK),
        ("nyc", nyc::WEEKMASK),
    ]);
    hmap.get(name).unwrap().to_vec()
}

fn get_holidays_by_name(name: &str) -> Vec<NaiveDateTime> {
    let hmap: HashMap<&str, &[&str]> = HashMap::from([
        ("bus", bus::HOLIDAYS),
        ("nyc", nyc::HOLIDAYS),
    ]);
    hmap.get(name).unwrap().iter().map(|x| NaiveDateTime::parse_from_str(x, "%Y-%m-%d %H:%M:%S").unwrap()).collect()
}

pub fn get_calendar_by_name(name: &str) -> HolCal {
    HolCal::new(get_holidays_by_name(name), get_weekmask_by_name(name))
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_weekmask() {
        let result = get_weekmask_by_name("bus");
        assert_eq!(result, vec![5,6]);
    }

    #[test]
    fn test_get_holidays() {
        let result = get_holidays_by_name("bus");
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_get_holcal() {
        let result = get_calendar_by_name("bus");
        let expected = HolCal::new(vec![], vec![5, 6]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_nyc() {
        let cal = get_calendar_by_name("nyc");
        assert!(cal.is_holiday(&NaiveDateTime::parse_from_str("2024-11-11 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()));
    }

}