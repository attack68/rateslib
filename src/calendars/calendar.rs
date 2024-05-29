use chrono::prelude::*;
use indexmap::set::IndexSet;

pub struct HolCal {
    holidays: IndexSet<NaiveDateTime>,
}

impl HolCal {
    pub fn new(holidays: IndexSet<NaiveDateTime>) -> Self {
        HolCal {holidays}
    }

    pub fn is_holiday(&self, date: &NaiveDateTime) -> bool {
        let i = self.holidays.get_index_of(date);
        match i {
            Some(index) => true,
            None => false,
        }
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_holiday() {
        let hols = vec![
            NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
            NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ];
        let h = HolCal::new(IndexSet::from_iter(hols));
        let test = NaiveDateTime::parse_from_str("2015-09-07 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let test2 = NaiveDateTime::parse_from_str("2015-09-10 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(h.is_holiday(&test));
        assert!(!h.is_holiday(&test2));
    }
}
