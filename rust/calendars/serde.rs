use crate::calendars::calendar::{Cal, CalType, NamedCal, UnionCal};
use crate::json::JSON;

impl JSON for Cal {}
impl JSON for UnionCal {}
impl JSON for NamedCal {}

impl JSON for CalType {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::calendar::ndt;

    #[test]
    fn test_cal_json() {
        let hols = vec![ndt(2015, 9, 8), ndt(2015, 9, 10)];
        let hcal = Cal::new(hols, vec![5, 6]);
        let js = hcal.to_json().unwrap();
        let hcal2 = Cal::from_json(&js).unwrap();
        assert_eq!(hcal, hcal2);
    }

    #[test]
    fn test_union_cal_json() {
        let hols = vec![ndt(2015, 9, 8), ndt(2015, 9, 10)];
        let settle = vec![ndt(2015, 9, 11)];
        let hcal = Cal::new(hols, vec![5, 6]);
        let scal = Cal::new(settle, vec![5, 6]);
        let ucal = UnionCal::new(vec![hcal], vec![scal].into());
        let js = ucal.to_json().unwrap();
        let ucal2 = UnionCal::from_json(&js).unwrap();
        assert_eq!(ucal, ucal2);
    }

    #[test]
    fn test_named_cal_json() {
        let ncal = NamedCal::try_new("tgt,ldn|fed").unwrap();
        let js = ncal.to_json().unwrap();
        let ncal2 = NamedCal::from_json(&js).unwrap();
        assert_eq!(ncal, ncal2);
    }

    #[test]
    fn test_cal_type_json() {
        let cal = CalType::NamedCal(NamedCal::try_new("tgt,ldn|fed").unwrap());
        let js = cal.to_json().unwrap();
        let cal2 = CalType::from_json(&js).unwrap();
        assert_eq!(cal, cal2);
    }
}
