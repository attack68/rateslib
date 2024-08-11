use crate::dual::dual::{Dual, Dual2, DualsOrF64};
use std::cmp::Ordering;

/// Compares `Dual` by `real` component only.
impl PartialOrd<Dual> for Dual {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl PartialOrd<f64> for Dual {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.real.partial_cmp(other)
    }
}

impl PartialOrd<f64> for Dual2 {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.real.partial_cmp(other)
    }
}

impl PartialOrd<Dual2> for Dual2 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl PartialOrd<Dual> for f64 {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {
        self.partial_cmp(&other.real)
    }
}

impl PartialOrd<Dual2> for f64 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {
        self.partial_cmp(&other.real)
    }
}

impl PartialOrd<DualsOrF64> for DualsOrF64 {
    fn partial_cmp(&self, other: &DualsOrF64) -> Option<Ordering> {
        match (self, other) {
            (DualsOrF64::F64(f), DualsOrF64::F64(f2)) => f.partial_cmp(f2),
            (DualsOrF64::F64(f), DualsOrF64::Dual(d2)) => f.partial_cmp(d2),
            (DualsOrF64::F64(f), DualsOrF64::Dual2(d2)) => f.partial_cmp(d2),
            (DualsOrF64::Dual(d), DualsOrF64::F64(f2)) => d.partial_cmp(f2),
            (DualsOrF64::Dual(d), DualsOrF64::Dual(d2)) => d.partial_cmp(d2),
            (DualsOrF64::Dual(_), DualsOrF64::Dual2(_)) => {
                panic!("Cannot mix dual types: Dual compare Dual2")
            }
            (DualsOrF64::Dual2(d), DualsOrF64::F64(f2)) => d.partial_cmp(f2),
            (DualsOrF64::Dual2(_), DualsOrF64::Dual(_)) => {
                panic!("Cannot mix dual types: Dual2 compare Dual")
            }
            (DualsOrF64::Dual2(d), DualsOrF64::Dual2(d2)) => d.partial_cmp(d2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ord() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        assert!(d1 < 2.0);
        assert!(d1 > 0.5);
        assert!(d1 <= 1.0);
        assert!(d1 >= 1.0);
        assert!(1.0 <= d1);
        assert!(1.0 >= d1);
        assert!(2.0 > d1);
        assert!(0.5 < d1);
        let d2 = Dual::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        assert!(d2 > d1);
        assert!(d1 < d2);
        let d3 = Dual::try_new(1.0, vec!["v3".to_string()], vec![10.0]).unwrap();
        assert!(d1 >= d3);
        assert!(d1 <= d3);
    }

    #[test]
    fn ord2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        assert!(d1 < 2.0);
        assert!(d1 > 0.5);
        assert!(d1 <= 1.0);
        assert!(d1 >= 1.0);
        assert!(1.0 <= d1);
        assert!(1.0 >= d1);
        assert!(2.0 > d1);
        assert!(0.5 < d1);
        let d2 = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        assert!(d2 > d1);
        assert!(d1 < d2);
        let d3 = Dual2::try_new(1.0, vec!["v3".to_string()], vec![10.0], Vec::new()).unwrap();
        assert!(d1 >= d3);
        assert!(d1 <= d3);
    }

    #[test]
    fn test_enum() {
        let d = DualsOrF64::Dual(Dual::new(2.0, vec!["x".to_string()]));
        let d2 = DualsOrF64::Dual(Dual::new(3.0, vec!["x".to_string()]));
        assert!(d <= d2)
    }

    #[test]
    fn test_cross_enum_eq() {
        let f = DualsOrF64::F64(2.5_f64);
        let d = DualsOrF64::Dual(Dual::new(3.5_f64, vec![]));
        assert!(f <= d);
    }

    #[test]
    #[should_panic]
    fn test_cross_enum_eq_error() {
        let d2 = DualsOrF64::Dual2(Dual2::new(2.5_f64, vec![]));
        let d = DualsOrF64::Dual(Dual::new(2.5_f64, vec![]));
        assert!(d <= d2);
    }
}
