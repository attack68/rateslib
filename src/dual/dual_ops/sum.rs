use crate::dual::dual::{Dual, Dual2, DualsOrF64};
use std::iter::Sum;

impl Sum for Dual {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Dual>,
    {
        iter.fold(Dual::new(0.0, [].to_vec()), |acc, x| acc + x)
    }
}

impl Sum for Dual2 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Dual2>,
    {
        iter.fold(Dual2::new(0.0, Vec::new()), |acc, x| acc + x)
    }
}

impl Sum for DualsOrF64 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = DualsOrF64>,
    {
        iter.fold(DualsOrF64::F64(0.0_f64), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum() {
        let v = vec![
            DualsOrF64::F64(2.5_f64),
            DualsOrF64::Dual(Dual::new(1.5, vec!["x".to_string()])),
            DualsOrF64::Dual(Dual::new(3.5, vec!["x".to_string()])),
        ];
        let s: DualsOrF64 = v.into_iter().sum();
        assert_eq!(s, DualsOrF64::Dual(Dual::try_new(7.5, vec!["x".to_string()], vec![2.0]).unwrap()));
    }
}
