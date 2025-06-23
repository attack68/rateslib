use crate::dual::dual::{Dual, Dual2, Number};
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

impl Sum for Number {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Number>,
    {
        iter.fold(Number::F64(0.0_f64), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum() {
        let v = vec![
            Number::F64(2.5_f64),
            Number::Dual(Dual::new(1.5, vec!["x".to_string()])),
            Number::Dual(Dual::new(3.5, vec!["x".to_string()])),
        ];
        let s: Number = v.into_iter().sum();
        assert_eq!(
            s,
            Number::Dual(Dual::try_new(7.5, vec!["x".to_string()], vec![2.0]).unwrap())
        );
    }
}
