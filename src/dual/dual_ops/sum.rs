use crate::dual::dual::{Dual, Dual2};
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
