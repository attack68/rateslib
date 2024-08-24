use crate::dual::dual::{Dual, Dual2, Number};
use std::ops::{Add, Div, Mul, Sub};

pub trait NumericOps<T>:
    Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Sized + Clone
{
}
impl<'a, T: 'a> NumericOps<T> for &'a T where
    &'a T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
}
impl NumericOps<Dual> for Dual {}
impl NumericOps<Dual2> for Dual2 {}
impl NumericOps<f64> for f64 {}
impl NumericOps<Number> for Number {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fieldops() {
        fn test_ops<T>(a: &T, b: &T) -> T
        where
            for<'a> &'a T: NumericOps<T>,
        {
            &(a + b) - a
        }

        fn test_ops2<T>(a: T, b: T) -> T
        where
            T: NumericOps<T>,
        {
            (a.clone() + b) - a
        }

        let x = 1.0;
        let y = 2.0;
        let z = test_ops(&x, &y);
        println!("{:?}", z);

        let z = test_ops2(x, y);
        println!("{:?}", z);
    }
}
