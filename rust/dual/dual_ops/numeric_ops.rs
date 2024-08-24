use crate::dual::dual::{Dual, Dual2, Number};
use std::ops::{Add, Div, Mul, Sub};

pub trait NumberOps<T>:
    Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Sized + Clone
{
}
impl<'a, T: 'a> NumberOps<T> for &'a T where
    &'a T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
}
impl NumberOps<Dual> for Dual {}
impl NumberOps<Dual2> for Dual2 {}
impl NumberOps<f64> for f64 {}
impl NumberOps<Number> for Number {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fieldops() {
        fn test_ops<T>(a: &T, b: &T) -> T
        where
            for<'a> &'a T: NumberOps<T>,
        {
            &(a + b) - a
        }

        fn test_ops2<T>(a: T, b: T) -> T
        where
            T: NumberOps<T>,
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
