use crate::dual::dual::{Dual, Dual2};
use std::ops::{Add, Div, Mul, Sub};

pub trait FieldOps<T>:
    Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Sized + Clone
{
}
impl<'a, T: 'a> FieldOps<T> for &'a T where
    &'a T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
}
impl FieldOps<Dual> for Dual {}
impl FieldOps<Dual2> for Dual2 {}
impl FieldOps<f64> for f64 {}


mod tests {
    use crate::dual::dual_ops::field_ops::FieldOps;

    #[test]
    fn test_fieldops() {
        fn test_ops<T>(a: &T, b: &T) -> T
        where
                for<'a> &'a T: FieldOps<T>,
        {
            &(a + b) - a
        }

        fn test_ops2<T>(a: T, b: T) -> T
        where
            T: FieldOps<T>,
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
