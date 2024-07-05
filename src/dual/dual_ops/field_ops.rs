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
