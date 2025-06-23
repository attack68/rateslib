use crate::dual::dual::{Dual, Dual2, Number};
use num_traits::Zero;

impl Zero for Dual {
    fn zero() -> Dual {
        Dual::new(0.0, Vec::new())
    }

    fn is_zero(&self) -> bool {
        *self == Dual::new(0.0, Vec::new())
    }
}

impl Zero for Dual2 {
    fn zero() -> Dual2 {
        Dual2::new(0.0, Vec::new())
    }

    fn is_zero(&self) -> bool {
        *self == Dual2::new(0.0, Vec::new())
    }
}

impl Zero for Number {
    fn zero() -> Number {
        Number::F64(0.0_f64)
    }

    fn is_zero(&self) -> bool {
        match self {
            Number::F64(f) => *f == 0.0_f64,
            Number::Dual(d) => *d == Dual::new(0.0, vec![]),
            Number::Dual2(d) => *d == Dual2::new(0.0, vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_zero_() {
        assert!(Dual::zero().is_zero())
    }

    #[test]
    fn is_zero2() {
        let d = Dual2::zero();
        assert!(d.is_zero());
    }

    #[test]
    fn is_zero_enum() {
        let d = Number::Dual2(Dual2::zero());
        assert!(d.is_zero());
    }
}
