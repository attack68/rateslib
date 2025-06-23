use crate::dual::dual::{Dual, Dual2};
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
}
