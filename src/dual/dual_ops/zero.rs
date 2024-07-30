use crate::dual::dual::{Dual, Dual2, DualsOrF64};
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

impl Zero for DualsOrF64 {
    fn zero() -> DualsOrF64 {
        DualsOrF64::F64(0.0_f64)
    }

    fn is_zero(&self) -> bool {
        match self {
            DualsOrF64::F64(f) => *f == 0.0_f64,
            DualsOrF64::Dual(d) => *d == Dual::new(0.0, vec![]),
            DualsOrF64::Dual2(d) => *d == Dual2::new(0.0, vec![]),
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
        let d = DualsOrF64::Dual2(Dual2::zero());
        assert!(d.is_zero());
    }
}
