use crate::dual::dual::{Dual, Dual2};
use num_traits::One;

impl One for Dual {
    fn one() -> Dual {
        Dual::new(1.0, Vec::new())
    }
}

impl One for Dual2 {
    fn one() -> Dual2 {
        Dual2::new(1.0, Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one() {
        let d = Dual::one();
        assert_eq!(d, Dual::new(1.0, vec![]));
    }

    #[test]
    fn one2() {
        let d = Dual2::one();
        assert_eq!(d, Dual2::new(1.0, vec![]));
    }

}
