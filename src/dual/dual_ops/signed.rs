use crate::dual::dual::{Dual, Dual2};
use num_traits::Signed;
use std::sync::Arc;

/// Sign for `Dual` is evaluated in terms of the `real` component.
impl Signed for Dual {
    /// Determine the absolute value of `Dual`.
    ///
    /// If `real` is negative the returned `Dual` will negate both its `real` value and
    /// `dual`.
    ///
    /// <div class="warning">This behaviour is undefined at zero. The derivative of the `abs` function is
    /// not defined there and care needs to be taken when implying gradients.</div>
    fn abs(&self) -> Self {
        if self.real > 0.0 {
            Dual {
                real: self.real,
                vars: Arc::clone(&self.vars),
                dual: self.dual.clone(),
            }
        } else {
            Dual {
                real: -self.real,
                vars: Arc::clone(&self.vars),
                dual: -1.0 * &self.dual,
            }
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other {
            Dual::new(0.0, Vec::new())
        } else {
            self - other
        }
    }

    fn signum(&self) -> Self {
        Dual::new(self.real.signum(), Vec::new())
    }

    fn is_positive(&self) -> bool {
        self.real.is_sign_positive()
    }

    fn is_negative(&self) -> bool {
        self.real.is_sign_negative()
    }
}

impl Signed for Dual2 {
    fn abs(&self) -> Self {
        if self.real > 0.0 {
            Dual2 {
                real: self.real,
                vars: Arc::clone(&self.vars),
                dual: self.dual.clone(),
                dual2: self.dual2.clone(),
            }
        } else {
            Dual2 {
                real: -self.real,
                vars: Arc::clone(&self.vars),
                dual: -1.0 * &self.dual,
                dual2: -1.0 * &self.dual2,
            }
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other {
            Dual2::new(0.0, Vec::new())
        } else {
            self - other
        }
    }

    fn signum(&self) -> Self {
        Dual2::new(self.real.signum(), Vec::new())
    }

    fn is_positive(&self) -> bool {
        self.real.is_sign_positive()
    }

    fn is_negative(&self) -> bool {
        self.real.is_sign_negative()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn signed() {
        let d1 = Dual::new(3.0, vec!["x".to_string()]);
        let d2 = Dual::new(-2.0, vec!["x".to_string()]);

        assert!(d2.is_negative());
        assert!(d1.is_positive());
        assert_eq!(d2.signum(), -1.0 * Dual::one());
        assert_eq!(d1.signum(), Dual::one());
        assert_eq!(d1.abs_sub(&d2), Dual::new(5.0, Vec::new()));
        assert_eq!(d2.abs_sub(&d1), Dual::zero());
    }

    #[test]
    fn signed_2() {
        let d1 = Dual2::new(3.0, vec!["x".to_string()]);
        let d2 = Dual2::new(-2.0, vec!["x".to_string()]);

        assert!(d2.is_negative());
        assert!(d1.is_positive());
        assert_eq!(d2.signum(), -1.0 * Dual2::one());
        assert_eq!(d1.signum(), Dual2::one());
        assert_eq!(d1.abs_sub(&d2), Dual2::new(5.0, Vec::new()));
        assert_eq!(d2.abs_sub(&d1), Dual2::zero());
    }

    #[test]
    fn abs() {
        let d1 = Dual::try_new(
            -2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
            .unwrap();
        let result = d1.abs();
        let expected = Dual::try_new(
            2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
        )
            .unwrap();
        assert_eq!(result, expected);

        let result = d1.abs();
        assert_eq!(result, expected);
    }

    #[test]
    fn abs2() {
        let d1 = Dual2::try_new(
            -2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
            .unwrap();
        let result = d1.abs();
        let expected = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
            Vec::new(),
        )
            .unwrap();
        assert_eq!(result, expected);

        let result = result.abs();
        assert_eq!(result, expected);
    }
}