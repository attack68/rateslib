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
