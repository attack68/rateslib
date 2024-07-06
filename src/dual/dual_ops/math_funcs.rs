use crate::dual::dual::{Dual, Dual2};
use crate::dual::linalg_f64::fouter11_;
use num_traits::Pow;
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::PI;
use std::sync::Arc;

/// Functions for common mathematical operations.
pub trait MathFuncs {
    /// Return the exponential of a value.
    fn exp(&self) -> Self;
    /// Return the natural logarithm of a value.
    fn log(&self) -> Self;
    /// Return the standard normal cumulative distribution function of a value.
    fn norm_cdf(&self) -> Self;
    /// Return the inverse standard normal cumulative distribution function of a value.
    fn inv_norm_cdf(&self) -> Self;
}

impl MathFuncs for Dual {
    fn exp(&self) -> Self {
        let c = self.real.exp();
        Dual {
            real: c,
            vars: Arc::clone(&self.vars),
            dual: c * &self.dual,
        }
    }
    fn log(&self) -> Self {
        Dual {
            real: self.real.ln(),
            vars: Arc::clone(&self.vars),
            dual: (1.0 / self.real) * &self.dual,
        }
    }
    fn norm_cdf(&self) -> Self {
        let n = Normal::new(0.0, 1.0).unwrap();
        let base = n.cdf(self.real);
        let scalar = 1.0 / (2.0 * PI).sqrt() * (-0.5_f64 * self.real.pow(2.0_f64)).exp();
        Dual {
            real: base,
            vars: Arc::clone(&self.vars),
            dual: scalar * &self.dual,
        }
    }
    fn inv_norm_cdf(&self) -> Self {
        let n = Normal::new(0.0, 1.0).unwrap();
        let base = n.inverse_cdf(self.real);
        let scalar = (2.0 * PI).sqrt() * (0.5_f64 * base.pow(2.0_f64)).exp();
        Dual {
            real: base,
            vars: Arc::clone(&self.vars),
            dual: scalar * &self.dual,
        }
    }
}

impl MathFuncs for Dual2 {
    fn exp(&self) -> Self {
        let c = self.real.exp();
        Dual2 {
            real: c,
            vars: Arc::clone(&self.vars),
            dual: c * &self.dual,
            dual2: c * (&self.dual2 + 0.5 * fouter11_(&self.dual.view(), &self.dual.view())),
        }
    }
    fn log(&self) -> Self {
        let scalar = 1.0 / self.real;
        Dual2 {
            real: self.real.ln(),
            vars: Arc::clone(&self.vars),
            dual: scalar * &self.dual,
            dual2: scalar * &self.dual2
                - fouter11_(&self.dual.view(), &self.dual.view()) * 0.5 * (scalar * scalar),
        }
    }
    fn norm_cdf(&self) -> Self {
        let n = Normal::new(0.0, 1.0).unwrap();
        let base = n.cdf(self.real);
        let scalar = 1.0 / (2.0 * PI).sqrt() * (-0.5_f64 * self.real.pow(2.0_f64)).exp();
        let scalar2 = scalar * -self.real;
        let cross_beta = fouter11_(&self.dual.view(), &self.dual.view());
        Dual2 {
            real: base,
            vars: Arc::clone(&self.vars),
            dual: scalar * &self.dual,
            dual2: scalar * &self.dual2 + 0.5_f64 * scalar2 * cross_beta,
        }
    }
    fn inv_norm_cdf(&self) -> Self {
        let n = Normal::new(0.0, 1.0).unwrap();
        let base = n.inverse_cdf(self.real);
        let scalar = (2.0 * PI).sqrt() * (0.5_f64 * base.pow(2.0_f64)).exp();
        let scalar2 = scalar.pow(2.0_f64) * base;
        let cross_beta = fouter11_(&self.dual.view(), &self.dual.view());
        Dual2 {
            real: base,
            vars: Arc::clone(&self.vars),
            dual: scalar * &self.dual,
            dual2: scalar * &self.dual2 + 0.5_f64 * scalar2 * cross_beta,
        }
    }
}
