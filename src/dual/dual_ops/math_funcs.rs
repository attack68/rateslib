use crate::dual::dual::{Dual, Dual2};
use crate::dual::linalg::fouter11_;
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

impl MathFuncs for f64 {
    fn inv_norm_cdf(&self) -> Self {
        Normal::new(0.0, 1.0).unwrap().inverse_cdf(*self)
    }
    fn norm_cdf(&self) -> Self {
        Normal::new(0.0, 1.0).unwrap().cdf(*self)
    }
    fn exp(&self) -> Self {
        f64::exp(*self)
    }
    fn log(&self) -> Self {
        f64::ln(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = d1.exp();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.exp();
        let expected = Dual::try_new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0 * c, 2.0 * c],
        )
        .unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn log() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = d1.log();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.ln();
        let expected =
            Dual::try_new(c, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn exp2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = d1.exp();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.exp();
        let expected = Dual2::try_new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0 * c, 2.0 * c],
            vec![
                1.0_f64.exp() * 0.5,
                1.0_f64.exp(),
                1.0_f64.exp(),
                1.0_f64.exp() * 2.,
            ],
        )
        .unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn log2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = d1.log();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.ln();
        let expected = Dual2::try_new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            vec![-0.5, -1.0, -1.0, -2.0],
        )
        .unwrap();
        println!("{:?}", result.dual2);
        assert_eq!(result, expected);
    }
}
