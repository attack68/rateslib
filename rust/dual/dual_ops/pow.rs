use crate::dual::dual::{Dual, Dual2, Vars};
use crate::dual::linalg::fouter11_;
use num_traits::Pow;
use std::sync::Arc;

impl Pow<f64> for Dual {
    type Output = Dual;
    fn pow(self, power: f64) -> Dual {
        Dual {
            real: self.real.pow(power),
            vars: self.vars,
            dual: self.dual * power * self.real.pow(power - 1.0),
        }
    }
}

impl Pow<f64> for &Dual {
    type Output = Dual;
    fn pow(self, power: f64) -> Dual {
        Dual {
            real: self.real.pow(power),
            vars: Arc::clone(self.vars()),
            dual: &self.dual * power * self.real.pow(power - 1.0),
        }
    }
}

impl Pow<f64> for Dual2 {
    type Output = Dual2;
    fn pow(self, power: f64) -> Dual2 {
        let coeff = power * self.real.powf(power - 1.);
        let coeff2 = 0.5 * power * (power - 1.) * self.real.powf(power - 2.);
        let beta_cross = fouter11_(&self.dual.view(), &self.dual.view());
        Dual2 {
            real: self.real.powf(power),
            vars: self.vars,
            dual: self.dual * coeff,
            dual2: self.dual2 * coeff + beta_cross * coeff2,
        }
    }
}

impl Pow<f64> for &Dual2 {
    type Output = Dual2;
    fn pow(self, power: f64) -> Dual2 {
        let coeff = power * self.real.powf(power - 1.);
        let coeff2 = 0.5 * power * (power - 1.) * self.real.powf(power - 2.);
        let beta_cross = fouter11_(&self.dual.view(), &self.dual.view());
        Dual2 {
            real: self.real.powf(power),
            vars: Arc::clone(self.vars()),
            dual: &self.dual * coeff,
            dual2: &self.dual2 * coeff + beta_cross * coeff2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn inv() {
        let d1 = Dual::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        )
        .unwrap();
        let result = d1.clone() * d1.pow(-1.0);
        let expected = Dual::new(1.0, vec![]);
        assert!(result == expected)
    }

    #[test]
    fn pow_ref() {
        let d1 = Dual::new(3.0, vec!["x".to_string()]);
        let d2 = (&d1).pow(2.0);
        assert_eq!(d2.real, 9.0);
        assert_eq!(d2.dual, Array1::from_vec(vec![6.0]));
    }

    #[test]
    fn pow_ref2() {
        let d1 = Dual2::new(3.0, vec!["x".to_string()]);
        let d2 = (&d1).pow(2.0);
        assert_eq!(d2.real, 9.0);
        assert_eq!(d2.dual, Array1::from_vec(vec![6.0]));
    }

    #[test]
    fn inv2() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        )
        .unwrap();
        let result = d1.clone() * d1.pow(-1.0);
        let expected = Dual2::new(1.0, vec![]);
        assert_eq!(result, expected)
    }
}
