use crate::dual::dual::{Dual, Dual2, Vars, VarsRelationship};
use crate::dual::enums::Number;
use crate::dual::linalg::fouter11_;
use num_traits::Pow;
use std::sync::Arc;

impl Pow<&Dual> for f64 {
    type Output = Dual;
    fn pow(self, power: &Dual) -> Self::Output {
        Dual {
            real: self.pow(power.real),
            vars: Arc::clone(power.vars()),
            dual: &power.dual * self.pow(power.real) * self.ln(),
        }
    }
}

impl Pow<Dual> for f64 {
    type Output = Dual;
    fn pow(self, power: Dual) -> Self::Output {
        Dual {
            real: self.pow(power.real),
            vars: power.vars,
            dual: power.dual * self.pow(power.real) * self.ln(),
        }
    }
}

impl Pow<&Dual> for &f64 {
    type Output = Dual;
    fn pow(self, power: &Dual) -> Self::Output {
        (*self).pow(power)
    }
}

impl Pow<Dual> for &f64 {
    type Output = Dual;
    fn pow(self, power: Dual) -> Self::Output {
        (*self).pow(power)
    }
}

impl Pow<&Dual2> for f64 {
    type Output = Dual2;
    fn pow(self, power: &Dual2) -> Self::Output {
        let df_dp = self.ln() * self.pow(power.real);
        let d2f_dp2 = df_dp * self.ln();
        Dual2 {
            real: self.pow(power.real),
            vars: Arc::clone(power.vars()),
            dual: &power.dual * self.pow(power.real) * self.ln(),
            dual2: df_dp * &power.dual2
                + 0.5_f64 * d2f_dp2 * fouter11_(&power.dual.view(), &power.dual.view()),
        }
    }
}

impl Pow<Dual2> for f64 {
    type Output = Dual2;
    fn pow(self, power: Dual2) -> Self::Output {
        let df_dp = self.ln() * self.pow(power.real);
        let d2f_dp2 = df_dp * self.ln();
        Dual2 {
            real: self.pow(power.real),
            vars: power.vars,
            dual: &power.dual * self.pow(power.real) * self.ln(),
            dual2: df_dp * &power.dual2
                + 0.5_f64 * d2f_dp2 * fouter11_(&power.dual.view(), &power.dual.view()),
        }
    }
}

impl Pow<&Dual2> for &f64 {
    type Output = Dual2;
    fn pow(self, power: &Dual2) -> Self::Output {
        (*self).pow(power)
    }
}

impl Pow<Dual2> for &f64 {
    type Output = Dual2;
    fn pow(self, power: Dual2) -> Self::Output {
        (*self).pow(power)
    }
}

impl Pow<f64> for Dual {
    type Output = Dual;
    fn pow(self, power: f64) -> Self::Output {
        Dual {
            real: self.real.pow(power),
            vars: self.vars,
            dual: self.dual * power * self.real.pow(power - 1.0),
        }
    }
}

impl Pow<&f64> for Dual {
    type Output = Dual;
    fn pow(self, power: &f64) -> Self::Output {
        self.pow(*power)
    }
}

impl Pow<f64> for &Dual {
    type Output = Dual;
    fn pow(self, power: f64) -> Self::Output {
        Dual {
            real: self.real.pow(power),
            vars: Arc::clone(self.vars()),
            dual: &self.dual * power * self.real.pow(power - 1.0),
        }
    }
}

impl Pow<&f64> for &Dual {
    type Output = Dual;
    fn pow(self, power: &f64) -> Self::Output {
        self.pow(*power)
    }
}

impl Pow<&Dual> for &Dual {
    type Output = Dual;
    fn pow(self, power: &Dual) -> Self::Output {
        let state = self.vars_cmp(power.vars());
        match state {
            VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => Dual {
                real: self.real.pow(power.real),
                vars: Arc::clone(&self.vars),
                dual: power.real * self.real.pow(power.real - 1_f64) * &self.dual
                    + self.real.ln() * self.real.pow(power.real) * &power.dual,
            },
            _ => {
                let (z, p) = self.to_union_vars(power, None);
                Dual {
                    real: z.real.pow(p.real),
                    vars: Arc::clone(z.vars()),
                    dual: p.real * z.real.pow(p.real - 1_f64) * &z.dual
                        + z.real.ln() * z.real.pow(p.real) * &p.dual,
                }
            }
        }
    }
}

impl Pow<&Dual> for Dual {
    type Output = Dual;
    fn pow(self, power: &Dual) -> Self::Output {
        (&self).pow(power)
    }
}

impl Pow<Dual> for &Dual {
    type Output = Dual;
    fn pow(self, power: Dual) -> Self::Output {
        self.pow(&power)
    }
}

impl Pow<Dual> for Dual {
    type Output = Dual;
    fn pow(self, power: Dual) -> Self::Output {
        (&self).pow(&power)
    }
}

impl Pow<f64> for Dual2 {
    type Output = Dual2;
    fn pow(self, power: f64) -> Self::Output {
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

impl Pow<&f64> for Dual2 {
    type Output = Dual2;
    fn pow(self, power: &f64) -> Self::Output {
        self.pow(*power)
    }
}

impl Pow<f64> for &Dual2 {
    type Output = Dual2;
    fn pow(self, power: f64) -> Self::Output {
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

impl Pow<&f64> for &Dual2 {
    type Output = Dual2;
    fn pow(self, power: &f64) -> Self::Output {
        self.pow(*power)
    }
}

impl Pow<&Dual2> for &Dual2 {
    type Output = Dual2;
    fn pow(self, power: &Dual2) -> Self::Output {
        let state = self.vars_cmp(power.vars());
        match state {
            VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
                let f_z = power.real * self.real.pow(power.real - 1_f64);
                let f_p = self.real.pow(power.real) * self.real.ln();
                let f_zz = power.real * (power.real - 1_f64) * self.real.pow(power.real - 2_f64);
                let f_pp = self.real.ln() * self.real.ln() * self.real.pow(power.real);
                let f_pz =
                    (power.real * self.real.ln() + 1_f64) * self.real.pow(power.real - 1_f64);
                let cross_beta = fouter11_(&power.dual.view(), &self.dual.view());
                Dual2 {
                    real: self.real.pow(power.real),
                    vars: Arc::clone(self.vars()),
                    dual: f_z * &self.dual + f_p * &power.dual,
                    dual2: f_z * &self.dual2
                        + f_p * &power.dual2
                        + 0.5_f64 * f_zz * fouter11_(&self.dual.view(), &self.dual.view())
                        + 0.5_f64 * f_pz * (&cross_beta + &cross_beta.t())
                        + 0.5_f64 * f_pp * fouter11_(&power.dual.view(), &power.dual.view()),
                }
            }
            _ => {
                let (z, p) = self.to_union_vars(power, None);
                let f_z = p.real * z.real.pow(p.real - 1_f64);
                let f_p = z.real.pow(p.real) * z.real.ln();
                let f_zz = p.real * (p.real - 1_f64) * z.real.pow(p.real - 2_f64);
                let f_pp = z.real.ln() * z.real.ln() * z.real.pow(p.real);
                let f_pz = (p.real * z.real.ln() + 1_f64) * z.real.pow(p.real - 1_f64);
                let cross_beta = fouter11_(&p.dual.view(), &z.dual.view());
                Dual2 {
                    real: z.real.pow(p.real),
                    vars: Arc::clone(z.vars()),
                    dual: f_z * &z.dual + f_p * &p.dual,
                    dual2: f_z * &z.dual2
                        + f_p * &p.dual2
                        + 0.5_f64 * f_zz * fouter11_(&z.dual.view(), &z.dual.view())
                        + 0.5_f64 * f_pz * (&cross_beta + &cross_beta.t())
                        + 0.5_f64 * f_pp * fouter11_(&p.dual.view(), &p.dual.view()),
                }
            }
        }
    }
}

impl Pow<&Dual2> for Dual2 {
    type Output = Dual2;
    fn pow(self, power: &Dual2) -> Self::Output {
        (&self).pow(power)
    }
}

impl Pow<Dual2> for &Dual2 {
    type Output = Dual2;
    fn pow(self, power: Dual2) -> Self::Output {
        self.pow(&power)
    }
}

impl Pow<Dual2> for Dual2 {
    type Output = Dual2;
    fn pow(self, power: Dual2) -> Self::Output {
        (&self).pow(&power)
    }
}

impl Pow<f64> for Number {
    type Output = Number;
    fn pow(self, power: f64) -> Self::Output {
        match self {
            Number::F64(f) => Number::F64(f.pow(power)),
            Number::Dual(d) => Number::Dual(d.pow(power)),
            Number::Dual2(d) => Number::Dual2(d.pow(power)),
        }
    }
}

impl Pow<&f64> for Number {
    type Output = Number;
    fn pow(self, power: &f64) -> Self::Output {
        match self {
            Number::F64(f) => Number::F64(f.pow(power)),
            Number::Dual(d) => Number::Dual(d.pow(power)),
            Number::Dual2(d) => Number::Dual2(d.pow(power)),
        }
    }
}

impl Pow<f64> for &Number {
    type Output = Number;
    fn pow(self, power: f64) -> Self::Output {
        match self {
            Number::F64(f) => Number::F64(f.pow(power)),
            Number::Dual(d) => Number::Dual(d.pow(power)),
            Number::Dual2(d) => Number::Dual2(d.pow(power)),
        }
    }
}

impl Pow<&f64> for &Number {
    type Output = Number;
    fn pow(self, power: &f64) -> Self::Output {
        match self {
            Number::F64(f) => Number::F64(f.pow(power)),
            Number::Dual(d) => Number::Dual(d.pow(power)),
            Number::Dual2(d) => Number::Dual2(d.pow(power)),
        }
    }
}

impl Pow<Number> for Number {
    type Output = Number;
    fn pow(self, power: Number) -> Self::Output {
        match (self, power) {
            (Number::F64(f), Number::F64(f2)) => Number::F64(f.pow(f2)),
            (Number::F64(f), Number::Dual(d2)) => Number::Dual(f.pow(d2)),
            (Number::F64(f), Number::Dual2(d2)) => Number::Dual2(f.pow(d2)),
            (Number::Dual(d), Number::F64(f2)) => Number::Dual(d.pow(f2)),
            (Number::Dual(d), Number::Dual(d2)) => Number::Dual(d.pow(d2)),
            (Number::Dual(_), Number::Dual2(_)) => {
                panic!("Cannot mix dual types: Dual/Dual2")
            }
            (Number::Dual2(d), Number::F64(f2)) => Number::Dual2(d.pow(f2)),
            (Number::Dual2(_), Number::Dual(_)) => {
                panic!("Cannot mix dual types: Dual2/Dual")
            }
            (Number::Dual2(d), Number::Dual2(d2)) => Number::Dual2(d.pow(d2)),
        }
    }
}

impl Pow<&Number> for Number {
    type Output = Number;
    fn pow(self, power: &Number) -> Self::Output {
        match (self, power) {
            (Number::F64(f), Number::F64(f2)) => Number::F64(f.pow(f2)),
            (Number::F64(f), Number::Dual(d2)) => Number::Dual(f.pow(d2)),
            (Number::F64(f), Number::Dual2(d2)) => Number::Dual2(f.pow(d2)),
            (Number::Dual(d), Number::F64(f2)) => Number::Dual(d.pow(f2)),
            (Number::Dual(d), Number::Dual(d2)) => Number::Dual(d.pow(d2)),
            (Number::Dual(_), Number::Dual2(_)) => {
                panic!("Cannot mix dual types: Dual/Dual2")
            }
            (Number::Dual2(d), Number::F64(f2)) => Number::Dual2(d.pow(f2)),
            (Number::Dual2(_), Number::Dual(_)) => {
                panic!("Cannot mix dual types: Dual2/Dual")
            }
            (Number::Dual2(d), Number::Dual2(d2)) => Number::Dual2(d.pow(d2)),
        }
    }
}

impl Pow<Number> for &Number {
    type Output = Number;
    fn pow(self, power: Number) -> Self::Output {
        match (self, power) {
            (Number::F64(f), Number::F64(f2)) => Number::F64(f.pow(f2)),
            (Number::F64(f), Number::Dual(d2)) => Number::Dual(f.pow(d2)),
            (Number::F64(f), Number::Dual2(d2)) => Number::Dual2(f.pow(d2)),
            (Number::Dual(d), Number::F64(f2)) => Number::Dual(d.pow(f2)),
            (Number::Dual(d), Number::Dual(d2)) => Number::Dual(d.pow(d2)),
            (Number::Dual(_), Number::Dual2(_)) => {
                panic!("Cannot mix dual types: Dual/Dual2")
            }
            (Number::Dual2(d), Number::F64(f2)) => Number::Dual2(d.pow(f2)),
            (Number::Dual2(_), Number::Dual(_)) => {
                panic!("Cannot mix dual types: Dual2/Dual")
            }
            (Number::Dual2(d), Number::Dual2(d2)) => Number::Dual2(d.pow(d2)),
        }
    }
}

impl Pow<&Number> for &Number {
    type Output = Number;
    fn pow(self, power: &Number) -> Self::Output {
        match (self, power) {
            (Number::F64(f), Number::F64(f2)) => Number::F64(f.pow(f2)),
            (Number::F64(f), Number::Dual(d2)) => Number::Dual(f.pow(d2)),
            (Number::F64(f), Number::Dual2(d2)) => Number::Dual2(f.pow(d2)),
            (Number::Dual(d), Number::F64(f2)) => Number::Dual(d.pow(f2)),
            (Number::Dual(d), Number::Dual(d2)) => Number::Dual(d.pow(d2)),
            (Number::Dual(_), Number::Dual2(_)) => {
                panic!("Cannot mix dual types: Dual/Dual2")
            }
            (Number::Dual2(d), Number::F64(f2)) => Number::Dual2(d.pow(f2)),
            (Number::Dual2(_), Number::Dual(_)) => {
                panic!("Cannot mix dual types: Dual2/Dual")
            }
            (Number::Dual2(d), Number::Dual2(d2)) => Number::Dual2(d.pow(d2)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::dual_ops::math_funcs::MathFuncs;
    use ndarray::Array1;

    fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
        // used rather than equality for float numbers
        (a - b).abs() < abs_tol.unwrap_or(1e-8)
    }

    fn assert_is_close_vecs(v1: &Vec<f64>, v2: &Vec<f64>) {
        let v: Vec<bool> = v1
            .iter()
            .zip(v2.iter())
            .map(|(x, y)| is_close(&x, &y, None))
            .collect();
        assert!(v.iter().all(|x| *x));
    }

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

    #[test]
    fn test_enum() {
        let f = Number::F64(2.0);
        let d = Dual::new(3.0, vec!["x".to_string()]);
        assert_eq!(Number::F64(4.0_f64), f.pow(2.0_f64));

        let res = (&d).pow(2.0_f64);
        assert_eq!(Number::Dual(res), Number::Dual(d).pow(2.0_f64));
    }

    #[test]
    fn test_dual_dual() {
        let z = Dual::new(2.0_f64, vec!["x".to_string()]);
        let p = Dual::new(3.0_f64, vec!["p".to_string()]);
        let result = (&z).pow(&p);
        let expected = Dual::try_new(
            8.0,
            vec!["x".to_string(), "p".to_string()],
            vec![12.0, 2.0_f64.ln() * 8.0],
        )
        .unwrap();
        assert_eq!(result, expected);

        let result2 = (&z).pow(p);
        assert_eq!(result2, expected);
        let p = Dual::new(3.0_f64, vec!["p".to_string()]);
        let result3 = (z).pow(&p);
        assert_eq!(result3, expected);
        let z = Dual::new(2.0_f64, vec!["x".to_string()]);
        let result4 = (z).pow(p);
        assert_eq!(result4, expected);
    }

    #[test]
    fn test_f64_dual() {
        let p = Dual::new(3.0_f64, vec!["p".to_string()]);
        let result = (&2_f64).pow(&p);
        let expected = Dual::try_new(8.0, vec!["p".to_string()], vec![2.0_f64.ln() * 8.0]).unwrap();
        assert_eq!(result, expected);

        let result2 = (&2_f64).pow(p);
        assert_eq!(result2, expected);
        let p = Dual::new(3.0_f64, vec!["p".to_string()]);
        let result3 = (2_f64).pow(&p);
        assert_eq!(result3, expected);
        let result4 = (2_f64).pow(p);
        assert_eq!(result4, expected);
    }

    #[test]
    fn test_f64_dual2() {
        let p = Dual2::new(3.0_f64, vec!["p".to_string()]);
        let result = (&2_f64).pow(&p);
        let expected = Dual2::try_new(
            8.0,
            vec!["p".to_string()],
            vec![2.0_f64.ln() * 8.0],
            vec![2_f64.ln() * 2_f64.ln() * 4_f64],
        )
        .unwrap();
        assert_eq!(result, expected);

        let result2 = (&2_f64).pow(p);
        assert_eq!(result2, expected);
        let p = Dual2::new(3.0_f64, vec!["p".to_string()]);
        let result3 = (2_f64).pow(&p);
        assert_eq!(result3, expected);
        let result4 = (2_f64).pow(p);
        assert_eq!(result4, expected);
    }

    #[test]
    fn test_dual2_dual2() {
        // test all ref and deref binary ops
        let p = Dual2::new(3.0_f64, vec!["p".to_string()]);
        let z = Dual2::new(3.0_f64, vec!["z".to_string()]);
        let x = Dual2::new(3.0_f64, vec!["x".to_string()]);
        let y = Dual2::new(3.0_f64, vec!["y".to_string()]);
        let mut _result = (&z).pow(&p);
        _result = z.pow(&p);
        _result = (&x).pow(p);
        _result = x.pow(y);
    }

    #[test]
    fn test_dual2_dual2_branch_equivalence() {
        // test match branches yield the same calculation for Var equivalence and difference
        let p = Dual2::try_new(
            3.0_f64,
            vec!["p".to_string(), "s".to_string()],
            vec![1.1, 2.1],
            vec![1.1, 2.2, 2.2, 1.4],
        )
        .unwrap();
        let z = Dual2::try_new(
            2.0_f64,
            vec!["s".to_string(), "p".to_string()],
            vec![1.9, 2.9],
            vec![3.4, 1.2, 1.2, 0.1],
        )
        .unwrap();
        let z_p = Dual2::try_new_from(
            &p,
            2.0_f64,
            vec!["p".to_string(), "s".to_string()],
            vec![2.9, 1.9],
            vec![0.1, 1.2, 1.2, 3.4],
        )
        .unwrap();
        let result1 = (&p).pow(z);
        let result2 = p.pow(z_p);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_dual2_dual2_op_equivalence() {
        // test the analytical derivative calculations match those expected from exp and log
        let p = Dual2::try_new(
            3.0_f64,
            vec!["p".to_string(), "s".to_string()],
            vec![1.1, 2.1],
            vec![1.1, 2.2, 2.2, 1.4],
        )
        .unwrap();
        let z = Dual2::try_new(
            2.0_f64,
            vec!["s".to_string(), "p".to_string()],
            vec![1.9, 2.9],
            vec![3.4, 1.2, 1.2, 0.1],
        )
        .unwrap();
        let r1 = (&z).pow(&p);
        let r2 = (z.log() * p).exp();
        assert_is_close_vecs(&r1.dual.to_vec(), &r2.dual.to_vec());
        assert_is_close_vecs(
            &r1.dual2.into_raw_vec_and_offset().0,
            &r2.dual2.into_raw_vec_and_offset().0,
        );
    }

    #[test]
    fn test_number_number() {
        // test implemented crosses
        fn x1() -> Number {
            Number::F64(2.3)
        }
        fn x2() -> Number {
            Number::Dual(Dual::new(2.3, vec![]))
        }
        fn x3() -> Number {
            Number::Dual2(Dual2::new(1.1, vec![]))
        }

        let mut _res: Number;
        _res = (&x1()).pow(&x1());
        _res = (&x1()).pow(&x2());
        _res = (&x1()).pow(&x3());
        _res = (&x2()).pow(&x1());
        _res = (&x2()).pow(&x2());
        _res = (&x3()).pow(&x1());
        _res = (&x3()).pow(&x3());

        _res = (x1()).pow(&x1());
        _res = (x1()).pow(&x2());
        _res = (x1()).pow(&x3());
        _res = (x2()).pow(&x1());
        _res = (x2()).pow(&x2());
        _res = (x3()).pow(&x1());
        _res = (x3()).pow(&x3());

        _res = (&x1()).pow(x1());
        _res = (&x1()).pow(x2());
        _res = (&x1()).pow(x3());
        _res = (&x2()).pow(x1());
        _res = (&x2()).pow(x2());
        _res = (&x3()).pow(x1());
        _res = (&x3()).pow(x3());

        _res = (x1()).pow(x1());
        _res = (x1()).pow(x2());
        _res = (x1()).pow(x3());
        _res = (x2()).pow(x1());
        _res = (x2()).pow(x2());
        _res = (x3()).pow(x1());
        _res = (x3()).pow(x3());
    }
}
