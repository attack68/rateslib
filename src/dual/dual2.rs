//! Create and use data types for calculating derivatives up to second order using automatic
//! differentiation (AD).
//!
//! A second order dual number represents a function value and a quadratic manifold of the
//! gradient at that point. Mathematical operations are defined to give dual numbers
//! the ability to combine.

use crate::dual::dual1::{VarsState, Gradient1, Vars, MathFuncs, FieldOps};
use crate::dual::linalg_f64::fouter11_;
use auto_ops::{impl_op, impl_op_ex, impl_op_ex_commutative};
use indexmap::set::IndexSet;
use ndarray::{Array, Array1, Array2, Axis};
use num_traits;
use num_traits::identities::{One, Zero};
use num_traits::{Num, Pow, Signed};
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use statrs::distribution::{Normal, ContinuousCDF};
use std::f64::consts::PI;
// use std::fmt;
use std::iter::Sum;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass]
#[derive(Clone, Default, Debug)]
pub struct Dual2 {
    pub real: f64,
    pub vars: Arc<IndexSet<String>>,
    pub dual: Array1<f64>,
    pub dual2: Array2<f64>,
}

impl Vars for Dual2 {
    /// Get a reference to the Arc pointer for the `IndexSet` containing the struct's variables.
    fn vars(&self) -> &Arc<IndexSet<String>> {
        &self.vars
    }

    /// Construct a new `Dual2` with `vars` set as the given Arc pointer and gradients shuffled in memory.
    ///
    /// Examples
    ///
    /// ```rust
    /// let x = Dual2::new(1.5, vec!["x".to_string()]);
    /// let xy = Dual2::new(2.5, vec!["x".to_string(), "y".to_string()]);
    /// let x_y = x.to_new_vars(xy.vars(), None);
    /// // x_y: <Dual2: 1.5, (x, y), [1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]>
    fn to_new_vars(&self, arc_vars: &Arc<IndexSet<String>>, state: Option<VarsState>) -> Self {
        let dual_: Array1<f64>;
        let mut dual2_: Array2<f64> = Array2::zeros((arc_vars.len(), arc_vars.len()));
        let match_val = state.unwrap_or_else(|| self.vars_cmp(&arc_vars));
        match match_val {
            VarsState::EquivByArc | VarsState::EquivByVal => {
                dual_ = self.dual.clone();
                dual2_ = self.dual2.clone();
            },
            _ => {
                let lookup_or_zero = |v| {
                    match self.vars.get_index_of(v) {
                        Some(idx) => self.dual[idx],
                        None => 0.0_f64,
                    }
                };
                dual_ = Array1::from_vec(arc_vars.iter().map(lookup_or_zero).collect());

                let indices: Vec<Option<usize>> = arc_vars.iter().map(|x| self.vars.get_index_of(x)).collect();
                for (i, row_index) in indices.iter().enumerate() {
                    match row_index {
                        Some(row_value) => {
                            for (j, col_index) in indices.iter().enumerate() {
                                match col_index {
                                    Some(col_value) => { dual2_[[i, j]] = self.dual2[[*row_value, *col_value]] },
                                    None => {}
                                }
                            }
                        },
                        None => {},
                    }
                }
            }
        }
        Self {real: self.real, vars: Arc::clone(arc_vars), dual: dual_, dual2: dual2_}
    }
}

impl Gradient1 for Dual2 {
    fn dual(&self) -> &Array1<f64> {
        &self.dual
    }
}

pub trait Gradient2: Gradient1 {
    /// Get a reference to the Array containing the second order gradients.
    fn dual2(&self) -> &Array2<f64>;

    /// Return a set of first order gradients ordered by the given vector.
    ///
    /// Duplicate `vars` are dropped before parsing.
    fn gradient2(&self, vars: Vec<String>) -> Array2<f64> {
        let arc_vars = Arc::new(IndexSet::from_iter(vars));
        let state = self.vars_cmp(&arc_vars);
        match state {
            VarsState::EquivByArc | VarsState::EquivByVal => 2.0_f64 * self.dual2(),
            _ => {
                let indices: Vec<Option<usize>> = arc_vars.iter().map(|x| self.vars().get_index_of(x)).collect();
                let mut dual2_ = Array::zeros((arc_vars.len(), arc_vars.len()));
                for (i, row_index) in indices.iter().enumerate() {
                    for (j, col_index) in indices.iter().enumerate() {
                        match row_index {
                            Some(row_value) => match col_index {
                                Some(col_value) => dual2_[[i, j]] = self.dual2()[[*row_value, *col_value]],
                                None => {}
                            },
                            None => {}
                        }
                    }
                }
                2_f64 * dual2_
            }
        }
    }

    fn gradient1_manifold(&self, vars: Vec<String>) -> Array1<Dual2> {
        let indices: Vec<Option<usize>> = vars.iter().map(|x| self.vars().get_index_of(x)).collect();

        let default_zero = Dual2::new(0., vars.clone());
        let mut grad: Array1<Dual2> = Array::zeros(vars.len());
        for (i, i_idx) in indices.iter().enumerate() {
            match i_idx {
                Some(i_val) => {
                    let mut dual: Array1<f64> = Array1::zeros(vars.len());
                    for (j, j_idx) in indices.iter().enumerate() {
                        match j_idx {
                            Some(j_val) => dual[j] = self.dual2()[[*i_val, *j_val]] * 2.0,
                            None => {}
                        }
                    }
                    grad[i] = Dual2 {
                        real: self.dual()[*i_val],
                        vars: Arc::clone(&default_zero.vars),
                        dual2: Array2::zeros((vars.len(), vars.len())),
                        dual
                    };
                },
                None => { grad[i] = default_zero.clone() }
            }
        }
        grad
    }
}

impl Gradient2 for Dual2 {
    fn dual2(&self) -> &Array2<f64> { &self.dual2 }
}

impl Dual2 {
    /// Constructs a new `Dual2`.
    ///
    /// - `vars` should be **unique**; duplicates will be removed by the `IndexSet`.
    ///
    /// Gradient values for each of the provided `vars` is set to 1.0_f64.
    /// Second order gradient values for each combination of provided `vars` is set
    /// to 0.0_f64.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Dual2::new(2.5, vec!["x".to_string()]);
    /// // x: <Dual2: 2.5, (x), [1.0], [[0.0]]>
    /// ```
    pub fn new(real: f64, vars: Vec<String>) -> Self {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        Self {
            real,
            dual: Array1::ones(unique_vars_.len()),
            dual2: Array2::zeros((unique_vars_.len(), unique_vars_.len())),
            vars: unique_vars_,
        }
    }

    /// Constructs a new `Dual2`.
    ///
    /// - `vars` should be **unique**; duplicates will be removed by the `IndexSet`.
    /// - `dual` can be empty; if so each gradient with respect to each `vars` is set to 1.0_f64.
    /// - `dual2` can be empty; if so each gradient with respect to each `vars` is set to 0.0_f64.
    ///   Input as a flattened 2d-array in row major order.
    ///
    /// # Errors
    ///
    /// If the length of `dual` and of `vars` are not the same after parsing.
    /// If the shape of two dimension `dual2` does not match `vars` after parsing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Dual2::try_new(2.5, vec!["x".to_string()], vec![], vec![])?;
    /// // x: <Dual2: 2.5, (x), [1.0], [[0.0]]>
    /// ```
    pub fn try_new(real: f64, vars: Vec<String>, dual: Vec<f64>,  dual2: Vec<f64>) -> Result<Self, PyErr> {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        let dual_ = if dual.is_empty() {Array1::ones(unique_vars_.len())} else {Array1::from_vec(dual)};
        if unique_vars_.len() != dual_.len() {
            return Err(PyValueError::new_err("`vars` and `dual` must have the same length."))
        }


        let dual2_ = if dual2.is_empty() {
            Array2::zeros((unique_vars_.len(), unique_vars_.len()))
        } else {
            if dual2.len() != (unique_vars_.len() * unique_vars_.len()) {
                return Err(PyValueError::new_err("`vars` and `dual2` must have compatible lengths."))
            }
            Array::from_vec(dual2).into_shape((unique_vars_.len(), unique_vars_.len()))
                                  .expect("Reshaping failed, which should not occur because shape is pre-checked.")
        };
        Ok(Self {real, vars: unique_vars_, dual: dual_, dual2: dual2_})
    }

    /// Construct a new `Dual2` cloning the `vars` Arc pointer from another.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Dual2::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0])?;
    /// let y = Dual2::new_from(&x, 1.5, vec["y".to_string()]);
    /// ```
    ///
    /// This is semantically the same as:
    ///
    /// ```rust
    /// let x = Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0])?;
    /// let y = Dual::new(1.5, vec!["y".to_string()]).to_new_vars(x.vars(), None);
    /// ```
    pub fn new_from(other: &Self, real: f64, vars: Vec<String>) -> Self {
        let new = Self::new(real, vars);
        new.to_new_vars(&other.vars, None)
    }

    /// Construct a new `Dual2` cloning the `vars` Arc pointer from another.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Dual2::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0], vec![])?;
    /// let y = Dual2::new_from(&x, 1.5, vec["y".to_string()]);
    /// ```
    ///
    /// This is semantically the same as:
    ///
    /// ```rust
    /// let x = Dual2::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0], vec![])?;
    /// let y = Dual2::new(1.5, vec!["y".to_string()]).to_new_vars(x.vars(), None);
    /// ```
    pub fn try_new_from(other: &Self, real: f64, vars: Vec<String>, dual: Vec<f64>, dual2: Vec<f64>) -> Result<Self, PyErr> {
        let new = Self::try_new(real, vars, dual, dual2)?;
        Ok(new.to_new_vars(&other.vars, None))
    }

    /// Construct a new `Dual2` cloning the `vars` Arc pointer from another.
    ///
    pub fn clone_from(other: &Self, real: f64, dual:Array1<f64>, dual2: Array2<f64>) -> Self {
        assert_eq!(other.vars().len(), dual.len());
        assert_eq!(other.vars().len(), dual2.len_of(Axis(0)));
        assert_eq!(other.vars().len(), dual2.len_of(Axis(1)));
        Dual2{
            real,
            vars: Arc::clone(&other.vars),
            dual,
            dual2,
        }
    }

    /// Get the real component value of the struct.
    pub fn real(&self) -> f64 {
        self.real
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
            dual2: scalar * &self.dual2 + 0.5_f64 * scalar2 * cross_beta
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
            dual2: scalar * &self.dual2 + 0.5_f64 * scalar2 * cross_beta
        }
    }
}

// impl fmt::Debug for Dual2 {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", &self.real)
//     }
// }

impl Num for Dual2 {
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual2".to_string())
    }
}

// impl Neg for Dual2
impl_op!(-|a: Dual2| -> Dual2 {
    Dual2 {
        vars: a.vars,
        real: -a.real,
        dual: -a.dual,
        dual2: -a.dual2,
    }
});
impl_op!(-|a: &Dual2| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&a.vars),
        real: -a.real,
        dual: &a.dual * -1.0,
        dual2: &a.dual2 * -1.0,
    }
});

// impl Add for Dual2
impl_op_ex!(+ |a: &Dual2, b: &Dual2| -> Dual2 {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsState::EquivByArc | VarsState::EquivByVal => {
            Dual2 {
                real: a.real + b.real,
                dual: &a.dual + &b.dual,
                dual2: &a.dual2 + &b.dual2,
                vars: Arc::clone(&a.vars)}
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual2 {
                real: x.real + y.real,
                dual: &x.dual + &y.dual,
                dual2: &x.dual2 + &y.dual2,
                vars: Arc::clone(&x.vars)}
        }
    }
});

// impl Sub
impl_op_ex!(-|a: &Dual2, b: &Dual2| -> Dual2 {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsState::EquivByArc | VarsState::EquivByVal => {
            Dual2 {
                real: a.real - b.real,
                dual: &a.dual - &b.dual,
                dual2: &a.dual2 - &b.dual2,
                vars: Arc::clone(&a.vars)}
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual2 {
                real: x.real - y.real,
                dual: &x.dual - &y.dual,
                dual2: &x.dual2 - &y.dual2,
                vars: Arc::clone(&x.vars)}
        }
    }
});

// impl Mul for Dual2
impl_op_ex!(*|a: &Dual2, b: &Dual2| -> Dual2 {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsState::EquivByArc | VarsState::EquivByVal => {
            let mut dual2: Array2<f64> = &a.dual2 * b.real + &b.dual2 * a.real;
            let cross_beta = fouter11_(&a.dual.view(), &b.dual.view());
            dual2 = dual2 + 0.5_f64 * (&cross_beta + &cross_beta.t());
            Dual2 {
                real: a.real * b.real,
                dual: &a.dual * b.real + &b.dual * a.real,
                vars: Arc::clone(&a.vars),
                dual2,
            }
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            let mut dual2: Array2<f64> = &x.dual2 * y.real + &y.dual2 * x.real;
            let cross_beta = fouter11_(&x.dual.view(), &y.dual.view());
            dual2 = dual2 + 0.5_f64 * (&cross_beta + &cross_beta.t());
            Dual2 {
                real: x.real * y.real,
                dual: &x.dual * y.real + &y.dual * x.real,
                vars: Arc::clone(&x.vars),
                dual2,
            }
        }
    }
});

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
        let beta_cross = fouter11_(&self.dual().view(), &self.dual().view());
        Dual2 {
            real: self.real.powf(power),
            vars: Arc::clone(self.vars()),
            dual: self.dual() * coeff,
            dual2: self.dual2() * coeff + beta_cross * coeff2,
        }
    }
}

// impl Div for Dual2
impl_op_ex!(/ |a: &Dual2, b: &Dual2| -> Dual2 { a * b.clone().pow(-1.0) });

// impl Rem for Dual2
impl_op_ex!(% |a: &Dual2, b: &Dual2| -> Dual2 {
    let d = f64::trunc(a.real / b.real);
    a - d * b
});

impl One for Dual2 {
    fn one() -> Dual2 {
        Dual2::new(1.0, Vec::new())
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

impl PartialEq<Dual2> for Dual2 {
    fn eq(&self, other: &Dual2) -> bool {
        if self.real != other.real {
            false
        } else {
            let state = self.vars_cmp(other.vars());
            match state {
                VarsState::EquivByArc | VarsState::EquivByVal => {
                    self.dual.iter().eq(other.dual.iter()) &&
                        self.dual2.iter().eq(other.dual2.iter())
                }
                _ => {
                    let (x, y) = self.to_union_vars(other, Some(state));
                    x.dual.iter().eq(y.dual.iter()) &&
                        x.dual2.iter().eq(y.dual2.iter())
                }
            }
        }
    }
}

impl PartialOrd<Dual2> for Dual2 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl Sum for Dual2 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Dual2>,
    {
        iter.fold(Dual2::new(0.0, Vec::new()), |acc, x| acc + x)
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

    fn signum(&self) -> Self { Dual2::new(self.real.signum(), Vec::new()) }

    fn is_positive(&self) -> bool {
        self.real.is_sign_positive()
    }

    fn is_negative(&self) -> bool {
        self.real.is_sign_negative()
    }
}

impl FieldOps<Dual2> for Dual2 {}

// f64 Crossover

// Add
impl_op_ex_commutative!(+ |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone(), dual2: a.dual2.clone()}
});
// Sub
impl_op_ex!(-|a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&a.vars),
        real: a.real - b,
        dual: a.dual.clone(),
        dual2: a.dual2.clone(),
    }
});
impl_op_ex!(-|a: &f64, b: &Dual2| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&b.vars),
        real: a - b.real,
        dual: -(b.dual.clone()),
        dual2: -(b.dual2.clone()),
    }
});
// Mul
impl_op_ex_commutative!(*|a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&a.vars),
        real: a.real * b,
        dual: *b * &a.dual,
        dual2: *b * &a.dual2,
    }
});
// Div
impl_op_ex!(/ |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real / b, dual: (1_f64/b) * &a.dual, dual2: (1_f64/b) * &a.dual2}
});
impl_op_ex!(/ |a: &f64, b: &Dual2| -> Dual2 { a * b.clone().pow(-1.0) });
// Rem
impl_op_ex!(% |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real % b, dual: a.dual.clone(), dual2: a.dual2.clone()}
});
impl_op_ex!(% |a: &f64, b: &Dual2| -> Dual2 {
    Dual2::new(*a, Vec::new()) % b }
);

impl PartialEq<f64> for Dual2 {
    fn eq(&self, other: &f64) -> bool {
        Dual2::new(*other, Vec::new()) == *self
    }
}

impl PartialEq<Dual2> for f64 {
    fn eq(&self, other: &Dual2) -> bool {
        Dual2::new(*self, Vec::new()) == *other
    }
}

impl PartialOrd<f64> for Dual2 {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {self.real.partial_cmp(other)}
}

impl PartialOrd<Dual2> for f64 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {self.partial_cmp(&other.real)}
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn clone_arc() {
        let d1 = Dual2::new(20.0, vec!["a".to_string()]);
        let d2 = d1.clone();
        assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
    }

    #[test]
    fn default_dual() {
        let result = Dual2::default();
        let expected = Dual2::new(0.0, Vec::new());
        assert_eq!(result, expected);
    }

    #[test]
    fn to_new_ordered_vars() {
        let d1 = Dual2::new(20.0, vec!["a".to_string()]);
        let d2 = Dual2::new(20.0, vec!["a".to_string(), "b".to_string()]);
        let d3 = d1.to_new_vars(&d2.vars, None);
        assert!(Arc::ptr_eq(&d3.vars, &d2.vars));
        assert!(d3.dual.len() == 2);
        let d4 = d2.to_new_vars(&d1.vars, None);
        assert!(Arc::ptr_eq(&d4.vars, &d1.vars));
        assert!(d4.dual.len() == 1);
    }

    #[test]
    fn new_dual() {
        Dual2::new(2.3, Vec::from([String::from("a")]));
    }

    #[test]
    fn new_dual_error() {
        assert!(Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([1.0]),
            Vec::new(),
        ).is_err()
        );
    }

    #[test]
    fn new_dual2_error() {
        assert!(Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([1.0, 2.3]),
            Vec::from([1.0, 2.4, 3.4]),
        ).is_err()
        );
    }

    #[test]
    fn try_new_from() {
        let x = Dual2::new(1.2, vec!["x".to_string(), "y".to_string()]);
        let y = Dual2::try_new_from(&x, 3.2, vec!["y".to_string()], vec![1.9], vec![2.1]).unwrap();
        let z = Dual2::try_new(3.2, vec!["x".to_string(), "y".to_string()], vec![0., 1.9], vec![0., 0., 0., 2.1]).unwrap();
        assert_eq!(y, z);
    }

    #[test]
    fn to_new_vars() {
        let d1 = Dual2::new(2.5, vec!["x".to_string()]);
        let d2 = Dual2::new(3.5, vec!["x".to_string()]);
        let d3 = d1.to_new_vars(d2.vars(), None);
        assert!(d3.ptr_eq(&d2));
        assert_eq!(d3.real, 2.5);
        assert_eq!(d3.dual, Array1::from_vec(vec![1.0]));
    }

    #[test]
    fn gradient2_equivval() {
        let d1 = Dual2::try_new(
            2.5,
            vec!["x".to_string(), "y".to_string()],
            vec![2.3, 4.5],
            vec![1.0, 2.5, 2.5, 5.0],
        ).unwrap();
        let result = d1.gradient2(vec!["x".to_string(), "y".to_string()]);
        let expected = arr2(&[[2., 5.], [5., 10.]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn gradient2_diffvars() {
        let d1 = Dual2::try_new(
            2.5,
            vec!["x".to_string(), "y".to_string()],
            vec![2.3, 4.5],
            vec![1.0, 2.5, 2.5, 5.0],
        ).unwrap();
        let result = d1.gradient2(vec!["z".to_string(), "y".to_string()]);
        let expected = arr2(&[[0., 0.], [0., 10.]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn zero_init() {
        let d = Dual2::new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
        );
        for (_, val) in d.dual.indexed_iter() {
            assert!(*val == 1.0)
        }
    }

    #[test]
    fn negate() {
        let d = Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([2., -1.4]),
            Vec::from([1.0, -1.0, -1.0, 2.0]),
        ).unwrap();
        let d2 = -d.clone();
        assert!(d2.real == -2.3);
        assert!(Arc::ptr_eq(&d.vars, &d2.vars));
        assert!(d2.dual[0] == -2.0);
        assert!(d2.dual[1] == 1.4);
        assert!(d2.dual2[[1, 0]] == 1.0);
    }

    #[test]
    fn negate_ref() {
        let d = Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([2., -1.4]),
            Vec::from([1.0, -1.0, -1.0, 2.0]),
        ).unwrap();
        let d2 = -&d;
        assert!(d2.real == -2.3);
        assert!(Arc::ptr_eq(&d.vars, &d2.vars));
        assert!(d2.dual[0] == -2.0);
        assert!(d2.dual[1] == 1.4);
        assert!(d2.dual2[[1, 0]] == 1.0);
    }

    #[test]
    fn one() {
        let d = Dual2::one();
        assert_eq!(d, Dual2::new(1.0, vec![]));
    }

    #[test]
    fn pow_ref() {
        let d1 = Dual2::new(3.0, vec!["x".to_string()]);
        let d2 = (&d1).pow(2.0);
        assert_eq!(d2.real, 9.0);
        assert_eq!(d2.dual, Array1::from_vec(vec![6.0]));
    }

    #[test]
    fn signed_() {
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
    fn rem_() {
        let d1 = Dual2::try_new(10.0, vec!["x".to_string()], vec![2.0], vec![]).unwrap();
        let d2 = Dual2::new(3.0, vec!["x".to_string()]);
        let result = d1 % d2;
        let expected = Dual2::try_new(1.0, vec!["x".to_string()], vec![-1.0], vec![]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn rem_f64_() {
        let d1 = Dual2::try_new(10.0, vec!["x".to_string()], vec![2.0], vec![]).unwrap();
        let result = &d1 % 3.0_f64;
        assert_eq!(result, Dual2::try_new(1.0, vec!["x".to_string()], vec![2.0], vec![]).unwrap());

        let result = 11.0_f64 % d1;
        assert_eq!(result, Dual2::try_new(1.0, vec!["x".to_string()], vec![-2.0], vec![]).unwrap());
    }

    #[test]
    fn is_zero() {
        let d = Dual2::zero();
        assert!(d.is_zero());
    }

    #[test]
    fn eq_ne() {
        // Dual with vars - f64
        assert!(Dual2::new(0.0, Vec::from([String::from("a")])) != 0.0);
        // Dual with no vars - f64 (+reverse)
        assert!(Dual2::new(2.0, Vec::new()) == 2.0);
        assert!(2.0 == Dual2::new(2.0, Vec::new()));
        // Dual - Dual (various real, vars, gradient mismatch)
        let d = Dual2::try_new(
            2.0,
            Vec::from([String::from("a")]),
            Vec::from([2.3]),
            Vec::new(),
        ).unwrap();
        assert!(
            d == Dual2::try_new(
                2.0,
                Vec::from([String::from("a")]),
                Vec::from([2.3]),
                Vec::new()
            ).unwrap()
        );
        assert!(
            d != Dual2::try_new(
                2.0,
                Vec::from([String::from("b")]),
                Vec::from([2.3]),
                Vec::new()
            ).unwrap()
        );
        assert!(
            d != Dual2::try_new(
                3.0,
                Vec::from([String::from("a")]),
                Vec::from([2.3]),
                Vec::new()
            ).unwrap()
        );
        assert!(
            d != Dual2::try_new(
                2.0,
                Vec::from([String::from("a")]),
                Vec::from([1.3]),
                Vec::new()
            ).unwrap()
        );
        // Dual - Dual (missing Vars are zero and upcasted)
        assert!(
            d == Dual2::try_new(
                2.0,
                Vec::from([String::from("a"), String::from("b")]),
                Vec::from([2.3, 0.0]),
                Vec::new()
            ).unwrap()
        );
    }

    #[test]
    fn add_f64() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = 10.0 + d1 + 15.0;
        let expected = Dual2::try_new(
            26.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn add() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let d2 = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        ).unwrap();
        let expected = Dual2::try_new(
            3.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, 3.0],
            Vec::new(),
        ).unwrap();
        let result = d1 + d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn ops_equiv() {
        let d1 = Dual2::try_new(1.5, vec!["x".to_string()], vec![1.0], vec![0.0]).unwrap();
        let d2 = Dual2::try_new(2.5, vec!["x".to_string()], vec![2.0], vec![0.0]).unwrap();
        let result = &d1 + &d2;
        assert_eq!(result, Dual2::try_new(4.0, vec!["x".to_string()], vec![3.0], vec![0.0]).unwrap());
        let result = &d1 - &d2;
        assert_eq!(result, Dual2::try_new(-1.0, vec!["x".to_string()], vec![-1.0], vec![0.0]).unwrap());
    }

    #[test]
    fn sub_f64() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = (10.0 - d1) - 15.0;
        let expected = Dual2::try_new(
            -6.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
            Vec::new(),
        ).unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn sub() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let d2 = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        ).unwrap();
        let expected = Dual2::try_new(
            -1.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, -3.0],
            Vec::new(),
        ).unwrap();
        let result = d1 - d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn mul_f64() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = 10.0 * d1 * 2.0;
        let expected = Dual2::try_new(
            20.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![20.0, 40.0],
            Vec::new(),
        ).unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn mul() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let d2 = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        ).unwrap();
        let expected = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![2.0, 4.0, 3.0],
            vec![
                0., 0., 1.5,
                0., 0., 3.,
                1.5, 3., 0.,
            ],
        ).unwrap();
        let result = d1 * d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn inv() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = d1.clone() * d1.pow(-1.0);
        let expected = Dual2::new(1.0, vec![]);
        assert_eq!(result, expected)
    }

    #[test]
    fn abs() {
        let d1 = Dual2::try_new(
            -2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = d1.abs();
        let expected = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
            Vec::new(),
        ).unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn div_f64() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = d1 / 2.0;
        let expected = Dual2::try_new(
            0.5,
            vec!["v0".to_string(), "v1".to_string()],
            vec![0.5, 1.0],
            Vec::new(),
        ).unwrap();
        assert_eq!(result, expected)
    }

    #[test]
    fn f64_div() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = 2.0 / d1.clone();
        let expected = Dual2::new(2.0, vec![]) / d1;
        assert_eq!(result, expected)
    }

    #[test]
    fn div() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let d2 = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        ).unwrap();
        let expected = Dual2::try_new(
            0.5,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![0.5, 1.0, -0.75],
            vec![
                0., 0., -0.375,
                0., 0., -0.75,
                -0.375, -0.75, 1.125,
            ],
        ).unwrap();
        let result = d1 / d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn ord() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        assert!(d1 < 2.0);
        assert!(d1 > 0.5);
        assert!(d1 <= 1.0);
        assert!(d1 >= 1.0);
        assert!(1.0 <= d1);
        assert!(1.0 >= d1);
        assert!(2.0 > d1);
        assert!(0.5 < d1);
        let d2 = Dual2::try_new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        assert!(d2 > d1);
        assert!(d1 < d2);
        let d3 = Dual2::try_new(1.0, vec!["v3".to_string()], vec![10.0], Vec::new()).unwrap();
        assert!(d1 >= d3);
        assert!(d1 <= d3);
    }

    #[test]
    fn exp() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = d1.exp();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.exp();
        let expected = Dual2::try_new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0 * c, 2.0 * c],
            vec![1.0_f64.exp() * 0.5, 1.0_f64.exp(), 1.0_f64.exp(), 1.0_f64.exp()*2.],
        ).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn log() {
        let d1 = Dual2::try_new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        ).unwrap();
        let result = d1.log();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.ln();
        let expected = Dual2::try_new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            vec![-0.5, -1.0, -1.0, -2.0],
        ).unwrap();
        println!("{:?}", result.dual2);
        assert_eq!(result, expected);
    }

    #[test]
    fn grad_manifold() {
        let d1 = Dual2::try_new(
            2.0,
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            vec![1., 2., 3.],
            vec![2., 3., 4., 3.,5., 6., 4., 6., 7.],
        ).unwrap();
        let result = d1.gradient1_manifold(vec!["y".to_string(), "z".to_string()]);
        assert_eq!(result[0].real, 2.);
        assert_eq!(result[1].real, 3.);
        assert_eq!(result[0].dual, Array1::from_vec(vec![10., 12.]));
        assert_eq!(result[1].dual, Array1::from_vec(vec![12., 14.]));
        assert_eq!(result[0].dual2, Array2::<f64>::zeros((2, 2)));
        assert_eq!(result[1].dual2, Array2::<f64>::zeros((2, 2)));
    }

    // #[test]
    // #[should_panic]
    // fn no_dual_cross(){
    //     let a = Dual::new(2.0, Vec::new(), Vec::new());
    //     let b = Dual2::new(3.0, Vec::new(), Vec::new(), Vec::new());
    //     a + b
    // }

}
