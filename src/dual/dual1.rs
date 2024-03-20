//! Create and use data types for calculating first order derivatives using AD.
//!
//! Dual number

use indexmap::set::IndexSet;
use std::sync::Arc;
use std::cmp::{Ordering, PartialEq};
use std::iter::Sum;
use std::ops::{Add, Sub, Mul, Div};
use ndarray::Array1;
use num_traits;
use num_traits::identities::{One, Zero};
use num_traits::{Num, Pow, Signed, NumOps};
use auto_ops::{impl_op, impl_op_ex, impl_op_ex_commutative};
use pyo3::pyclass;

/// Struct for defining a dual number data type supporting first order derivatives.
#[pyclass]
#[derive(Clone, Default, Debug)]
pub struct Dual {
    real: f64,
    vars: Arc<IndexSet<String>>,
    dual: Array1<f64>,
}

/// Enum defining the `vars` state of two dual number type structs, a LHS relative to a RHS.
#[derive(Clone, Debug, PartialEq)]
pub enum VarsState {
    EquivByArc,  // Duals share an Arc ptr to their Vars
    EquivByVal,  // Duals share the same vars in the same order but no Arc ptr
    Superset,    // The Dual vars contains all of the queried values and is larger set
    Subset,      // The Dual vars is contained in the queried values and is smaller set
    Difference,  // The Dual vars and the queried set contain different values.
}

pub trait Vars where Self: Clone {
    fn vars(&self) -> &Arc<IndexSet<String>>;
    fn to_new_vars(&self, arc_vars: &Arc<IndexSet<String>>, state: Option<VarsState>) -> Self;

    /// Compare the `vars` on a `Dual` with a given Arc pointer.
    fn vars_cmp(&self, arc_vars: &Arc<IndexSet<String>>) -> VarsState {
        if Arc::ptr_eq(self.vars(), arc_vars) {
            VarsState::EquivByArc
        } else if self.vars().len() == arc_vars.len()
            && self.vars().iter().zip(arc_vars.iter()).all(|(a, b)| a == b) {
            VarsState::EquivByVal
        } else if self.vars().len() >= arc_vars.len()
            && arc_vars.iter().all(|var| self.vars().contains(var)) {
            VarsState::Superset
        } else if self.vars().len() < arc_vars.len()
            && self.vars().iter().all(|var| arc_vars.contains(var)) {
            VarsState::Subset
        } else {
            VarsState::Difference
        }
    }
    // fn vars_cmp(&self, arc_vars: &Arc<IndexSet<String>>) -> VarsState;

    /// Construct a tuple of 2 `Self` types whose `vars` are linked by an Arc pointer.
    ///
    /// Gradient values contained in fields may be shuffled in memory if necessary
    /// according to the calculated `VarsState`. Do not use `state` directly unless you have
    /// performed a pre-check.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Dual::new(1.0, vec!["x".to_string()], vec![]);
    /// let y = Dual::new(1.5, vec!["y".to_string()], vec![]);
    /// let (a, b) = x.to_union_vars(&y, Some(VarsState::Difference));
    /// // a: <Dual: 1.0, (x, y), [1.0, 0.0]>
    /// // b: <Dual: 1.5, (x, y), [0.0, 1.0]>
    /// ```
    fn to_union_vars(&self, other: &Self, state: Option<VarsState>) -> (Self, Self) where Self: Sized {
        let state_ = state.unwrap_or_else(|| self.vars_cmp(other.vars()));
        match state_ {
            VarsState::EquivByArc => (self.clone(), other.clone()),
            VarsState::EquivByVal => (self.clone(), other.to_new_vars(self.vars(), Some(state_))),
            VarsState::Superset => (self.clone(), other.to_new_vars(self.vars(), Some(VarsState::Subset))),
            VarsState::Subset => (self.to_new_vars(other.vars(), Some(state_)), other.clone()),
            VarsState::Difference => self.to_combined_vars(other),
        }
    }

    fn to_combined_vars(&self, other: &Self) -> (Self, Self) where Self: Sized {
        let comb_vars = Arc::new(IndexSet::from_iter(
            self.vars().union(&other.vars()).map(|x| x.clone()),
        ));
        (self.to_new_vars(&comb_vars, Some(VarsState::Difference)),
         other.to_new_vars(&comb_vars, Some(VarsState::Difference)))
    }

    /// Compare if two `Dual` structs share the same `vars`by Arc pointer equivalence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x1 = Dual::new(1.5, vec!["x".to_string()], vec![]);
    /// let x2 = Dual::new(2.5, vec!["x".to_string()], vec![]);
    /// x1.ptr_eq(&x2); // false
    /// ```
    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.vars(), other.vars())
    }
}

impl Vars for Dual {
    /// Get a reference to the Arc pointer for the `IndexSet` containing the struct's variables.
    fn vars(&self) -> &Arc<IndexSet<String>> {
        &self.vars
    }

    /// Construct a new `Dual` with `vars` set as the given Arc pointer and gradients shuffled in memory.
    ///
    /// Examples
    ///
    /// ```rust
    /// let x = Dual::new(1.5, vec!["x".to_string()], vec![]);
    /// let xy = Dual::new(2.5, vec!["x".to_string(), "y".to_string()], vec![]);
    /// let x_y = x.to_new_vars(xy.vars(), None);
    /// // x_y: <Dual: 1.5, (x, y), [1.0, 0.0]>
    fn to_new_vars(&self, arc_vars: &Arc<IndexSet<String>>, state: Option<VarsState>) -> Self {
        let dual_: Array1<f64>;
        let match_val = state.unwrap_or_else(|| self.vars_cmp(&arc_vars));
        match match_val {
            VarsState::EquivByArc | VarsState::EquivByVal => dual_ = self.dual.clone(),
            _ => {
                let lookup_or_zero = |v| {
                    match self.vars.get_index_of(v) {
                        Some(idx) => self.dual[idx],
                        None => 0.0_f64,
                    }
                };
                dual_ = Array1::from_vec(arc_vars.iter().map(lookup_or_zero).collect());
            }
        }
        Self {real: self.real, vars: Arc::clone(arc_vars), dual: dual_}
    }
}

pub trait Gradient1: Vars {
    /// Get a reference to the Array containing the first order gradients.
    fn dual(&self) -> &Array1<f64>;

    /// Return a set of first order gradients ordered by the given vector.
    ///
    /// Duplicate `vars` are dropped before parsing.
    fn gradient1(&self, vars: Vec<String>) -> Array1<f64> {
        let arc_vars = Arc::new(IndexSet::from_iter(vars));
        let state = self.vars_cmp(&arc_vars);
        match state {
            VarsState::EquivByArc | VarsState::EquivByVal => self.dual().clone(),
            _ =>{
                let mut dual_ = Array1::<f64>::zeros(arc_vars.len());
                for (i, index) in arc_vars.iter().map(|x| self.vars().get_index_of(x)).enumerate() {
                    match index {
                        Some(value) => dual_[i] = self.dual()[value],
                        None => dual_[i] = 0.0,
                    }
                }
                dual_
            }
        }
    }
}

impl Gradient1 for Dual {
    fn dual(&self) -> &Array1<f64> {
        &self.dual
    }
}

impl Dual {
    /// Constructs a new `Dual`.
    ///
    /// - `vars` should be **unique**; duplicates will be removed by the `IndexSet`.
    /// - `dual` can be empty; if so each gradient with respect to each `vars` is set to 1.0_f64.
    ///
    /// # Panics
    ///
    /// If the length of `dual` and of `vars` are not the same after parsing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Dual::new(2.5, vec!["x".to_string()], vec![]);
    /// // x: <Dual: 2.5, (x), [1.0]>
    /// ```
    pub fn new(real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        let dual_ = if dual.is_empty() {Array1::ones(unique_vars_.len())} else {Array1::from_vec(dual)};
        assert_eq!(unique_vars_.len(), dual_.len());
        Self {real, vars: unique_vars_, dual: dual_}
    }

    /// Construct a new `Dual` cloning the `vars` Arc pointer from another.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x = Dual::new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]);
    /// let y = Dual::new_from(&x, 1.5, vec["y".to_string()], vec![]);
    /// ```
    ///
    /// This is semantically the same as:
    ///
    /// ```rust
    /// let x = Dual::new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]);
    /// let y = Dual::new(1.5, vec!["y".to_string()], vec![]).to_new_vars(x.vars(), None);
    /// ```
    pub fn new_from(other: &Self, real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
        let new = Self::new(real, vars, dual);
        new.to_new_vars(&other.vars, None)
    }

    /// Get the real component value of the struct.
    pub fn real(&self) -> f64 {
        self.real.clone()
    }
}

impl Num for Dual {  // PartialEq + Zero + One + NumOps (Add + Sub + Mul + Div + Rem)
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual".to_string())
    }
}

// impl Neg for Dual
impl_op!(-|a: Dual| -> Dual {
    Dual {
        vars: a.vars,
        real: -a.real,
        dual: -a.dual,
    }
});
impl_op!(-|a: &Dual| -> Dual {
    Dual {
        vars: Arc::clone(&a.vars),
        real: -a.real,
        dual: &a.dual * -1.0,
    }
});

// impl Add for Dual
impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsState::EquivByArc | VarsState::EquivByVal => {
            Dual {real: a.real + b.real, dual: &a.dual + &b.dual, vars: Arc::clone(&a.vars)}
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual {real: x.real + y.real, dual: &x.dual + &y.dual, vars: Arc::clone(&x.vars)}
        }
    }
});

// impl Sub for Dual
impl_op_ex!(-|a: &Dual, b: &Dual| -> Dual {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsState::EquivByArc | VarsState::EquivByVal => {
            Dual {real: a.real - b.real, dual: &a.dual - &b.dual, vars: Arc::clone(&a.vars)}
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual {real: x.real - y.real, dual: &x.dual - &y.dual, vars: Arc::clone(&x.vars)}
        }
    }
});

// impl Mul for Dual
impl_op_ex!(*|a: &Dual, b: &Dual| -> Dual {
    let state = a.vars_cmp(b.vars());
    match state {
        VarsState::EquivByArc | VarsState::EquivByVal => {
            Dual {real: a.real * b.real, dual:  &a.dual * b.real + &b.dual * a.real, vars: Arc::clone(&a.vars)}
        }
        _ => {
            let (x, y) = a.to_union_vars(b, Some(state));
            Dual {real: x.real * y.real, dual: &x.dual * y.real + &y.dual * x.real, vars: Arc::clone(&x.vars)}
        }
    }
});

// impl Div for Dual
impl_op_ex!(/ |a: &Dual, b: &Dual| -> Dual {
    let b_ = Dual {real: 1.0 / b.real, vars: Arc::clone(&b.vars), dual: -1.0 / (b.real * b.real) * &b.dual};
    a * b_
});

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
            dual: self.dual() * power * self.real.pow(power - 1.0),
        }
    }
}

// impl REM for Dual
impl_op_ex!(% |a: &Dual, b: & Dual| -> Dual {
    let d = f64::trunc(a.real / b.real);
    a - d * b
});

impl One for Dual {
    fn one() -> Dual {
        Dual::new(1.0, Vec::new(), Vec::new())
    }
}

impl Zero for Dual {
    fn zero() -> Dual {
        Dual::new(0.0, Vec::new(), Vec::new())
    }

    fn is_zero(&self) -> bool {
        *self == Dual::new(0.0, Vec::new(), Vec::new())
    }
}

/// Measures value equivalence of `Dual`.
///
/// Returns `true` if:
///
/// - `real` components are equal: `lhs.real == rhs.real`.
/// - `dual` components are equal after aligning `vars`.
impl PartialEq<Dual> for Dual {
    fn eq(&self, other: &Dual) -> bool {
        if self.real != other.real {
            false
        } else {
            let state = self.vars_cmp(other.vars());
            match state {
                VarsState::EquivByArc | VarsState::EquivByVal => {
                    self.dual.iter().eq(other.dual.iter())
                }
                _ => {
                    let (x, y) = self.to_union_vars(other, Some(state));
                    x.dual.iter().eq(y.dual.iter())
                }
            }
        }
    }
}

/// Compares `Dual` by `real` component only.
impl PartialOrd<Dual> for Dual {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl Sum for Dual {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Dual>,
    {
        iter.fold(Dual::new(0.0, [].to_vec(), [].to_vec()), |acc, x| acc + x)
    }
}

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
            Dual {real: self.real, vars: Arc::clone(&self.vars), dual: self.dual.clone()}
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
            Dual::new(0.0, Vec::new(), Vec::new())
        } else {
            other - self
        }
    }

    fn signum(&self) -> Self { Dual::new(self.real.signum(), Vec::new(), Vec::new()) }

    fn is_positive(&self) -> bool { self.real.is_sign_positive() }

    fn is_negative(&self) -> bool { self.real.is_sign_negative() }
}

pub trait MathFuncs {
    fn exp(&self) -> Self;
    fn log(&self) -> Self;
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
}

// f64 Crossover

// Add
impl_op_ex_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
// Sub
impl_op_ex!(-|a: &Dual, b: &f64| -> Dual {
    Dual {
        vars: Arc::clone(&a.vars),
        real: a.real - b,
        dual: a.dual.clone(),
    }
});
impl_op_ex!(-|a: &f64, b: &Dual| -> Dual {
    Dual {
        vars: Arc::clone(&b.vars),
        real: a - b.real,
        dual: -(b.dual.clone()),
    }
});
// Mul
impl_op_ex_commutative!(*|a: &Dual, b: &f64| -> Dual {
    Dual {
        vars: Arc::clone(&a.vars),
        real: a.real * b,
        dual: *b * &a.dual,
    }
});
// Div
impl_op_ex!(/ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real / b, dual: (1_f64/b) * &a.dual} });
impl_op_ex!(/ |a: &f64, b: &Dual| -> Dual { a * b.clone().pow(-1.0) });
// Rem
impl_op_ex!(% |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real % b, dual: a.dual.clone()} });
impl_op_ex!(% |a: &f64, b: &Dual| -> Dual { Dual::new(*a, Vec::new(), Vec::new()) % b });

impl PartialEq<f64> for Dual {
    fn eq(&self, other: &f64) -> bool {
        Dual::new(*other, [].to_vec(), [].to_vec()) == *self
    }
}

impl PartialEq<Dual> for f64 {
    fn eq(&self, other: &Dual) -> bool {
        Dual::new(*self, [].to_vec(), [].to_vec()) == *other
    }
}

impl PartialOrd<f64> for Dual {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {self.real.partial_cmp(other)}
}

impl PartialOrd<Dual> for f64 {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {self.partial_cmp(&other.real)}
}


pub trait FieldOps<T>: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>+ Sized + Clone {}
impl<'a, T: 'a> FieldOps<T> for &'a T
    where &'a T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> {}
impl FieldOps<Dual> for Dual {}
impl FieldOps<f64> for f64 {}


// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_fieldops() {
        fn test_ops<T>(a: &T, b: &T) -> T
        where for <'a> &'a T: FieldOps<T>
        {
            &(a + b) - a
        }

        fn test_ops2<T>(a: T, b: T) -> T
        where T: FieldOps<T>
        {
            (a.clone() + b) - a
        }

         let x = 1.0;
         let y = 2.0;
         let z = test_ops(&x, &y);
         println!("{:?}", z);

         let z = test_ops2(x, y);
         println!("{:?}", z);
    }

    #[test]
    fn new() {
        let x = Dual::new(1.0, vec!["a".to_string(), "a".to_string()], Vec::new());
        assert_eq!(x.real, 1.0_f64);
        assert_eq!(*x.vars, IndexSet::<String>::from_iter(vec!["a".to_string()]));
        assert_eq!(x.dual, Array1::from_vec(vec![1.0_f64]));
    }

    #[test]
    fn new_with_dual() {
        let x = Dual::new(1.0, vec!["a".to_string(), "a".to_string()], vec![2.5]);
        assert_eq!(x.real, 1.0_f64);
        assert_eq!(*x.vars, IndexSet::<String>::from_iter(vec!["a".to_string()]));
        assert_eq!(x.dual, Array1::from_vec(vec![2.5_f64]));
    }

    #[test]
    #[should_panic]
    fn new_len_mismatch() {
        Dual::new(1.0, vec!["a".to_string(), "a".to_string()], vec![1.0, 2.0]);
    }

    #[test]
    fn ptr_eq() {
        let x = Dual::new(1.0, vec!["a".to_string()], vec![]);
        let y = Dual::new(1.0, vec!["a".to_string()], vec![]);
        assert!(x.ptr_eq(&y)==false);
    }

    #[test]
    fn to_new_vars() {
        let x = Dual::new(1.5, vec!["a".to_string(), "b".to_string()], vec![1., 2.]);
        let y = Dual::new(2.0, vec!["a".to_string(), "c".to_string()], vec![3., 3.]);
        let z = x.to_new_vars(&y.vars, None);
        assert_eq!(z.real, 1.5_f64);
        assert!(y.ptr_eq(&z));
        assert_eq!(z.dual, Array1::from_vec(vec![1.0, 0.0]));
        let u = x.to_new_vars(x.vars(), None);
        assert!(u.ptr_eq(&x))
    }

    #[test]
    fn new_from() {
        let x = Dual::new(2.0, vec!["a".to_string(), "b".to_string()], vec![3., 3.]);
        let y = Dual::new_from(&x, 2.0, vec!["a".to_string(), "c".to_string()], vec![3., 3.]);
        assert_eq!(y.real, 2.0_f64);
        assert!(y.ptr_eq(&x));
        assert_eq!(y.dual, Array1::from_vec(vec![3.0, 0.0]));
    }

    #[test]
    fn vars() {
        let x = Dual::new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]);
        let y = Dual::new(1.5, vec!["y".to_string()], vec![]).to_new_vars(x.vars(), None);
        assert!(x.ptr_eq(&y));
        assert_eq!(y.dual, Array1::from_vec(vec![0.0, 1.0]));
    }

    #[test]
    fn vars_cmp() {
        let x = Dual::new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]);
        let y = Dual::new(1.5, vec!["y".to_string()], vec![]);
        let y2 = Dual::new(1.5, vec!["y".to_string()], vec![]);
        let z = x.to_new_vars(y.vars(), None);
        let u = Dual::new(1.5, vec!["u".to_string()], vec![]);
        assert_eq!(x.vars_cmp(y.vars()), VarsState::Superset);
        assert_eq!(y.vars_cmp(z.vars()), VarsState::EquivByArc);
        assert_eq!(y.vars_cmp(y2.vars()), VarsState::EquivByVal);
        assert_eq!(y.vars_cmp(x.vars()), VarsState::Subset);
        assert_eq!(y.vars_cmp(u.vars()), VarsState::Difference);
    }

    #[test]
    fn default() {
        let x = Dual::default();
        assert_eq!(x.real, 0.0_f64);
        assert_eq!(x.vars.len(), 0_usize);
        assert_eq!(x.dual, Array1::<f64>::from_vec(vec![]));
    }

    // OPS TESTS

    #[test]
    fn zero_init() {
        let d = Dual::new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::new(),
        );
        for (_, val) in d.dual.indexed_iter() {
            assert!(*val == 1.0)
        }
    }

    #[test]
    fn negate() {
        let d = Dual::new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([2., -1.4]),
        );
        let d2 = -d.clone();
        assert!(d2.real == -2.3);
        assert!(Arc::ptr_eq(&d.vars, &d2.vars));
        assert!(d2.dual[0] == -2.0);
        assert!(d2.dual[1] == 1.4);
    }

    #[test]
    fn eq_ne() {
        // Dual with vars - f64
        assert!(Dual::new(0.0, Vec::from([String::from("a")]), Vec::new()) != 0.0);
        // Dual with no vars - f64 (+reverse)
        assert!(Dual::new(2.0, Vec::new(), Vec::new()) == 2.0);
        assert!(2.0 == Dual::new(2.0, Vec::new(), Vec::new()));
        // Dual - Dual (various real, vars, gradient mismatch)
        let d = Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([2.3]));
        assert!(d == Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([2.3])));
        assert!(d != Dual::new(2.0, Vec::from([String::from("b")]), Vec::from([2.3])));
        assert!(d != Dual::new(3.0, Vec::from([String::from("a")]), Vec::from([2.3])));
        assert!(d != Dual::new(2.0, Vec::from([String::from("a")]), Vec::from([1.3])));
        // Dual - Dual (missing Vars are zero and upcasted)
        assert!(
            d == Dual::new(
                2.0,
                Vec::from([String::from("a"), String::from("b")]),
                Vec::from([2.3, 0.0])
            )
        );
    }

    #[test]
    fn add_f64() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = 10.0 + d1 + 15.0;
        let expected = Dual::new(
            26.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn add() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let d2 = Dual::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
        );
        let expected = Dual::new(
            3.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, 3.0],
        );
        let result = d1 + d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn sub_f64() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = (10.0 - d1) - 15.0;
        let expected = Dual::new(
            -6.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn sub() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let d2 = Dual::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
        );
        let expected = Dual::new(
            -1.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, -3.0],
        );
        let result = d1 - d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn mul_f64() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = 10.0 * d1 * 2.0;
        let expected = Dual::new(
            20.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![20.0, 40.0],
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn mul() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let d2 = Dual::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
        );
        let expected = Dual::new(
            2.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![2.0, 4.0, 3.0],
        );
        let result = d1 * d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn inv() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = d1.clone() * d1.pow(-1.0);
        let expected = Dual::new(1.0, vec![], vec![]);
        assert!(result == expected)
    }

    #[test]
    fn abs() {
        let d1 = Dual::new(
            -2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = d1.abs();
        let expected = Dual::new(
            2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn div_f64() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = d1 / 2.0;
        let expected = Dual::new(
            0.5,
            vec!["v0".to_string(), "v1".to_string()],
            vec![0.5, 1.0],
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn f64_div() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = 2.0 / d1.clone();
        let expected = Dual::new(2.0, vec![], vec![]) / d1;
        assert_eq!(result, expected)
    }

    #[test]
    fn div() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let d2 = Dual::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
        );
        let expected = Dual::new(
            0.5,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![0.5, 1.0, -0.75],
        );
        let result = d1 / d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn ord() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        assert!(d1 < 2.0);
        assert!(d1 > 0.5);
        assert!(d1 <= 1.0);
        assert!(d1 >= 1.0);
        assert!(1.0 <= d1);
        assert!(1.0 >= d1);
        assert!(2.0 > d1);
        assert!(0.5 < d1);
        let d2 = Dual::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![1.0, 2.0],
        );
        assert!(d2 > d1);
        assert!(d1 < d2);
        let d3 = Dual::new(1.0, vec!["v3".to_string()], vec![10.0]);
        assert!(d1 >= d3);
        assert!(d1 <= d3);
    }

    #[test]
    fn exp() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = d1.exp();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.exp();
        let expected = Dual::new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0 * c, 2.0 * c],
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn log() {
        let d1 = Dual::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
        );
        let result = d1.log();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.ln();
        let expected = Dual::new(c, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
        assert_eq!(result, expected);
    }


    // PROFILING

    #[test]
    fn vars_cmp_profile() {
        // Setup
        let VARS = 500_usize;
        let x = Dual::new(
            1.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (1..=VARS).map(|x| x as f64).collect(),
        );
        let y = Dual::new(
            1.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (1..=VARS).map(|x| x as f64).collect(),
        );
        let z = Dual::new_from(&x, 1.0, Vec::new(), Vec::new());
        let u = Dual::new(
            1.5,
            (1..VARS).map(|x| x.to_string()).collect(),
            (1..VARS).map(|x| x as f64).collect(),
        );
        let s = Dual::new(
            1.5,
            (0..(VARS-1)).map(|x| x.to_string()).collect(),  // 2..Vars+1 13us  0..Vars-1  48ns
            (1..VARS).map(|x| x as f64).collect(),
        );

        println!("\nProfiling vars_cmp (VarsState::EquivByArc):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&z.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100000);

        println!("\nProfiling vars_cmp (VarsState::EquivByVal):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&y.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);

        println!("\nProfiling vars_cmp (VarsState::Superset):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&u.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);

        println!("\nProfiling vars_cmp (VarsState::Different):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&s.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);
    }

    #[test]
    fn to_union_vars_profile() {
        // Setup
        let VARS = 500_usize;
        let x = Dual::new(
            1.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );
        let y = Dual::new(
            1.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );
        let z = Dual::new_from(&x, 1.0, Vec::new(), Vec::new());
        let u = Dual::new(
            1.5,
            (1..VARS).map(|x| x.to_string()).collect(),
            (1..VARS).map(|x| x as f64).collect(),
        );
        let s = Dual::new(
            1.5,
            (0..(VARS-1)).map(|x| x.to_string()).collect(),
            (0..(VARS-1)).map(|x| x as f64).collect(),
        );

        println!("\nProfiling to_union_vars (VarsState::EquivByArc):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&z, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100000);

        println!("\nProfiling to_union_vars (VarsState::EquivByVal):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&y, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);

        println!("\nProfiling to_union_vars (VarsState::Superset):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&u, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100);

        println!("\nProfiling to_union_vars (VarsState::Different):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&s, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100);
    }

    #[test]
    fn std_ops_ref_profile() {
        fn four_ops<T>(a: &T, b: &T, c: &T, d: &T) -> T
        where for <'a> &'a T: Add<&'a T, Output = T>
                            + Sub<&'a T, Output = T>
                            + Div<&'a T, Output = T>
                            + Mul<&'a T, Output = T>
        {
            &(&(a + b) * &(c / d)) - a
        }

        let VARS = 500_usize;
        let a = Dual::new(
            1.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );
        // let b = Dual::new(
        //     3.5,
        //     (2..=(VARS+1)).map(|x| x.to_string()).collect(),
        //     (0..VARS).map(|x| x as f64).collect(),
        // );
        // let c = Dual::new(
        //     5.5,
        //     (3..=(VARS+2)).map(|x| x.to_string()).collect(),
        //     (0..VARS).map(|x| x as f64).collect(),
        // );
        // let d = Dual::new(
        //     6.5,
        //     (4..=(VARS+3)).map(|x| x.to_string()).collect(),
        //     (0..VARS).map(|x| x as f64).collect(),
        // );
        let b = Dual::new_from(
            &a,
            3.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );
        let c = Dual::new_from(
            &a,
            5.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );
        let d = Dual::new_from(
            &a,
            6.5,
            (1..=VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );

        println!("\nProfiling f64 std ops:");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                let x = four_ops(&a, &b, &c, &d);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.9?}", elapsed / 1000);

    }

}

