use numpy::{PyArray, ToPyArray, PyArray1};
use ndarray::{Array1, Array, arr1};
use num_traits;
use num_traits::{Pow, Num, Signed};
use std::sync::Arc;
use std::ops::{Neg, Rem};
use std::num::ParseFloatError;
use indexmap::set::IndexSet;
use std::cmp::Ordering;
use auto_ops::{impl_op, impl_op_commutative, impl_op_ex, impl_op_ex_commutative};


use pyo3::exceptions::PyIndexError;
use pyo3::types::PyFloat;
use pyo3::prelude::*;
use pyo3::conversion::FromPyObject;

// fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
//     // used rather than equality for float numbers
//     return (a-b).abs() < abs_tol.unwrap_or(1e-8)
// }

#[pyclass]
#[derive(Clone, Debug)]
pub struct Dual {
    pub real : f64,
    pub vars : Arc<IndexSet<String>>,
    pub dual : Array1<f64>,
}

impl Dual {
    /// Return two equivalent Dual with same vars.
    ///
    /// # Arguments
    ///
    /// * `other` - Alternative Dual against which vars comparison is made
    ///
    /// # Notes
    ///
    ///
    fn to_combined_vars(&self, other: &Dual) -> (Dual, Dual) {
        if Arc::ptr_eq(&self.vars, &other.vars) {
            (self.clone(), other.clone())
        } else if self.vars.len() >= other.vars.len() && other.vars.iter().all(|var| self.vars.contains(var)) {
            // vars in other are contained within self
            (self.clone(), other.to_new_ordered_vars(&self.vars))
        } else if self.vars.len() < other.vars.len() && self.vars.iter().all(|var| other.vars.contains(var)) {
            // vars in self are contained within other
            (self.to_new_ordered_vars(&other.vars), other.clone())
        } else {
            // vars in both self and other are different so recast
            self.to_combined_vars_explicit(other)
        }
    }

    /// Return two equivalent Dual with the unionised same, but explicitly recast, vars.
    ///
    /// # Arguments
    ///
    /// * `other` - Alternative Dual against which vars comparison is made
    ///
    /// # Notes
    ///
    ///
    fn to_combined_vars_explicit(&self, other: &Dual) -> (Dual, Dual) {
        let comb_vars = Arc::new(IndexSet::from_iter(self.vars.union(&other.vars).map(|x| x.clone())));
        (self.to_new_vars(&comb_vars), other.to_new_vars(&comb_vars))
    }

    /// Return a Dual with recast vars if required.
    fn to_new_ordered_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        if self.vars.len() == new_vars.len() && self.vars.iter().zip(new_vars.iter()).all(|(a,b)| a==b) {
            Dual {vars: Arc::clone(new_vars), real: self.real, dual: self.dual.clone()}
        } else {
            self.to_new_vars(new_vars)
        }
    }

    fn to_new_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        // Return a Dual with a new set of vars.

        let mut dual = Array::zeros(new_vars.len());
        for (i, index) in new_vars.iter().map(|x| self.vars.get_index_of(x)).enumerate() {
            match index {
                Some(value) => { dual[[i]] = self.dual[[value]] }
                None => {}
            }
        }
        Dual {vars: Arc::clone(new_vars), real: self.real, dual}
    }

    // fn is_same_vars(&self, other: &Dual) -> bool {
    //     // test if the vars of a Dual have the same elements but possibly a different order
    //     return self.vars.len() == other.vars.len() && self.vars.intersection(&other.vars).count() == self.vars.len()
    // }

    fn ggradient(&self, vars: Vec<String>) -> Array1<f64> {
        let mut dual = Array::zeros(vars.len());
        for (i, index) in vars.iter().map(|x| self.vars.get_index_of(x)).enumerate() {
            match index {
                Some(value) => { dual[[i]] = self.dual[[value]] }
                None => { dual[[i]] = 0.0 }
            }
        }
        dual
    }

    pub fn exp(&self) -> Self {
        let c = self.real.exp();
        Dual {
            real: c,
            vars: Arc::clone(&self.vars),
            dual: c * &self.dual,
        }
    }

    pub fn log(&self) -> Self {
        Dual {
            real: self.real.ln(),
            vars: Arc::clone(&self.vars),
            dual: (1.0 / self.real) * &self.dual,
        }
    }
}

impl Num for Dual {
    type FromStrRadixErr = String;
    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual".to_string())
    }
}

impl_op!(- |a: Dual| -> Dual { Dual {vars: a.vars, real: -a.real, dual: -a.dual}});
impl_op!(- |a: &Dual| -> Dual { Dual {vars: Arc::clone(&a.vars), real: -a.real, dual: &a.dual * -1.0}});

impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real + b.real, dual: &a.dual + &b.dual, vars: Arc::clone(&a.vars)}
    }
    else { let (x, y) = a.to_combined_vars(b); x + y }
});

// impl std::ops::AddAssign for Dual {
//     fn add_assign(&mut self, other: Self) {
//         let z = self.clone() + other;
//         self.vars = z.vars.clone();
//         self.dual = z.dual.clone();
//         self.real = z.real;
//     }
// }

impl_op_ex!(- |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real - b.real, dual: &a.dual - &b.dual, vars: Arc::clone(&a.vars)}
    }
    else { let (x, y) = a.to_combined_vars(b); x - y }
});

impl_op_ex!(* |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real * b.real, dual: &a.dual * b.real + &b.dual * a.real, vars: Arc::clone(&a.vars)}
    }
    else { let (x, y) = a.to_combined_vars(b); x * y }
});
//
// impl std::ops::MulAssign<DUal> for Dual {
//     fn mul_assign(&mut self, other: Dual) {
//         let z = self.clone() * other;
//         self.vars = z.vars.clone();
//         self.dual = z.dual.clone();
//         self.real = z.real;
//     }
// }

impl num_traits::Pow<f64> for Dual {
    type Output = Dual;
    fn pow(self, power: f64) -> Dual {
        return Dual {
            real: self.real.pow(power),
            vars: self.vars,
            dual: self.dual * power * self.real.pow(power - 1.0),
        }
    }
}

impl_op_ex!(/ |a: &Dual, b: &Dual| -> Dual { a * b.clone().pow(-1.0) });

impl num_traits::identities::One for Dual {
    fn one() -> Dual {
        return Dual::new(1.0, Vec::new(), Vec::new())
    }
}

impl num_traits::identities::Zero for Dual {
    fn zero() -> Dual {
        return Dual::new(0.0, Vec::new(), Vec::new())
    }

    fn is_zero(&self) -> bool {
        return *self == Dual::new(0.0, Vec::new(), Vec::new())
    }
}

impl PartialEq<Dual> for Dual {
    fn eq(&self, other: &Dual) -> bool {
        if self.real != other.real {
            false
        } else if Arc::ptr_eq(&self.vars, &other.vars) {
            let boo = self.dual.iter().eq(other.dual.iter());
            boo
        } else {
            let (x, y) = self.to_combined_vars(other);
            x.eq(&y)
        }
    }
}

impl PartialOrd<Dual> for Dual {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering>{
        if self.real == other.real {
            Some(Ordering::Equal)
        } else if self.real < other.real {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

impl std::iter::Sum for Dual {
    fn sum<I>(iter: I) -> Self
    where I: Iterator<Item = Dual> {
        return iter.fold(Dual::new(0.0, [].to_vec(), [].to_vec()), |acc, x| acc + x)
    }
}

impl_op_ex!(% |a: &Dual, b: & Dual| -> Dual {
    let d = f64::trunc(a.real / b.real);
    a - d * b
});

impl Signed for Dual {
    fn abs(&self) -> Self {
        if self.real > 0.0 { Dual{real: self.real, vars: Arc::clone(&self.vars), dual: -1.0 * &self.dual} }
        else {Dual{real: -self.real, vars: Arc::clone(&self.vars), dual: -1.0 * &self.dual }}
    }

    fn abs_sub(&self, other: &Self) -> Self {
        if self <= other {
            Dual::new(0.0, Vec::new(), Vec::new())
        } else {
            other - self
        }
    }

    fn signum(&self) -> Self {
       if self.real == 0.0 {
           Dual::new(1.0, Vec::new(), Vec::new())
       } else if self.real > 0.0 {
           Dual::new(0.0, Vec::new(), Vec::new())
       } else {
           Dual::new(-1.0, Vec::new(), Vec::new())
       }
    }

    fn is_positive(&self) -> bool {
        self.real > 0.0_f64
    }

    fn is_negative(&self) -> bool {
        self.real < 0.0_f64
    }
}



// impl Add<Dual> for Dual {
//     type Output = Dual;
//     fn add(self, other: Dual) -> Dual {
//         if Arc::ptr_eq(&self.vars, &other.vars) {
//             Dual {real: self.real + other.real, dual: self.dual + other.dual, vars: self.vars}
//         } else {
//             let (x, y) = self.to_combined_vars(other);
//             x + y
//         }
//     }
// }
//
// impl Add<&Dual> for Dual {
//     type Output = Dual;
//     fn add(self, other: &Dual) -> Dual {
//         if Arc::ptr_eq(&self.vars, &other.vars) {
//             Dual {real: self.real + other.real, dual: self.dual + other.dual, vars: self.vars}
//         } else {
//             let (x, y) = self.to_combined_vars(other);
//             x + y
//         }
//     }
// }
//
// impl Add<Dual> for &Dual {
//     type Output = Dual;
//     fn add(self, other: Dual) -> Dual {
//         if Arc::ptr_eq(&self.vars, &other.vars) {
//             Dual {real: self.real + other.real, dual: self.dual + other.dual, vars: other.vars}
//         } else {
//             let (x, y) = self.to_combined_vars(other);
//             x + y
//         }
//     }
// }
//
// impl Add<&Dual> for &Dual {
//     type Output = Dual;
//     fn add(self, other: &Dual) -> Dual {
//         if Arc::ptr_eq(&self.vars, &other.vars) {
//             Dual {real: self.real + other.real, dual: self.dual + other.dual, vars: self.vars.clone()}
//         } else {
//             let (x, y) = self.to_combined_vars(other);
//             x + y
//         }
//     }
// }


#[derive(Debug, Clone, PartialEq, PartialOrd, FromPyObject)]
pub enum DualOrF64 {
    Dual(Dual),
    F64(f64),
}

// impl DualOrF64 {
//     pub fn abs(&self) -> Self {
//         match self {
//             DualOrF64::Dual(d) => DualOrF64::F64(d.abs()),
//             DualOrF64::F64(f) => DualOrF64::F64(f.abs()),
//         }
//     }
// }


#[pymethods]
impl Dual {
    /// Return a Dual with associated metrics.
    ///
    /// # Arguments
    ///
    /// * `real` - An f64 holding the representative value of the function.
    /// * `vars` - A Vec of String that labels the variables of the function. Must contain unique
    ///            values.
    /// * `dual` - A Vec of f64 that contains the first derivative information of the function.
    ///            Must be same length as `vars` or empty.
    ///
    /// # Notes
    ///
    /// If `dual` is an empty vector it will be automatically set to vector of 1.0's with the same
    /// length as `vars`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::dual::dual1::Dual;
    /// let f = Dual(2.5, Vec::from([String::from("x")]), Vec::new())
    /// ```

    #[new]
    pub fn new(real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
        let new_dual;
        if dual.len() != 0 && vars.len() != dual.len() {
            panic!("`dual` must have same length as `vars` or have zero length.")
        } else if dual.len() == 0 && vars.len() > 0 {
            new_dual = Array::ones(vars.len());
        } else {
            new_dual = Array::from_vec(dual);
        }
        Self {
            real: real,
            vars: Arc::new(IndexSet::from_iter(vars)),
            dual: new_dual,
        }
    }

    #[getter]
    fn real(&self) -> PyResult<f64> {
        Ok(self.real)
    }

    #[getter]
    fn vars(&self) -> PyResult<Vec<&String>> {
        Ok(Vec::from_iter(self.vars.iter()))
    }

    #[getter]
    fn dual<'py>(&'py self, py: Python<'py>) -> PyResult<&PyArray1<f64>> {
        Ok(self.dual.to_pyarray(py))
    }

    fn gradient<'py>(&'py self, py: Python<'py>, vars: Vec<String>) -> PyResult<&PyArray1<f64>> {
        Ok(self.ggradient(vars).to_pyarray(py))
    }

    fn arc_check(&self, other: &Dual) -> PyResult<bool> {
        Ok(Arc::ptr_eq(&self.vars, &other.vars))
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut _vars = Vec::from_iter(self.vars.iter().take(3).map(String::as_str)).join(", ");
        let mut _dual = Vec::from_iter(self.dual.iter().take(3).map(|x| x.to_string())).join(", ");
        if self.vars.len() > 3 {
            _vars.push_str(", ...");
            _dual.push_str(", ...");
        }
        let fs = format!("<Dual: {:.6}, ({}), [{}]>", self.real, _vars, _dual);
        Ok(fs)
    }

    fn __eq__(&self, other:DualOrF64) -> PyResult<bool> {
        match other {
            DualOrF64::Dual(d) => Ok(d.eq(self)),
            DualOrF64::F64(f) => Ok(Dual::new(f, Vec::new(), Vec::new()).eq(self))
        }
    }

    fn __lt__(&self, other:DualOrF64) -> PyResult<bool> {
        match other {
            DualOrF64::Dual(d) => Ok(self < &d),
            DualOrF64::F64(f) => Ok(self < &f)
        }
    }

    fn __le__(&self, other:DualOrF64) -> PyResult<bool> {
        match other {
            DualOrF64::Dual(d) => Ok(self <= &d),
            DualOrF64::F64(f) => Ok(self <= &f)
        }
    }

    fn __gt__(&self, other:DualOrF64) -> PyResult<bool> {
        match other {
            DualOrF64::Dual(d) => Ok(self > &d),
            DualOrF64::F64(f) => Ok(self > &f)
        }
    }

    fn __ge__(&self, other:DualOrF64) -> PyResult<bool> {
        match other {
            DualOrF64::Dual(d) => Ok(self >= &d),
            DualOrF64::F64(f) => Ok(self >= &f)
        }
    }

    fn __neg__(&self) -> Self {-self}

    fn __add__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => self + d,
            DualOrF64::F64(f) => self + f
        }
    }

    fn __radd__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => self + d,
            DualOrF64::F64(f) => self + f
        }
    }

    fn __sub__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => self - d,
            DualOrF64::F64(f) => self - f
        }
    }

    fn __rsub__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => d - self,
            DualOrF64::F64(f) => f - self
        }
    }

    fn __mul__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => self * d,
            DualOrF64::F64(f) => self * f
        }
    }

    fn __rmul__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => d * self,
            DualOrF64::F64(f) => f * self
        }
    }

    fn __truediv__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => self / d,
            DualOrF64::F64(f) => self / f
        }
    }

    fn __rtruediv__(&self, other: DualOrF64) -> Self {
        match other {
            DualOrF64::Dual(d) => d / self,
            DualOrF64::F64(f) => f / self
        }
    }

    fn __pow__(&self, power: f64, modulo: Option<i32>) -> Self {
        if modulo.unwrap_or(0) != 0 {
            panic!("Power function with mod not available for Dual.")
        }
        self.clone().pow(power)
    }

    fn __exp__(&self) -> Self {
        self.exp()
    }

    fn __abs__(&self) -> f64 {
        self.abs().real
    }

    fn __log__(&self) -> Self {
        self.log()
    }

    fn __float__(&self) -> f64 {
        self.real
    }
}









// impl num_traits::Num for Dual {
//
//     fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
//          num_traits::Num::FromStrRadixErr("No implementation of Dual from str.")
//     }
// }

// impl num_traits::Num::FromStrRadixErr for Dual {
//     fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
//         Err("No implementation of Dual from str.")
//     }
// }

// Addition
impl_op_ex_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
// impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
//     if Arc::ptr_eq(&a.vars, &b.vars) {
//         Dual {real: a.real + b.real, dual: &a.dual + &b.dual, vars: Arc::clone(&a.vars)}
//     }
//     else {
//         let (x, y) = a.to_combined_vars(b);
//         x + y
//     }
// });

// Subtraction
impl_op_ex!(- |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real - b, dual: a.dual.clone()} });
impl_op_ex!(- |a: &f64, b: &Dual| -> Dual { Dual {vars: Arc::clone(&b.vars), real: a - b.real, dual: -(b.dual.clone())} });

// multiplication
impl_op_ex_commutative!(* |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real * b, dual: *b * &a.dual} });

// division
impl_op_ex!(/ |a: &Dual, b: f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real / b, dual: &a.dual / b} });
impl_op_ex!(/ |a: f64, b: &Dual| -> Dual { a * b.clone().pow(-1.0) });

// remainder
impl_op_ex!(% |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real % b, dual: a.dual.clone()} });
impl_op_ex!(% |a: &f64, b: &Dual| -> Dual { Dual::new(*a, Vec::new(), Vec::new()) % b });

impl PartialEq<f64> for Dual {
    fn eq(&self, other: &f64) -> bool {
        return Dual::new(*other, [].to_vec(), [].to_vec()) == *self;
    }
}

impl PartialEq<Dual> for f64 {
    fn eq(&self, other: &Dual) -> bool {
        return Dual::new(*self, [].to_vec(), [].to_vec()) == *other;
    }
}

impl PartialOrd<f64> for Dual {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering>{
        if self.real == *other {
            Some(Ordering::Equal)
        } else if self.real < *other {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

impl PartialOrd<Dual> for f64 {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering>{
        if *self == other.real {
            Some(Ordering::Equal)
        } else if *self < other.real {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}





pub fn arr1_dot(a1: Array1<Dual>, a2: Array1<Dual>) -> Dual {
    // Consumes two one dimensional arrays and produces a scalar value of their dot product.
    let z = a1.into_iter().zip(a2.into_iter()).map(|(x, y)| x * y).collect::<Vec<Dual>>();
    return z.into_iter().sum::<Dual>()
}



// UNIT TESTS


#[test]
fn clone_arc() {
    let d1 = Dual::new(20.0, vec!["a".to_string()], vec![]);
    let d2 = d1.clone();
    assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
}

#[test]
fn to_new_ordered_vars() {
    let d1 = Dual::new(20.0, vec!["a".to_string()], vec![]);
    let d2 = Dual::new(20.0, vec!["a".to_string(), "b".to_string()], vec![]);
    let d3 = d1.to_new_ordered_vars(&d2.vars);
    assert!(Arc::ptr_eq(&d3.vars, &d2.vars));
    assert!(d3.dual.len() == 2);
    let d4 = d2.to_new_ordered_vars(&d1.vars);
    assert!(Arc::ptr_eq(&d4.vars, &d1.vars));
    assert!(d4.dual.len() == 1);
}

#[test]
fn new_dual() {
    Dual::new(2.3, Vec::from([String::from("a")]), Vec::new());
}

#[test]
#[should_panic]
fn new_dual_panic() {
    Dual::new(2.3, Vec::from([String::from("a"), String::from("b")]), Vec::from([1.0]));
}

#[test]
fn zero_init() {
    let d = Dual::new(2.3, Vec::from([String::from("a"), String::from("b")]), Vec::new());
    for (_, val) in d.dual.indexed_iter() {
        assert!(*val == 1.0)
    }
}

#[test]
fn negate() {
    let d = Dual::new(2.3, Vec::from([String::from("a"), String::from("b")]), Vec::from([2., -1.4]));
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
    assert!(d == Dual::new(2.0, Vec::from([String::from("a"), String::from("b")]), Vec::from([2.3, 0.0])));
}

#[test]
fn add_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = 10.0 + d1 + 15.0;
    let expected = Dual::new(26.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    assert_eq!(result, expected)
}

#[test]
fn add() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(3.0, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![1.0, 2.0, 3.0]);
    let result = d1 + d2;
    assert_eq!(result, expected)
}

#[test]
fn sub_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = (10.0 - d1) -15.0;
    let expected = Dual::new(-6.0, vec!["v0".to_string(), "v1".to_string()], vec![-1.0, -2.0]);
    assert_eq!(result, expected)
}

#[test]
fn sub() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(-1.0, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![1.0, 2.0, -3.0]);
    let result = d1 - d2;
    assert_eq!(result, expected)
}

#[test]
fn mul_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = 10.0 * d1 * 2.0;
    let expected = Dual::new(20.0, vec!["v0".to_string(), "v1".to_string()], vec![20.0, 40.0]);
    assert_eq!(result, expected)
}

#[test]
fn mul() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(2.0, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![2.0, 4.0, 3.0]);
    let result = d1 * d2;
    assert_eq!(result, expected)
}

#[test]
fn inv() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1.clone() * d1.pow(-1.0);
    let expected = Dual::new(1.0, vec![], vec![]);
    assert_eq!(result, expected)
}

#[test]
fn abs() {
    let d1 = Dual::new(-2.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1.abs();
    let expected = Dual::new(2.0, vec!["v0".to_string(), "v1".to_string()], vec![-1.0, -2.0]);
    assert_eq!(result, expected)
}

#[test]
fn div_f64() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1 / 2.0;
    let expected = Dual::new(0.5, vec!["v0".to_string(), "v1".to_string()], vec![0.5, 1.0]);
    assert_eq!(result, expected)
}

#[test]
fn f64_div() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = 2.0 / d1.clone();
    let expected = Dual::new(2.0, vec![], vec![]) / d1;
    assert_eq!(result, expected)
}

#[test]
fn div() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![0.0, 3.0]);
    let expected = Dual::new(0.5, vec!["v0".to_string(), "v1".to_string(), "v2".to_string()], vec![0.5, 1.0, -0.75]);
    let result = d1 / d2;
    assert_eq!(result, expected)
}

#[test]
fn ord() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    assert!(d1 < 2.0);
    assert!(d1 > 0.5);
    assert!(d1 <= 1.0);
    assert!(d1 >= 1.0);
    assert!(1.0 <= d1);
    assert!(1.0 >= d1);
    assert!(2.0 > d1);
    assert!(0.5 < d1);
    let d2 = Dual::new(2.0, vec!["v0".to_string(), "v2".to_string()], vec![1.0, 2.0]);
    assert!(d2 > d1);
    assert!(d1 < d2);
    let d3 = Dual::new(1.0, vec!["v3".to_string()], vec![10.0]);
    assert!(d1 >= d3);
    assert!(d1 <= d3);
}

#[test]
fn exp() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1.exp();
    assert!(Arc::ptr_eq(&d1.vars, &result.vars));
    let c = 1.0_f64.exp();
    let expected = Dual::new(c, vec!["v0".to_string(), "v1".to_string()], vec![1.0 * c, 2.0* c]);
    assert_eq!(result, expected);
}

#[test]
fn log() {
    let d1 = Dual::new(1.0, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    let result = d1.log();
    assert!(Arc::ptr_eq(&d1.vars, &result.vars));
    let c = 1.0_f64.ln();
    let expected = Dual::new(c, vec!["v0".to_string(), "v1".to_string()], vec![1.0, 2.0]);
    assert_eq!(result, expected);
}