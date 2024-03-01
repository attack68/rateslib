use crate::dual::dual2::Dual2;
use auto_ops::{impl_op, impl_op_ex, impl_op_ex_commutative};
use indexmap::set::IndexSet;
use ndarray::{Array, Array1};
use num_traits;
use num_traits::identities::{One, Zero};
use num_traits::{Num, Pow, Signed};
use numpy::{Element, PyArray1, PyArray2, PyArrayDescr, ToPyArray};
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};

#[pyclass]
#[derive(Clone, Default)]
pub struct Dual {
    pub real: f64,
    pub vars: Arc<IndexSet<String>>,
    pub dual: Array1<f64>,
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
        } else if self.vars.len() >= other.vars.len()
            && other.vars.iter().all(|var| self.vars.contains(var))
        {
            // vars in other are contained within self
            (self.clone(), other.to_new_ordered_vars(&self.vars))
        } else if self.vars.len() < other.vars.len()
            && self.vars.iter().all(|var| other.vars.contains(var))
        {
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
        let comb_vars = Arc::new(IndexSet::from_iter(
            self.vars.union(&other.vars).map(|x| x.clone()),
        ));
        (self.to_new_vars(&comb_vars), other.to_new_vars(&comb_vars))
    }

    /// Return a Dual with recast vars if required.
    fn to_new_ordered_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        if self.vars.len() == new_vars.len()
            && self.vars.iter().zip(new_vars.iter()).all(|(a, b)| a == b)
        {
            Dual {
                vars: Arc::clone(new_vars),
                real: self.real,
                dual: self.dual.clone(),
            }
        } else {
            self.to_new_vars(new_vars)
        }
    }

    fn to_new_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        // Return a Dual with a new set of vars.

        let mut dual = Array::zeros(new_vars.len());
        for (i, index) in new_vars
            .iter()
            .map(|x| self.vars.get_index_of(x))
            .enumerate()
        {
            match index {
                Some(value) => dual[[i]] = self.dual[[value]],
                None => {}
            }
        }
        Dual {
            vars: Arc::clone(new_vars),
            real: self.real,
            dual,
        }
    }

    // fn is_same_vars(&self, other: &Dual) -> bool {
    //     // test if the vars of a Dual have the same elements but possibly a different order
    //     return self.vars.len() == other.vars.len() && self.vars.intersection(&other.vars).count() == self.vars.len()
    // }

    fn ggradient(&self, vars: Vec<String>) -> Array1<f64> {
        let mut dual = Array::zeros(vars.len());
        for (i, index) in vars.iter().map(|x| self.vars.get_index_of(x)).enumerate() {
            match index {
                Some(value) => dual[[i]] = self.dual[[value]],
                None => dual[[i]] = 0.0,
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

impl fmt::Debug for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.real)
    }
}

impl Num for Dual {
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual".to_string())
    }
}

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

impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real + b.real, dual: &a.dual + &b.dual, vars: Arc::clone(&a.vars)}
    }
    else { let (x, y) = a.to_combined_vars(b); x + y }
});

impl_op_ex!(-|a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {
            real: a.real - b.real,
            dual: &a.dual - &b.dual,
            vars: Arc::clone(&a.vars),
        }
    } else {
        let (x, y) = a.to_combined_vars(b);
        x - y
    }
});

impl_op_ex!(*|a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {
            real: a.real * b.real,
            dual: &a.dual * b.real + &b.dual * a.real,
            vars: Arc::clone(&a.vars),
        }
    } else {
        let (x, y) = a.to_combined_vars(b);
        x * y
    }
});

// Assignment operators might be slower due to clones

// impl std::ops::AddAssign for Dual {
//     fn add_assign(&mut self, other: Self) {
//         let z = self.clone() + other;
//         self.vars = z.vars.clone();
//         self.dual = z.dual.clone();
//         self.real = z.real;
//     }
// }

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
        Dual {
            real: self.real.pow(power),
            vars: self.vars,
            dual: self.dual * power * self.real.pow(power - 1.0),
        }
    }
}

impl_op_ex!(/ |a: &Dual, b: &Dual| -> Dual { a * b.clone().pow(-1.0) });

impl num_traits::identities::One for Dual {
    fn one() -> Dual {
        Dual::new(1.0, Vec::new(), Vec::new())
    }
}

impl num_traits::identities::Zero for Dual {
    fn zero() -> Dual {
        Dual::new(0.0, Vec::new(), Vec::new())
    }

    fn is_zero(&self) -> bool {
        *self == Dual::new(0.0, Vec::new(), Vec::new())
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
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {
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
    where
        I: Iterator<Item = Dual>,
    {
        iter.fold(Dual::new(0.0, [].to_vec(), [].to_vec()), |acc, x| acc + x)
    }
}

impl_op_ex!(% |a: &Dual, b: & Dual| -> Dual {
    let d = f64::trunc(a.real / b.real);
    a - d * b
});

impl Signed for Dual {
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

#[derive(Debug, Clone, PartialEq, PartialOrd, FromPyObject)]
pub enum DualsOrF64 {
    Dual(Dual),
    Dual2(Dual2),
    F64(f64),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, FromPyObject)]
pub enum DualOrF64 {
    Dual(Dual),
    F64(f64),
}

// NumOps with f64

impl_op_ex_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });

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

impl_op_ex_commutative!(*|a: &Dual, b: &f64| -> Dual {
    Dual {
        vars: Arc::clone(&a.vars),
        real: a.real * b,
        dual: *b * &a.dual,
    }
});

impl_op_ex!(/ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real / b, dual: (1_f64/b) * &a.dual} });
impl_op_ex!(/ |a: &f64, b: &Dual| -> Dual { a * b.clone().pow(-1.0) });

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
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
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
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {
        if *self == other.real {
            Some(Ordering::Equal)
        } else if *self < other.real {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

// pub trait LinAlgOps<T>
// where T: PartialOrd + Signed + Clone + Sum + Zero + Add + Mul + Sub + Div + One,
//   for<'a> &'a T: Add<&'a T, Output=T> + Sub<&'a T, Output=T> + Mul<&'a T, Output=T> + Div<&'a T, Output=T>,
// {
// }
//
// pub trait LinAlgOpsF64<T>
// where
//   for<'a> &'a T: Add<&'a f64, Output=T> + Sub<&'a f64, Output=T> + Mul<&'a f64, Output=T> + Div<&'a f64, Output=T>,
//   for<'a> &'a f64: Add<&'a T, Output=T> + Sub<&'a T, Output=T> + Mul<&'a T, Output=T> + Div<&'a T, Output=T>
// {
// }
//
// impl LinAlgOps<Dual> for Dual {}
//
// impl LinAlgOpsF64<Dual> for Dual {}

unsafe impl Element for Dual {
    const IS_COPY: bool = false;
    fn get_dtype(py: Python<'_>) -> &PyArrayDescr {
        PyArrayDescr::object(py)
    }
}

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
    /// let f = Dual::new(2.5, Vec::from([String::from("x")]), Vec::new());
    /// ```

    #[new]
    pub fn new(real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
        let new_dual;
        if !dual.is_empty() && vars.len() != dual.len() {
            panic!("`dual` must have same length as `vars` or have zero length.")
        } else if dual.is_empty() && !vars.is_empty() {
            new_dual = Array::ones(vars.len());
        } else {
            new_dual = Array::from_vec(dual);
        }
        Self {
            real,
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

    #[getter]
    fn dual2<'py>(&'py self, py: Python<'py>) -> PyResult<&PyArray2<f64>> {
        Err(PyValueError::new_err("`Dual` variable cannot possess `dual2` attribute."))
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

    fn __eq__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(d.eq(self)),
            DualsOrF64::F64(f) => Ok(Dual::new(f, Vec::new(), Vec::new()).eq(self)),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __lt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self < &d),
            DualsOrF64::F64(f) => Ok(self < &f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __le__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self <= &d),
            DualsOrF64::F64(f) => Ok(self <= &f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __gt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self > &d),
            DualsOrF64::F64(f) => Ok(self > &f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __ge__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual(d) => Ok(self >= &d),
            DualsOrF64::F64(f) => Ok(self >= &f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Cannot compare Dual with incompatible type (Dual2)."))
        }
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __add__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __radd__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __sub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self - d),
            DualsOrF64::F64(f) => Ok(self - f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __rsub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(d - self),
            DualsOrF64::F64(f) => Ok(f - self),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __mul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self * d),
            DualsOrF64::F64(f) => Ok(self * f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __rmul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(d * self),
            DualsOrF64::F64(f) => Ok(f * self),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __truediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(self / d),
            DualsOrF64::F64(f) => Ok(self / f),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
        }
    }

    fn __rtruediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual(d) => Ok(d / self),
            DualsOrF64::F64(f) => Ok(f / self),
            DualsOrF64::Dual2(d) => Err(PyTypeError::new_err("Dual operation with incompatible type (Dual2)."))
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

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_arc() {
        let d1 = Dual::new(20.0, vec!["a".to_string()], vec![]);
        let d2 = d1.clone();
        assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
    }

    #[test]
    fn default_dual() {
        let result = Dual::default();
        let expected = Dual::new(0.0, Vec::new(), Vec::new());
        assert_eq!(result, expected);
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
        Dual::new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([1.0]),
        );
    }

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
        assert_eq!(result, expected)
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
}
