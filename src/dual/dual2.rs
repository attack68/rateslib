use crate::dual::linalg_f64::outer11_;
use crate::dual::dual1::DualsOrF64;
use auto_ops::{impl_op, impl_op_ex, impl_op_ex_commutative};
use indexmap::set::IndexSet;
use ndarray::{Array, Array1, Array2};
use num_traits;
use num_traits::identities::{One, Zero};
use num_traits::{Num, Pow, Signed};
use numpy::{Element, PyArray, PyArray1, PyArray2, PyArrayDescr, ToPyArray};
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
pub struct Dual2 {
    pub real: f64,
    pub vars: Arc<IndexSet<String>>,
    pub dual: Array1<f64>,
    pub dual2: Array2<f64>,
}

impl Dual2 {
    /// Return two equivalent Dual with same vars.
    ///
    /// # Arguments
    ///
    /// * `other` - Alternative Dual against which vars comparison is made
    ///
    /// # Notes
    ///
    ///
    fn to_combined_vars(&self, other: &Dual2) -> (Dual2, Dual2) {
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
    fn to_combined_vars_explicit(&self, other: &Dual2) -> (Dual2, Dual2) {
        let comb_vars = Arc::new(IndexSet::from_iter(
            self.vars.union(&other.vars).map(|x| x.clone()),
        ));
        (self.to_new_vars(&comb_vars), other.to_new_vars(&comb_vars))
    }

    /// Return a Dual with recast vars if required.
    fn to_new_ordered_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual2 {
        if self.vars.len() == new_vars.len()
            && self.vars.iter().zip(new_vars.iter()).all(|(a, b)| a == b)
        {
            Dual2 {
                vars: Arc::clone(new_vars),
                real: self.real,
                dual: self.dual.clone(),
                dual2: self.dual2.clone(),
            }
        } else {
            self.to_new_vars(new_vars)
        }
    }

    fn to_new_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual2 {
        // Return a Dual with a new set of vars.
        let indices: Vec<Option<usize>> =
            new_vars.iter().map(|x| self.vars.get_index_of(x)).collect();
        let mut dual = Array::zeros(new_vars.len());

        for (i, index) in indices.iter().enumerate() {
            match index {
                Some(value) => dual[i] = self.dual[*value],
                None => {}
            }
        }

        let mut dual2 = Array::zeros((new_vars.len(), new_vars.len()));
        for (i, row_index) in indices.iter().enumerate() {
            for (j, col_index) in indices.iter().enumerate() {
                match row_index {
                    Some(row_value) => match col_index {
                        Some(col_value) => dual2[[i, j]] = self.dual2[[*row_value, *col_value]],
                        None => {}
                    },
                    None => {}
                }
            }
        }

        Dual2 {
            vars: Arc::clone(new_vars),
            real: self.real,
            dual,
            dual2,
        }
    }

    fn grad1(&self, vars: Vec<String>) -> Array1<f64> {
        let mut dual = Array::zeros(vars.len());
        for (i, index) in vars.iter().map(|x| self.vars.get_index_of(x)).enumerate() {
            match index {
                Some(value) => dual[i] = self.dual[value],
                None => dual[i] = 0.0,
            }
        }
        dual
    }

    fn grad2(&self, vars: Vec<String>) -> Array2<f64> {
        let indices: Vec<Option<usize>> = vars.iter().map(|x| self.vars.get_index_of(x)).collect();

        let mut dual2 = Array::zeros((vars.len(), vars.len()));
        for (i, row_index) in indices.iter().enumerate() {
            for (j, col_index) in indices.iter().enumerate() {
                match row_index {
                    Some(row_value) => match col_index {
                        Some(col_value) => dual2[[i, j]] = self.dual2[[*row_value, *col_value]],
                        None => {}
                    },
                    None => {}
                }
            }
        }
        2_f64 * dual2
    }

    pub fn exp(&self) -> Self {
        let c = self.real.exp();
        Dual2 {
            real: c,
            vars: Arc::clone(&self.vars),
            dual: c * &self.dual,
            dual2: c * (&self.dual2 + 0.5 * outer11_(&self.dual.view(), &self.dual.view())),
        }
    }

    pub fn log(&self) -> Self {
        let scalar = 1.0 / self.real;
        Dual2 {
            real: self.real.ln(),
            vars: Arc::clone(&self.vars),
            dual: scalar * &self.dual,
            dual2: scalar * &self.dual2
                - outer11_(&self.dual.view(), &self.dual.view()) * 0.5 * (scalar * scalar),
        }
    }
}

impl fmt::Debug for Dual2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.real)
    }
}

impl Num for Dual2 {
    type FromStrRadixErr = String;
    fn from_str_radix(_src: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Err("No implementation for sting radix for Dual2".to_string())
    }
}

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

impl_op_ex!(+ |a: &Dual2, b: &Dual2| -> Dual2 {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual2 {
            real: a.real + b.real,
            dual: &a.dual + &b.dual,
            vars: Arc::clone(&a.vars),
            dual2: &a.dual2 + &b.dual2,
        }
    }
    else { let (x, y) = a.to_combined_vars(b); x + y }
});

impl_op_ex!(-|a: &Dual2, b: &Dual2| -> Dual2 {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual2 {
            real: a.real - b.real,
            dual: &a.dual - &b.dual,
            vars: Arc::clone(&a.vars),
            dual2: &a.dual2 - &b.dual2,
        }
    } else {
        let (x, y) = a.to_combined_vars(b);
        x - y
    }
});

impl_op_ex!(*|a: &Dual2, b: &Dual2| -> Dual2 {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        let mut dual2: Array2<f64> = &a.dual2 * b.real + &b.dual2 * a.real;
        let cross_beta = outer11_(&a.dual.view(), &b.dual.view());
        dual2 = dual2 + 0.5_f64 * (&cross_beta + &cross_beta.t());
        Dual2 {
            real: a.real * b.real,
            dual: &a.dual * b.real + &b.dual * a.real,
            vars: Arc::clone(&a.vars),
            dual2,
        }
    } else {
        let (x, y) = a.to_combined_vars(b);
        x * y
    }
});

impl num_traits::Pow<f64> for Dual2 {
    type Output = Dual2;
    fn pow(self, power: f64) -> Dual2 {
        let coeff = power * self.real.powf(power - 1.);
        let coeff2 = 0.5 * power * (power - 1.) * self.real.powf(power - 2.);
        let beta_cross = outer11_(&self.dual.view(), &self.dual.view());
        Dual2 {
            real: self.real.powf(power),
            vars: self.vars,
            dual: self.dual * coeff,
            dual2: self.dual2 * coeff + beta_cross * coeff2,
        }
    }
}

impl_op_ex!(/ |a: &Dual2, b: &Dual2| -> Dual2 { a * b.clone().pow(-1.0) });

impl num_traits::identities::One for Dual2 {
    fn one() -> Dual2 {
        Dual2::new(1.0, Vec::new(), Vec::new(), Vec::new())
    }
}

impl num_traits::identities::Zero for Dual2 {
    fn zero() -> Dual2 {
        Dual2::new(0.0, Vec::new(), Vec::new(), Vec::new())
    }

    fn is_zero(&self) -> bool {
        *self == Dual2::new(0.0, Vec::new(), Vec::new(), Vec::new())
    }
}

impl PartialEq<Dual2> for Dual2 {
    fn eq(&self, other: &Dual2) -> bool {
        if self.real != other.real {
            false
        } else if Arc::ptr_eq(&self.vars, &other.vars) {
            let boo = self.dual.iter().eq(other.dual.iter());
            let boo2 = self.dual2.iter().eq(other.dual2.iter());
            boo && boo2
        } else {
            let (x, y) = self.to_combined_vars(other);
            x.eq(&y)
        }
    }
}

impl PartialOrd<Dual2> for Dual2 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {
        if self.real == other.real {
            Some(Ordering::Equal)
        } else if self.real < other.real {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

impl std::iter::Sum for Dual2 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Dual2>,
    {
        iter.fold(
            Dual2::new(0.0, Vec::new(), Vec::new(), Vec::new()),
            |acc, x| acc + x,
        )
    }
}

impl_op_ex!(% |a: &Dual2, b: & Dual2| -> Dual2 {
    let d = f64::trunc(a.real / b.real);
    a - d * b
});

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
            Dual2::new(0.0, Vec::new(), Vec::new(), Vec::new())
        } else {
            other - self
        }
    }

    fn signum(&self) -> Self {
        if self.real == 0.0 {
            Dual2::new(1.0, Vec::new(), Vec::new(), Vec::new())
        } else if self.real > 0.0 {
            Dual2::new(0.0, Vec::new(), Vec::new(), Vec::new())
        } else {
            Dual2::new(-1.0, Vec::new(), Vec::new(), Vec::new())
        }
    }

    fn is_positive(&self) -> bool {
        self.real > 0.0_f64
    }

    fn is_negative(&self) -> bool {
        self.real < 0.0_f64
    }
}

// NumOps with f64

impl_op_ex_commutative!(+ |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone(), dual2: a.dual2.clone()}
});

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

impl_op_ex_commutative!(*|a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {
        vars: Arc::clone(&a.vars),
        real: a.real * b,
        dual: *b * &a.dual,
        dual2: *b * &a.dual2,
    }
});

impl_op_ex!(/ |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real / b, dual: (1_f64/b) * &a.dual, dual2: (1_f64/b) * &a.dual2}
});
impl_op_ex!(/ |a: &f64, b: &Dual2| -> Dual2 { a * b.clone().pow(-1.0) });

impl_op_ex!(% |a: &Dual2, b: &f64| -> Dual2 {
    Dual2 {vars: Arc::clone(&a.vars), real: a.real % b, dual: a.dual.clone(), dual2: a.dual2.clone()}
});
impl_op_ex!(% |a: &f64, b: &Dual2| -> Dual2 {
    Dual2::new(*a, Vec::new(), Vec::new(), Vec::new()) % b }
);

impl PartialEq<f64> for Dual2 {
    fn eq(&self, other: &f64) -> bool {
        Dual2::new(*other, Vec::new(), Vec::new(), Vec::new()) == *self
    }
}

impl PartialEq<Dual2> for f64 {
    fn eq(&self, other: &Dual2) -> bool {
        Dual2::new(*self, Vec::new(), Vec::new(), Vec::new()) == *other
    }
}

impl PartialOrd<f64> for Dual2 {
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

impl PartialOrd<Dual2> for f64 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {
        if *self == other.real {
            Some(Ordering::Equal)
        } else if *self < other.real {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

unsafe impl Element for Dual2 {
    const IS_COPY: bool = false;
    fn get_dtype(py: Python<'_>) -> &PyArrayDescr {
        PyArrayDescr::object(py)
    }
}

#[pymethods]
impl Dual2 {
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
    /// use crate::dual::dual2::Dual2;
    /// let f = Dual2(2.5, Vec::from([String::from("x")]), Vec::new(), Vec::new(), Vec::new());
    /// ```

    #[new]
    pub fn new(real: f64, vars: Vec<String>, dual: Vec<f64>, dual2: Vec<f64>) -> Self {
        let new_dual;
        if !dual.is_empty() && vars.len() != dual.len() {
            panic!("`dual` must have same length as `vars` or have zero length.")
        } else if dual.is_empty() && !vars.is_empty() {
            new_dual = Array::ones(vars.len());
        } else {
            new_dual = Array::from_vec(dual);
        }

        let new_dual2;
        if !dual2.is_empty() && (vars.len() * vars.len()) != dual2.len() {
            panic!("`dual2` must have same length as `vars` squared or have zero length.")
        } else if dual2.is_empty() && !vars.is_empty() {
            new_dual2 = Array::zeros((vars.len(), vars.len()));
        } else {
            new_dual2 = Array::from_vec(dual2)
                .into_shape((vars.len(), vars.len()))
                .expect("`dual2` was not a vector of correct length");
        }

        Self {
            real,
            vars: Arc::new(IndexSet::from_iter(vars)),
            dual: new_dual,
            dual2: new_dual2,
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
        Ok(self.dual2.to_pyarray(py))
    }

    fn gradient<'py>(&'py self, py: Python<'py>, vars: Vec<String>) -> PyResult<&PyArray1<f64>> {
        Ok(self.grad1(vars).to_pyarray(py))
    }

    fn gradient2<'py>(&'py self, py: Python<'py>, vars: Vec<String>) -> PyResult<&PyArray2<f64>> {
        Ok(self.grad2(vars).to_pyarray(py))
    }

    fn arc_check(&self, other: &Dual2) -> PyResult<bool> {
        Ok(Arc::ptr_eq(&self.vars, &other.vars))
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut _vars = Vec::from_iter(self.vars.iter().take(3).map(String::as_str)).join(", ");
        let mut _dual = Vec::from_iter(self.dual.iter().take(3).map(|x| x.to_string())).join(", ");
        if self.vars.len() > 3 {
            _vars.push_str(", ...");
            _dual.push_str(", ...");
        }
        let fs = format!(
            "<Dual2: {:.6}, ({}), [{}], [[...]]>",
            self.real, _vars, _dual
        );
        Ok(fs)
    }

    fn __eq__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d.eq(self)),
            DualsOrF64::F64(f) => Ok(Dual2::new(f, Vec::new(), Vec::new(), Vec::new()).eq(self)),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __lt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self < &d),
            DualsOrF64::F64(f) => Ok(self < &f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __le__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self <= &d),
            DualsOrF64::F64(f) => Ok(self <= &f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __gt__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self > &d),
            DualsOrF64::F64(f) => Ok(self > &f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __ge__(&self, other: DualsOrF64) -> PyResult<bool> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self >= &d),
            DualsOrF64::F64(f) => Ok(self >= &f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Cannot compare Dual2 with incompatible type (Dual)."))
        }
    }

    fn __neg__(&self) -> Self {
        -self
    }

    fn __add__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __radd__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self + d),
            DualsOrF64::F64(f) => Ok(self + f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __sub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self - d),
            DualsOrF64::F64(f) => Ok(self - f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __rsub__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d - self),
            DualsOrF64::F64(f) => Ok(f - self),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __mul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self * d),
            DualsOrF64::F64(f) => Ok(self * f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __rmul__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d * self),
            DualsOrF64::F64(f) => Ok(f * self),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __truediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(self / d),
            DualsOrF64::F64(f) => Ok(self / f),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
        }
    }

    fn __rtruediv__(&self, other: DualsOrF64) -> PyResult<Self> {
        match other {
            DualsOrF64::Dual2(d) => Ok(d / self),
            DualsOrF64::F64(f) => Ok(f / self),
            DualsOrF64::Dual(d) => Err(PyTypeError::new_err("Dual2 operation with incompatible type (Dual)."))
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
    use crate::dual::dual1::Dual;

    #[test]
    fn clone_arc() {
        let d1 = Dual2::new(20.0, vec!["a".to_string()], Vec::new(), Vec::new());
        let d2 = d1.clone();
        assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
    }

    #[test]
    fn default_dual() {
        let result = Dual2::default();
        let expected = Dual2::new(0.0, Vec::new(), Vec::new(), Vec::new());
        assert_eq!(result, expected);
    }

    #[test]
    fn to_new_ordered_vars() {
        let d1 = Dual2::new(20.0, vec!["a".to_string()], vec![], vec![]);
        let d2 = Dual2::new(20.0, vec!["a".to_string(), "b".to_string()], vec![], vec![]);
        let d3 = d1.to_new_ordered_vars(&d2.vars);
        assert!(Arc::ptr_eq(&d3.vars, &d2.vars));
        assert!(d3.dual.len() == 2);
        let d4 = d2.to_new_ordered_vars(&d1.vars);
        assert!(Arc::ptr_eq(&d4.vars, &d1.vars));
        assert!(d4.dual.len() == 1);
    }

    #[test]
    fn new_dual() {
        Dual2::new(2.3, Vec::from([String::from("a")]), Vec::new(), Vec::new());
    }

    #[test]
    #[should_panic]
    fn new_dual_panic() {
        Dual2::new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([1.0]),
            Vec::new(),
        );
    }

    #[test]
    fn zero_init() {
        let d = Dual2::new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::new(),
            Vec::new(),
        );
        for (_, val) in d.dual.indexed_iter() {
            assert!(*val == 1.0)
        }
    }

    #[test]
    fn negate() {
        let d = Dual2::new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([2., -1.4]),
            Vec::from([1.0, -1.0, -1.0, 2.0]),
        );
        let d2 = -d.clone();
        assert!(d2.real == -2.3);
        assert!(Arc::ptr_eq(&d.vars, &d2.vars));
        assert!(d2.dual[0] == -2.0);
        assert!(d2.dual[1] == 1.4);
        assert!(d2.dual2[[1, 0]] == 1.0);
    }

    #[test]
    fn eq_ne() {
        // Dual with vars - f64
        assert!(Dual2::new(0.0, Vec::from([String::from("a")]), Vec::new(), Vec::new()) != 0.0);
        // Dual with no vars - f64 (+reverse)
        assert!(Dual2::new(2.0, Vec::new(), Vec::new(), Vec::new()) == 2.0);
        assert!(2.0 == Dual2::new(2.0, Vec::new(), Vec::new(), Vec::new()));
        // Dual - Dual (various real, vars, gradient mismatch)
        let d = Dual2::new(
            2.0,
            Vec::from([String::from("a")]),
            Vec::from([2.3]),
            Vec::new(),
        );
        assert!(
            d == Dual2::new(
                2.0,
                Vec::from([String::from("a")]),
                Vec::from([2.3]),
                Vec::new()
            )
        );
        assert!(
            d != Dual2::new(
                2.0,
                Vec::from([String::from("b")]),
                Vec::from([2.3]),
                Vec::new()
            )
        );
        assert!(
            d != Dual2::new(
                3.0,
                Vec::from([String::from("a")]),
                Vec::from([2.3]),
                Vec::new()
            )
        );
        assert!(
            d != Dual2::new(
                2.0,
                Vec::from([String::from("a")]),
                Vec::from([1.3]),
                Vec::new()
            )
        );
        // Dual - Dual (missing Vars are zero and upcasted)
        assert!(
            d == Dual2::new(
                2.0,
                Vec::from([String::from("a"), String::from("b")]),
                Vec::from([2.3, 0.0]),
                Vec::new()
            )
        );
    }

    #[test]
    fn add_f64() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = 10.0 + d1 + 15.0;
        let expected = Dual2::new(
            26.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn add() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let d2 = Dual2::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        );
        let expected = Dual2::new(
            3.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, 3.0],
            Vec::new(),
        );
        let result = d1 + d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn sub_f64() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = (10.0 - d1) - 15.0;
        let expected = Dual2::new(
            -6.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
            Vec::new(),
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn sub() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let d2 = Dual2::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        );
        let expected = Dual2::new(
            -1.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![1.0, 2.0, -3.0],
            Vec::new(),
        );
        let result = d1 - d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn mul_f64() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = 10.0 * d1 * 2.0;
        let expected = Dual2::new(
            20.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![20.0, 40.0],
            Vec::new(),
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn mul() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let d2 = Dual2::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        );
        let expected = Dual2::new(
            2.0,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![2.0, 4.0, 3.0],
            vec![
                0., 0., 1.5,
                0., 0., 3.,
                1.5, 3., 0.,
            ],
        );
        let result = d1 * d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn inv() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = d1.clone() * d1.pow(-1.0);
        let expected = Dual2::new(1.0, vec![], vec![], Vec::new());
        assert_eq!(result, expected)
    }

    #[test]
    fn abs() {
        let d1 = Dual2::new(
            -2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = d1.abs();
        let expected = Dual2::new(
            2.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![-1.0, -2.0],
            Vec::new(),
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn div_f64() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = d1 / 2.0;
        let expected = Dual2::new(
            0.5,
            vec!["v0".to_string(), "v1".to_string()],
            vec![0.5, 1.0],
            Vec::new(),
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn f64_div() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = 2.0 / d1.clone();
        let expected = Dual2::new(2.0, vec![], vec![], Vec::new()) / d1;
        assert_eq!(result, expected)
    }

    #[test]
    fn div() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let d2 = Dual2::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![0.0, 3.0],
            Vec::new(),
        );
        let expected = Dual2::new(
            0.5,
            vec!["v0".to_string(), "v1".to_string(), "v2".to_string()],
            vec![0.5, 1.0, -0.75],
            vec![
                0., 0., -0.375,
                0., 0., -0.75,
                -0.375, -0.75, 1.125,
            ],
        );
        let result = d1 / d2;
        assert_eq!(result, expected)
    }

    #[test]
    fn ord() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        assert!(d1 < 2.0);
        assert!(d1 > 0.5);
        assert!(d1 <= 1.0);
        assert!(d1 >= 1.0);
        assert!(1.0 <= d1);
        assert!(1.0 >= d1);
        assert!(2.0 > d1);
        assert!(0.5 < d1);
        let d2 = Dual2::new(
            2.0,
            vec!["v0".to_string(), "v2".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        assert!(d2 > d1);
        assert!(d1 < d2);
        let d3 = Dual2::new(1.0, vec!["v3".to_string()], vec![10.0], Vec::new());
        assert!(d1 >= d3);
        assert!(d1 <= d3);
    }

    #[test]
    fn exp() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = d1.exp();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.exp();
        let expected = Dual2::new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0 * c, 2.0 * c],
            vec![1.0_f64.exp() * 0.5, 1.0_f64.exp(), 1.0_f64.exp(), 1.0_f64.exp()*2.],
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn log() {
        let d1 = Dual2::new(
            1.0,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            Vec::new(),
        );
        let result = d1.log();
        assert!(Arc::ptr_eq(&d1.vars, &result.vars));
        let c = 1.0_f64.ln();
        let expected = Dual2::new(
            c,
            vec!["v0".to_string(), "v1".to_string()],
            vec![1.0, 2.0],
            vec![-0.5, -1.0, -1.0, -2.0],
        );
        println!("{:?}", result.dual2);
        assert_eq!(result, expected);
    }

    // #[test]
    // #[should_panic]
    // fn no_dual_cross(){
    //     let a = Dual::new(2.0, Vec::new(), Vec::new());
    //     let b = Dual2::new(3.0, Vec::new(), Vec::new(), Vec::new());
    //     a + b
    // }
}
