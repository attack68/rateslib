use numpy::{PyArray, ToPyArray, PyArray1};
use ndarray::{Array1, Array, arr1};
use num_traits;
use num_traits::Pow;
use std::sync::Arc;
use indexmap::set::IndexSet;
use auto_ops::{impl_op, impl_op_commutative, impl_op_ex, impl_op_ex_commutative};

use pyo3::exceptions::PyIndexError;
use pyo3::types::PyFloat;
use pyo3::prelude::*;
use pyo3::conversion::FromPyObject;

fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
    // used rather than equality for float numbers
    return (a-b).abs() < abs_tol.unwrap_or(1e-8)
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Dual {
    pub real : f64,
    pub vars : Arc<IndexSet<String>>,
    pub dual : Array1<f64>,
}

#[derive(Debug, Clone, PartialEq, FromPyObject)]
pub enum DualOrF64 {
    Dual(Dual),
    F64(f64),
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

    fn gradient<'py>(&'py self, py: Python<'py>, vars: Vec<String>) -> PyResult<&PyArray1<f64>> {
        Ok(self.ggradient(vars).to_pyarray(py))
    }

    #[getter]
    fn dual<'py>(&'py self, py: Python<'py>) -> PyResult<&PyArray1<f64>> {
        Ok(self.dual.to_pyarray(py))
    }

    fn arc_check(&self, other: &Dual) -> PyResult<bool> {
        Ok(Arc::ptr_eq(&self.vars, &other.vars))
    }

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
        if self.vars.len() >= other.vars.len() && other.vars.iter().all(|var| self.vars.contains(var)) {
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
    pub fn to_new_ordered_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        if self.vars.iter().zip(new_vars.iter()).all(|(a,b)| a==b) {
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

    fn is_same_vars(&self, other: &Dual) -> bool {
        // test if the vars of a Dual have the same elements but possibly a different order
        return self.vars.len() == other.vars.len() && self.vars.intersection(&other.vars).count() == self.vars.len()
    }

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
}

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

impl std::ops::AddAssign for Dual {
    fn add_assign(&mut self, other: Self) {
        let z = self.clone() + other;
        self.vars = z.vars.clone();
        self.dual = z.dual.clone();
        self.real = z.real;
    }
}

impl std::ops::MulAssign for Dual {
    fn mul_assign(&mut self, other: Self) {
        let z = self.clone() * other;
        self.vars = z.vars.clone();
        self.dual = z.dual.clone();
        self.real = z.real;
    }
}

impl_op!(- |a: Dual| -> Dual { Dual {vars: a.vars, real: -a.real, dual: -a.dual}});
impl_op!(- |a: &Dual| -> Dual { Dual {vars: a.vars.clone(), real: -a.real, dual: -(a.dual.clone())}});

// impl_op_commutative!(+ |a: Dual, b: f64| -> Dual { Dual {vars: a.vars, real: a.real + b, dual: a.dual} });
// impl_op_commutative!(+ |a: Dual, b: &f64| -> Dual { Dual {vars: a.vars, real: a.real + b, dual: a.dual} });
// impl_op_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
// impl_op_commutative!(+ |a: &Dual, b: f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
impl_op_ex_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real + b.real, dual: &a.dual + &b.dual, vars: Arc::clone(&a.vars)}
    }
    else {
        let (x, y) = a.to_combined_vars(b);
        x + y
    }
});

impl_op_ex!(- |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
impl_op_ex!(- |a: &f64, b: &Dual| -> Dual { Dual {vars: Arc::clone(&b.vars), real: a - b.real, dual: -(b.dual.clone())} });
impl_op_ex!(- |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real - b.real, dual: &a.dual - &b.dual, vars: a.vars.clone()}
    }
    else {
        let (x, y) = a.to_combined_vars(b);
        x - y
    }
});

impl_op_ex!(* |a: &Dual, b: f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real * b, dual: b * &a.dual} });
impl_op_ex!(* |a: f64, b: &Dual| -> Dual { Dual {vars: Arc::clone(&b.vars), real: a * b.real, dual: a * &b.dual} });
impl_op_ex!(* |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real * b.real, dual: &a.dual * b.real + &b.dual * a.real, vars: a.vars.clone()}
    }
    else {
        let (x, y) = a.to_combined_vars(b);
        x * y
    }
});

impl_op!(/ |a: Dual, b: f64| -> Dual { Dual {vars: a.vars, real: a.real / b, dual: a.dual / b} });
impl_op!(/ |a: f64, b: Dual| -> Dual { a * b.pow(-1.0) });
impl_op_ex!(/ |a: &Dual, b: &Dual| -> Dual { a * b.clone().pow(-1.0) });

impl PartialEq<f64> for Dual {
    fn eq(&self, other: &f64) -> bool {
        return Dual::new(*other, [].to_vec(), [].to_vec()) == *self;
    }
}

impl std::iter::Sum for Dual {
    fn sum<I>(iter: I) -> Self
    where I: Iterator<Item = Dual> {
        return iter.fold(Dual::new(0.0, [].to_vec(), [].to_vec()), |acc, x| acc + x)
    }
}

impl PartialEq<Dual> for Dual {
    fn eq(&self, other: &Dual) -> bool {
        if self.real != other.real {
            return false
        }
        if ! self.is_same_vars(&other) {
            return false
        }
        let another = other.to_new_vars(&self.vars);
        for (i, elem) in self.dual.iter().enumerate() {
            if ! is_close(&another.dual[[i]], &elem, None) {
                return false
            }
        }
        return true
    }
}

pub fn arr1_dot(a1: Array1<Dual>, a2: Array1<Dual>) -> Dual {
    // Consumes two one dimensional arrays and produces a scalar value of their dot product.
    let z = a1.into_iter().zip(a2.into_iter()).map(|(x, y)| x * y).collect::<Vec<Dual>>();
    return z.into_iter().sum::<Dual>()
}

// #[derive(Clone, Debug, Add, Sub, Mul)]
// pub enum Duals {Dual(Dual), Float(f64)}

// impl_op!(+ |a: Duals, b: Duals| -> Duals {
//     match a {
//         Dual => {
//             match b {
//                 Dual => Duals::Dual(a) + Duals::Dual(b),
//                 Float=> Duals::Dual(a) + Duals::Float(b),
//             }
//         },
//         Float => {
//             match b {
//                 Dual => Duals::Float(a) + Duals::Dual(b),
//                 Float=> Duals::Float(a) + Duals::Float(b),
//             }
//         }
//     }
// });
// impl_op!(- |a: Duals, b: Duals| -> Duals { a - b });
// impl_op!(* |a: Duals, b: Duals| -> Duals { a * b });
// impl_op!(/ |a: Duals, b: Duals| -> Duals { a / b });


// impl ops::Add<f64> for Dual {
//     type Output = Dual;
//     fn add(self, other: f64) -> Dual {
//         Dual {vars: self.vars, real: self.real + other, dual: self.dual}
//     }
// }

// impl ops::Sub<f64> for Dual {
//     type Output = Dual;
//     fn sub(self, other: f64) -> Dual {
//         Dual {vars: self.vars, real: self.real - other, dual: self.dual}
//     }
// }

// impl ops::Mul<f64> for Dual {
//     type Output = Dual;
//     fn mul(self, other: f64) -> Dual {
//         Dual {vars: self.vars, real: self.real * other, dual: self.dual * other}
//     }
// }

// impl ops::Neg for Dual {
//     type Output = Dual;
//     fn neg(self) -> Dual {
//         Dual {vars: self.vars, real: -self.real, dual: -self.dual}
//     }
// }

// impl ops::Add<Dual> for Dual {
//     type Output = Dual;
//     fn add(self, other: Dual) -> Dual {
//         let (x, y) = self.to_combined_vars(other);
//         Dual {
//             real: x.real + y.real,
//             dual: x.dual + y.dual,
//             vars: x.vars,
//         }
//     }
// }

// impl ops::Sub<Dual> for Dual {
//     type Output = Dual;
//     fn sub(self, other: Dual) -> Dual {
//         let (x, y) = self.to_combined_vars(other);
//         Dual {
//             real: x.real - y.real,
//             dual: x.dual - y.dual,
//             vars: x.vars
//         }
//     }
// }

// impl ops::Mul<Dual> for Dual {
//     type Output = Dual;
//     fn mul(self, other: Dual) -> Dual {
//         let (x, y) = self.to_combined_vars(other);
//         Dual {
//             real: x.real * y.real,
//             dual: x.dual * y.real + y.dual * x.real,
//             vars: x.vars
//         }
//     }
// }
