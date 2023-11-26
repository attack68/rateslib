// use ndarray_einsum_beta::*;
// use indexmap::indexset;
// use std::collections::HashSet;

use ndarray::{Array1, Array, arr1};
use num_traits;
use num_traits::Pow;
use std::sync::Arc;
use indexmap::set::IndexSet;
use auto_ops::{impl_op, impl_op_commutative, impl_op_ex};

use pyo3::exceptions::PyIndexError;
use pyo3::types::PyFloat;
use pyo3::prelude::*;


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

#[pymethods]
impl Dual {
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
}


impl Dual {
    fn to_combined_vars(&self, other: &Dual) -> (Dual, Dual) {
        // check the set of vars in each Dual are equivalent
        // println!("check vars len");
        if self.vars.len() == other.vars.len() {
            // vars may be the same or different or same but not ordered similarly
            // println!("check vars are same");
            if self.vars.iter().all(|var| other.vars.contains(var)) {
                // vars are the same but may be ordered differently
                // println!("check vars are same order");
                (self.clone(), other.to_new_ordered_vars(&self.vars))
            } else {
                // vars are different so both must be recast
                self.to_combined_vars_explicit(other)
            }
        } else {
            // vars are definitely different
            // println!("not same number vars");
            self.to_combined_vars_explicit(other)
        }
    }

    fn to_combined_vars_explicit(&self, other: &Dual) -> (Dual, Dual) {
        // Both Duals assumed to have different vars so combine the vars and recast the Duals
        let comb_vars = Arc::new(IndexSet::from_iter(self.vars.union(&other.vars).map(|x| x.clone())));
        (self.to_new_vars(&comb_vars), other.to_new_vars(&comb_vars))
    }

    fn to_new_ordered_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        // new vars are the same as self.vars but may have a different order
        if self.vars.iter().zip(new_vars.iter()).all(|(a,b)| a==b) {
            // vars are identical
            // println!("clone vars");
            self.clone()
        } else {
            // println!("to new vars");
            self.to_new_vars(new_vars)
        }
    }

    fn to_new_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        // Take a Dual and redefine its derivatives according to a new set of variable tags.

        // let a: HashSet<_> = self.vars.iter().collect();
        // let b: HashSet<_> = new_vars.iter().collect();
        // let same_vars = a == b;
        // let same_vars: bool = self.vars == *new_vars;
        // let same_vars: bool = self.vars.iter().zip(new_vars.iter()).filter(|&(a, b)| a == b).count() == self.vars.len();

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
// impl_op!(- |a: &Dual| -> Dual { Dual {vars: a.vars.clone(), real: -a.real, dual: -a.dual.clone()}});

impl_op_commutative!(+ |a: Dual, b: f64| -> Dual { Dual {vars: a.vars, real: a.real + b, dual: a.dual} });
impl_op_commutative!(+ |a: Dual, b: &f64| -> Dual { Dual {vars: a.vars, real: a.real + b, dual: a.dual} });
impl_op_commutative!(+ |a: &Dual, b: &f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
impl_op_commutative!(+ |a: &Dual, b: f64| -> Dual { Dual {vars: Arc::clone(&a.vars), real: a.real + b, dual: a.dual.clone()} });
impl_op_ex!(+ |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real + b.real, dual: &a.dual + &b.dual, vars: Arc::clone(&a.vars)}
    }
    else {
        let (x, y) = a.to_combined_vars_explicit(b);
        x + y
    }
});

impl_op!(- |a: Dual, b: f64| -> Dual { Dual {vars: a.vars, real: a.real - b, dual: a.dual} });
impl_op!(- |a: f64, b: Dual| -> Dual { Dual {vars: b.vars, real: a - b.real, dual: -b.dual} });
impl_op_ex!(- |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real - b.real, dual: &a.dual - &b.dual, vars: a.vars.clone()}
    }
    else {
        let (x, y) = a.to_combined_vars_explicit(b);
        x - y
    }
});

impl_op!(* |a: Dual, b: f64| -> Dual { Dual {vars: a.vars, real: a.real * b, dual: a.dual * b} });
impl_op!(* |a: f64, b: Dual| -> Dual { Dual {vars: b.vars, real: a * b.real, dual: b.dual * a} });
impl_op_ex!(* |a: &Dual, b: &Dual| -> Dual {
    if Arc::ptr_eq(&a.vars, &b.vars) {
        Dual {real: a.real * b.real, dual: &a.dual * b.real + &b.dual * a.real, vars: a.vars.clone()}
    }
    else {
        let (x, y) = a.to_combined_vars_explicit(b);
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

