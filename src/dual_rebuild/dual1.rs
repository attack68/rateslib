use ndarray::{Array1, Array, arr1};
use std::sync::Arc;
use indexmap::set::IndexSet;

// use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct Dual {
    pub real : f64,
    pub vars : Arc<IndexSet<String>>,
    pub dual : Array1<f64>,
}

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
    /// use crate::dual_rebuild::Dual1;
    /// let f = Dual(2.5, Vec::from([String::from("x")]), Vec::new())
    /// ```
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
        /// Return two equivalent Duals with same vars.
        ///
        /// # Arguments
        ///
        /// * `other` - Alternative Dual against which vars comparison is made

        // check the set of vars in each Dual are equivalent
        // println!("check vars len");
        if self.vars.len() == other.vars.len() {
            // vars may be the same, or different, or same but not ordered similarly
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
        /// Return two equivalent Duals with the same, but recast, vars.

        // Both Duals assumed to have different vars so combine the vars and recast the Duals
        let comb_vars = Arc::new(IndexSet::from_iter(self.vars.union(&other.vars).map(|x| x.clone())));
        (self.to_new_vars(&comb_vars), other.to_new_vars(&comb_vars))
    }

    fn to_new_ordered_vars(&self, new_vars: &Arc<IndexSet<String>>) -> Dual {
        /// Return a Dual with its vars re-ordered if necessary.

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
        /// Return a Dual with a new set of vars.

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