//! Create and use data types for calculating first order derivatives using AD.
//!
//! Dual number

use indexmap::set::IndexSet;
use std::sync::Arc;
use ndarray::Array1;

/// Struct for defining a dual number data type supporting first order derivatives.
#[derive(Clone, Default)]
pub struct Dual {
    real: f64,
    vars: Arc<IndexSet<String>>,
    dual: Array1<f64>,
}

/// Enum defining the `vars` state of two `Dual` objects, a LHS relative to a RHS.
#[derive(Clone)]
pub enum VarsState {
    EquivByArc,  // Duals share an Arc ptr to their Vars
    EquivByVal,  // Duals share the same vars in the same order but no Arc ptr
    Superset,    // The Dual vars contains all of the queried values and is larger set
    Subset,      // The Dual vars is contained in the queried values and is smaller set
    Difference,  // The Dual vars and the queried set contain different values.
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
    /// let y = Dual::new(1.5, vec!["y".to_string()], vec![]).to_new_vars(x.vars());
    /// ```
    pub fn new_from(other: &Self, real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
        let new = Self::new(real, vars, dual);
        new.to_new_vars(&other.vars, None)
    }

    /// Get a reference to the Arc pointer for the `IndexSet` containing the object's variables.
    pub fn vars(&self) -> &Arc<IndexSet<String>> {
        &self.vars
    }

    /// Compare if two `Dual` objects share the same `vars`by Arc pointer equivalence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let x1 = Dual::new(1.5, vec!["x".to_string()], vec![]);
    /// let x2 = Dual::new(2.5, vec!["x".to_string()], vec![]);
    /// x1.ptr_eq(&x2); // false
    /// ```
    pub fn ptr_eq(&self, other: &Dual) -> bool {
        Arc::ptr_eq(&self.vars, &other.vars)
    }

    fn vars_cmp(&self, arc_vars: &Arc<IndexSet<String>>) -> VarsState {
        if Arc::ptr_eq(&self.vars, arc_vars) {
            VarsState::EquivByArc
        } else if self.vars.len() == arc_vars.len()
            && self.vars.iter().zip(arc_vars.iter()).all(|(a, b)| a == b) {
            VarsState::EquivByVal
        } else if self.vars.len() >= arc_vars.len()
            && arc_vars.iter().all(|var| self.vars.contains(var)) {
            VarsState::Superset
        } else if self.vars.len() < arc_vars.len()
            && self.vars.iter().all(|var| arc_vars.contains(var)) {
            VarsState::Subset
        } else {
            VarsState::Difference
        }
    }

    /// Construct a new `Dual` with `vars` set as the given Arc pointer and gradients shuffled in memory.
    ///
    /// Examples
    ///
    /// ```rust
    /// let x = Dual::new(1.5, vec!["x".to_string()], vec![]);
    /// let xy = Dual::new(2.5, vec!["x".to_string(), "y".to_string()], vec![]);
    /// let x_y = x.to_new_vars(xy.vars());
    /// // x_y: <Dual: 1.5, (x, y), [1.0, 0.0]>
    pub fn to_new_vars(&self, arc_vars: &Arc<IndexSet<String>>, state: Option<VarsState>) -> Self {
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

    pub fn to_union_vars(&self, other: &Self) -> (Self, Self) {
        let state = self.vars_cmp(&other.vars);
        match state {
            VarsState::EquivByArc => (self.clone(), other.clone()),
            VarsState::EquivByVal => (self.clone(), other.to_new_vars(&self.vars, Some(state))),
            VarsState::Superset => (self.clone(), other.to_new_vars(&self.vars, Some(VarsState::Subset))),
            VarsState::Subset => (self.to_new_vars(&other.vars, Some(state)), other.clone()),
            VarsState::Difference => self.to_combined_vars(other),
        }
    }

    fn to_combined_vars(&self, other: &Self) -> (Self, Self) {
        let comb_vars = Arc::new(IndexSet::from_iter(
            self.vars.union(&other.vars).map(|x| x.clone()),
        ));
        (self.to_new_vars(&comb_vars, Some(VarsState::Difference)),
         other.to_new_vars(&comb_vars, Some(VarsState::Difference)))
    }
}


// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

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
    fn vars_cmp_profile() {
        // Setup
        let VARS = 1000_usize;
        let x = Dual::new(
            1.5,
            (0..VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );
        let y = Dual::new(
            1.5,
            (0..VARS).map(|x| x.to_string()).collect(),
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
            (20..(VARS+20)).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
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
        let VARS = 1000_usize;
        let x = Dual::new(
            1.5,
            (0..VARS).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );
        let y = Dual::new(
            1.5,
            (0..VARS).map(|x| x.to_string()).collect(),
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
            (20..(VARS+20)).map(|x| x.to_string()).collect(),
            (0..VARS).map(|x| x as f64).collect(),
        );

        println!("\nProfiling to_union_vars (VarsState::EquivByArc):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&z);
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
                x.to_union_vars(&y);
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
                x.to_union_vars(&u);
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
                x.to_union_vars(&s);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100);
    }

}


