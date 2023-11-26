use ndarray::{Array1, Array, arr1};
use std::sync::Arc;
use indexmap::set::IndexSet;

// use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct Dual1 {
    pub real : f64,
    pub vars : Arc<IndexSet<String>>,
    pub dual : Array1<f64>,
}

impl Dual1 {
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

