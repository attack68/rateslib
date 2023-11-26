use ndarray::{Array1, Array, arr1};
use std::sync::Arc;
use indexmap::set::IndexSet;

#[derive(Clone, Debug)]
pub struct Dual1 {
    pub real : f64,
    pub vars : Arc<IndexSet<String>>,
    pub dual : Array1<f64>,
}

impl Dual1 {

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

