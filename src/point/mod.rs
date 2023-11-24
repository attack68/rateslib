use pyo3::exceptions::PyIndexError;
use pyo3::types::PyFloat;
use pyo3::prelude::*;

use std::sync::Arc;
use indexmap::set::IndexSet;
use ndarray::{Array1, Array};
use auto_ops::{impl_op, impl_op_commutative, impl_op_ex};


// POINT DEMO

struct Point {
    x: f64,
    y: f64,
}

#[pyclass]
pub struct PointVec {
   vec: Vec<Point>,
   label: Vec<String>,
}

#[pymethods]
impl PointVec {
    #[new]
    fn new(points: Vec<(f64, f64)>, labels: Vec<String>) -> Self {
        Self {
            vec: points.into_iter().map(|(x, y)| Point{x, y}).collect(),
            label: labels,
        }
    }

    fn __getitem__(&self, index: usize) -> PyResult<(f64, f64)> {
        if self.vec.len() <= index {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        return Ok((self.vec[index].x, self.vec[index].y))
    }

    fn __setitem__(&mut self, index: usize, t: (f64, f64)) -> PyResult<()> {
        let (x, y) = t;
        if self.vec.len() <= index {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        self.vec[index] = Point {x, y};
        return Ok(());
    }

    fn __repr__(&self) -> String {
        return format!(
            "PointVec[{}{}]",
            self.vec
                .iter()
                .map(|t| format!("({}, {})", t.x, t.y))
                .collect::<Vec<String>>()
                .join(", "),
            self.label.join(", "),
        );
    }
}

// POINT DEMO END

#[pyclass]
#[derive(Clone, Debug)]
pub struct Dual {
    real : f64,
    vars : Arc<IndexSet<String>>,
    dual : Array1<f64>,
}

#[pymethods]
impl Dual {

    #[new]
    fn new(real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
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

    fn __repr__(&self) -> String {
        return format!("<Dual: {}, [...], [...]>", self.real);
    }

    fn vars_check(&self, other: &Dual) -> String {
        if Arc::ptr_eq(&self.vars, &other.vars) {
            "SAME".to_string()
        } else {
            "DIFFERENT ARC".to_string()
        }
    }

    fn add_one(&self) -> Dual {
        Dual {real: self.real + 1.0, dual: self.dual.clone(), vars: self.vars.clone()}
    }

    fn is_same_vars(&self, other: &Dual) -> bool {
        // test if the vars of a Dual have the same elements but possibly a different order
        return self.vars.len() == other.vars.len() && self.vars.intersection(&other.vars).count() == self.vars.len()
    }

}


impl Dual {



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


}