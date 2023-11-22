use pyo3::exceptions::PyIndexError;
use pyo3::types::PyFloat;
use pyo3::prelude::*;

use std::sync::Arc;
use indexmap::set::IndexSet;
use ndarray::{Array1, Array};

struct Point {
    x: f64,
    y: f64,
}

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

    fn __repr__(&self) -> String {
        return format!("<Dual: ... , [...], [...]>");
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