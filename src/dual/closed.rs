use indexmap::set::IndexSet;
use std::sync::Arc;
use ndarray::Array1;

#[derive(Clone, Default)]
pub struct Dual {
    real: f64,
    vars: Arc<IndexSet<String>>,
    dual: Array1<f64>,
}

impl Dual {
    pub fn new(real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        let dual_ = if dual.is_empty() {Array1::ones(unique_vars_.len())} else {Array1::from_vec(dual)};
        assert!(unique_vars_.len() == dual_.len());
        Self {real, vars: unique_vars_, dual: dual_}
    }

    pub fn ptr_eq(&self, other: &Dual) -> bool {
        Arc::ptr_eq(&self.vars, &other.vars)
    }

    pub fn to_new_vars(&self, arc_vars: &Arc<IndexSet<String>>) -> Self {
        // check if vars are the same (but not same ARC) and construct directly
        let dual_: Array1<f64>;
        if self.vars.len() == arc_vars.len()
            && self.vars.iter().zip(arc_vars.iter()).all(|(a, b)| a == b)
        {
            dual_ = self.dual.clone();
        } else {
            let lookup_or_zero = |v| {
                match self.vars.get_index_of(v) {
                    Some(idx) => self.dual[idx],
                    None => 0.0_f64,
                }
            };
            dual_ = Array1::from_vec(arc_vars.iter().map(lookup_or_zero).collect());
        }
        Self {real: self.real, vars: Arc::clone(arc_vars), dual: dual_}
    }

    pub fn new_from(other: &Self, real: f64, vars: Vec<String>, dual: Vec<f64>) -> Self {
        let new = Self::new(real, vars, dual);
        new.to_new_vars(&other.vars)
    }
}


// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

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
        let z = x.to_new_vars(&y.vars);
        assert_eq!(z.real, 1.5_f64);
        assert!(y.ptr_eq(&z));
        assert_eq!(z.dual, Array1::from_vec(vec![1.0, 0.0]));
    }

    #[test]
    fn new_from() {
        let x = Dual::new(1.5, vec!["a".to_string(), "b".to_string()], vec![1., 2.]);
        let y = Dual::new_from(&x, 2.0, vec!["a".to_string(), "c".to_string()], vec![3., 3.]);
        assert_eq!(y.real, 2.0_f64);
        assert!(y.ptr_eq(&x));
        assert_eq!(y.dual, Array1::from_vec(vec![3.0, 0.0]));
    }

}


