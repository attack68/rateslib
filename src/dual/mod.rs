//! Toolset for automatic differentiation (AD).
//!

pub mod dual;
mod dual_ops;
pub mod dual_py;
pub mod linalg;
pub mod linalg_f64;
pub mod linalg_py;


/// Utility for creating an ordered list of variable tags from a string and enumerator
pub(crate) fn get_variable_tags(name: &str, range: usize) -> Vec<String>{
    Vec::from_iter((0..range).map(|i| name.to_string() + &i.to_string() ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_variable_tags() {
        let result = get_variable_tags("x", 3);
        assert_eq!(result, vec!["x0".to_string(), "x1".to_string(), "x2".to_string()])
    }
}