use crate::curves::interpolation::utils::index_left;
use pyo3::pyfunction;

macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[pyfunction]
        pub fn $name(list_input: Vec<$type>, value: $type, left_count: Option<usize>) -> usize {
            index_left(&list_input[..], &value, left_count)
        }
    };
}

create_interface!(index_left_f64, f64);
