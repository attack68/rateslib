
pub mod dual;
use dual::dual1::Dual;
use ndarray::{Array1, Array};

fn main() {
    let dual_: Array1<f64> = Array::ones(2);
    let dual2_: Array1<f64> = Array::ones(2);
    println!("{}", dual_.iter().eq(dual2_.iter()));
}
