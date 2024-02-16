#[path = "../dual/mod.rs"]
mod dual;
use dual::dual1::{Dual, DualOrF64};
use ndarray::{Array2};


fn main() {
    let P: Array2<Dual> = Array2::eye(2);
    println!("{:?}", P);
}
