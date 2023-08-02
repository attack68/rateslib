use crate::dual::{Dual, Duals};
use ndarray::Array;

pub fn dual_tensordot(a: &Array<Duals>, b:&Array<Duals>) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let i: u16; let j: u16;
    (i, j) = (a_shape[a_shape.len()-1], b_shape[0]);
    let mut sum;
    for i in 0..(a_shape[a_shape.len()-1) {
        for j in 0..b_shape[0] {
            let sum = 0;

            sum = sum + a[]
        }
    }

}