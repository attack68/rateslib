use crate::dual::dual1::DualOrF64;
use ndarray::{Array, Array2, Array1, Zip, Axis, s, ArrayView1};

// pub fn dual_tensordot(a: &Array<Duals>, b:&Array<Duals>) {
//     let a_shape = a.shape();
//     let b_shape = b.shape();
//     let i: u16; let j: u16;
//     (i, j) = (a_shape[a_shape.len()-1], b_shape[0]);
//     let mut sum;
//     for i in 0..(a_shape[a_shape.len()-1) {
//         for j in 0..b_shape[0] {
//             let sum = 0;
//
//             sum = sum + a[]
//         }
//     }
// }

enum Pivoting {
    OnCopy,
    OnUnderlying,
}

fn argabsmax(a: ArrayView1<DualOrF64>) -> usize {
    let a: (usize, DualOrF64) = a.iter().enumerate().fold((0, DualOrF64::F64(0.0)), |acc, (i, elem)| {
        if elem.abs() > acc.1 { (i, elem.clone()) } else { acc }
    });
    a.0
}

pub fn pivot_matrix(A: &Array2<DualOrF64>) -> (Array2<i32>, Array2<DualOrF64>) {
    // pivot square matrix
    let n = A.len_of(Axis(0));
    let P: Array2<i32> = Array::eye(n);
    let PA = A.clone();
    let O = A.clone();
    for j in 0..n {
        let i = argabsmax(O.slice(s![j.., j]));
        if j != i {
            // define row swaps i <-> j
            let row_i = P.slice_mut(s![i, ..]);
            let row_j = P.slice_mut(s![j, ..]);
            Zip::from(row_i).and(row_j).apply(std::mem::swap);
        }
    }
    (P, PA)
}