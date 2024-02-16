use crate::dual::dual1::DualOrF64;
use ndarray::{Array, Array2, Array1, Zip, Axis, s, ArrayView1};
use num_traits::Signed;
use std::cmp::PartialOrd;

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

// enum Pivoting {
//     OnCopy,
//     OnUnderlying,
// }
//
// fn argabsmax<T: Signed + PartialOrd>(a: ArrayView1<T>) -> usize {
//     let a: (&T, usize) = a.iter().zip(0..).max_by(
//         |x,y| x.0.abs().partial_cmp(&y.0.abs()).unwrap()
//     ).unwrap();
//     a.1
// }
//
// pub fn pivot_matrix(A: &Array2<i32>) -> (Array2<i32>, Array2<i32>) {
//     // pivot square matrix
//     let n = A.len_of(Axis(0));
//     let mut P: Array2<i32> = Array::eye(n);
//     let mut Pa = A.to_owned();  // initialise PA and Original (or)
//     // let Or = A.to_owned();
//     for j in 0..n {
//         let k = argabsmax(Pa.slice(s![j.., j])) + j;
//         if j != k {
//             // define row swaps j <-> k  (note that k > j by definition)
//             let (mut Pt, mut Pb) = P.slice_mut(s![.., ..]).split_at(Axis(0), k);
//             let (r1, r2) = (Pt.row_mut(j), Pb.row_mut(0));
//             Zip::from(r1).and(r2).apply(std::mem::swap);
//
//             let (mut Pt, mut Pb) = Pa.slice_mut(s![.., ..]).split_at(Axis(0), k);
//             let (r1, r2) = (Pt.row_mut(j), Pb.row_mut(0));
//             Zip::from(r1).and(r2).apply(std::mem::swap);
//         }
//     }
//     (P, Pa)
// }