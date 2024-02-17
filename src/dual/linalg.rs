use crate::dual::dual1::{Dual};
use ndarray::{Array, Array2, Array1, Zip, Axis, s, ArrayView1, arr1, arr2};
use num_traits::{Signed, Num};
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


fn argabsmax<T>(a: ArrayView1<T>) -> usize
where T: Signed + PartialOrd
{
    let a: (&T, usize) = a.iter().zip(0..).max_by(
        |x,y| x.0.abs().partial_cmp(&y.0.abs()).unwrap()
    ).unwrap();
    a.1
}

enum PivotMethod {
    OnUpdate,
    OnOriginal,
}

pub fn pivot_matrix<T>(A: &Array2<T>, method: PivotMethod) -> (Array2<i32>, Array2<T>)
where T: Signed + Num + PartialOrd + Clone
{
    // pivot square matrix
    let n = A.len_of(Axis(0));
    let mut P: Array2<i32> = Array::eye(n);
    let mut Pa = A.to_owned();  // initialise PA and Original (or)
    // let Or = A.to_owned();
    for j in 0..n {
        let k;
        match &method {
            PivotMethod::OnOriginal => { k = argabsmax(A.slice(s![j.., j])) + j;},
            PivotMethod::OnUpdate => { k = argabsmax(Pa.slice(s![j.., j])) + j;}
        }
        if j != k {
            // define row swaps j <-> k  (note that k > j by definition)
            let (mut Pt, mut Pb) = P.slice_mut(s![.., ..]).split_at(Axis(0), k);
            let (r1, r2) = (Pt.row_mut(j), Pb.row_mut(0));
            Zip::from(r1).and(r2).for_each(std::mem::swap);

            let (mut Pt, mut Pb) = Pa.slice_mut(s![.., ..]).split_at(Axis(0), k);
            let (r1, r2) = (Pt.row_mut(j), Pb.row_mut(0));
            Zip::from(r1).and(r2).for_each(std::mem::swap);
        }
    }
    (P, Pa)
}


// UNIT TESTS


#[test]
fn argabsmx_i32() {
    let A: Array1<i32> = arr1(&[1, 4, 2, -5, 2]);
    let result = argabsmax(A.view());
    let expected: usize = 3;
    assert_eq!(result, expected);
}

#[test]
fn argabsmx_dual() {
    let A: Array1<Dual> = arr1(
        &[Dual::new(1.0, Vec::new(), Vec::new()),
              Dual::new(-2.5, Vec::from(["a".to_string()]), Vec::from([2.0]))]
    );
    let result = argabsmax(A.view());
    let expected: usize = 1;
    assert_eq!(result, expected);
}

#[test]
fn pivot_i32_update() {
    let P: Array2<i32> = arr2(
        &[[1, 2, 3, 4],
          [10, 2, 5, 6],
          [7, 8, 1, 1],
          [2, 2, 2, 9]]
    );
    let (result0, result1) = pivot_matrix(&P, PivotMethod::OnUpdate);
    let expected0: Array2<i32> = arr2(
        &[[0, 1, 0, 0],
          [0, 0, 1, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1]]
    );
    let expected1: Array2<i32> = arr2(
        &[[10, 2, 5, 6],
          [7, 8, 1, 1],
          [1, 2, 3, 4],
          [2, 2, 2, 9]]
    );
    assert_eq!(result0, expected0);
    assert_eq!(result1, expected1);
}

#[test]
fn pivot_i32_original() {
    let P: Array2<i32> = arr2(
        &[[1, 2, 3, 4],
          [10, 2, 5, 6],
          [7, 8, 1, 1],
          [2, 2, 2, 9]]
    );
    let (result0, result1) = pivot_matrix(&P, PivotMethod::OnOriginal);
    let expected0: Array2<i32> = arr2(
        &[[0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
          [1, 0, 0, 0]]
    );
    let expected1: Array2<i32> = arr2(
        &[[10, 2, 5, 6],
          [7, 8, 1, 1],
          [2, 2, 2, 9],
          [1, 2, 3, 4]]
    );
    assert_eq!(result0, expected0);
    assert_eq!(result1, expected1);
}