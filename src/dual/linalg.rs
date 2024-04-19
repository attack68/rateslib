//! Perform linear algebra operations on arrays containing generic data types.

use ndarray::prelude::*;
use ndarray::Zip;
use num_traits::identities::{Zero};
use num_traits::{Signed};
use std::cmp::PartialOrd;
use std::iter::Sum;
use std::ops::{Div, Mul, Sub};
use itertools::Itertools;

// Tensor ops

/// Inner product between two 1d-arrays.
pub fn dmul11_<T>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> T
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix multiplication between a 2d-array and a 1d-array.
pub fn dmul21_<T>(a: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len_of(Axis(1)), b.len_of(Axis(0)));
    Array1::from_vec(a.axis_iter(Axis(0)).map(|row| dmul11_(&row, b)).collect())
}

/// Matrix multiplication between two 2d-arrays.
pub fn dmul22_<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len_of(Axis(1)), b.len_of(Axis(0)));
    Array1::<T>::from_vec(
        a.axis_iter(Axis(0))
         .cartesian_product(b.axis_iter(Axis(1)))
         .map(|(row, col)| dmul11_(&row, &col)).collect())
         .into_shape((a.len_of(Axis(0)), b.len_of(Axis(1))))
         .expect("Dim are pre-checked")
}



// Linalg solver

pub(crate) fn argabsmax<T>(a: ArrayView1<T>) -> usize
where
    T: Signed + PartialOrd,
{
    let vi: (&T, usize) = a
        .iter()
        .zip(0..)
        .max_by(|x, y| x.0.abs().partial_cmp(&y.0.abs()).unwrap())
        .unwrap();
    vi.1
}

// pub(crate) fn argabsmax2<T>(a: ArrayView2<T>) -> (usize, usize)
// where
//     T: Signed + PartialOrd,
// {
//     let vi: (&T, usize) = a
//         .iter()
//         .zip(0..)
//         .max_by(|x, y| x.0.abs().partial_cmp(&y.0.abs()).unwrap())
//         .unwrap();
//     let n = a.len_of(Axis(0));
//     (vi.1 / n, vi.1 % n)
// }

pub(crate) fn row_swap<T>(p: &mut Array2<T>, j: &usize, kr: &usize)
{
    let (mut pt, mut pb) = p.slice_mut(s![.., ..]).split_at(Axis(0), *kr);
    let (r1, r2) = (pt.row_mut(*j), pb.row_mut(0));
    Zip::from(r1).and(r2).for_each(std::mem::swap);
}

// pub(crate) fn col_swap<T>(p: &mut Array2<T>, j: &usize, kc: &usize)
// {
//     let (mut pl, mut pr) = p.slice_mut(s![.., ..]).split_at(Axis(1), *kc);
//     let (c1, c2) = (pl.column_mut(*j), pr.column_mut(0));
//     Zip::from(c1).and(c2).for_each(std::mem::swap);
// }

pub(crate) fn el_swap<T>(p: &mut Array1<T>, j: &usize, k: &usize)
{
    let (mut pl, mut pr) = p.slice_mut(s![..]).split_at(Axis(0), *k);
    std::mem::swap(&mut pl[*j], &mut pr[0]);
}

// fn partial_pivot_matrix<T>(a: &ArrayView2<T>) -> (Array2<f64>, Array2<f64>, Array2<T>)
// where
//     T: Signed + Num + PartialOrd + Clone,
// {
//     // pivot square matrix
//     let n = a.len_of(Axis(0));
//     let mut p: Array2<f64> = Array::eye(n);
//     let q: Array2<f64> = Array::eye(n);
//     let mut pa = a.to_owned();
//     for j in 0..n {
//         let k = argabsmax(pa.slice(s![j.., j])) + j;
//         if j != k {
//             // define row swaps j <-> k  (note that k > j by definition)
//             let (mut pt, mut pb) = p.slice_mut(s![.., ..]).split_at(Axis(0), k);
//             let (r1, r2) = (pt.row_mut(j), pb.row_mut(0));
//             Zip::from(r1).and(r2).for_each(std::mem::swap);
//
//             let (mut pt, mut pb) = pa.slice_mut(s![.., ..]).split_at(Axis(0), k);
//             let (r1, r2) = (pt.row_mut(j), pb.row_mut(0));
//             Zip::from(r1).and(r2).for_each(std::mem::swap);
//         }
//     }
//     (p, q, pa)
// }
//
// fn complete_pivot_matrix<T>(a: &ArrayView2<T>) -> (Array2<f64>, Array2<f64>, Array2<T>)
// where
//     T: Signed + Num + PartialOrd + Clone,
// {
//     // pivot square matrix
//     let n = a.len_of(Axis(0));
//     let mut p: Array2<f64> = Array::eye(n);
//     let mut q: Array2<f64> = Array::eye(n);
//     let mut at = a.to_owned();
//
//     for j in 0..n {
//         // iterate diagonally through
//         let (mut kr, mut kc) = argabsmax2(at.slice(s![j.., j..]));
//         kr += j;
//         kc += j; // align with out scope array indices
//
//         match (kr, kc) {
//             (kr, kc) if kr > j && kc > j => {
//                 row_swap(&mut p, &j, &kr);
//                 row_swap(&mut at, &j, &kr);
//                 col_swap(&mut q, &j, &kc);
//                 col_swap(&mut at, &j, &kc);
//             }
//             (kr, kc) if kr > j && kc == j => {
//                 row_swap(&mut p, &j, &kr);
//                 row_swap(&mut at, &j, &kr);
//             }
//             (kr, kc) if kr == j && kc > j => {
//                 col_swap(&mut q, &j, &kc);
//                 col_swap(&mut at, &j, &kc);
//             }
//             _ => {}
//         }
//     }
//     (p, q, at)
// }
//
// fn rook_pivot_matrix<T>(a: &ArrayView2<T>) -> (Array2<f64>, Array2<f64>, Array2<T>)
// where
//     T: Signed + Num + PartialOrd + Clone,
// {
//     // Implement a modified Rook Pivot.
//     // If Original is the largest Abs in the row, and it is greater than some
//     // tolerance then use that. This prevents row swapping where the rightmost columns
//     // are zero, which ultimately leads to failure in sparse matrices.
//
//     // pivot square matrix
//     let n = a.len_of(Axis(0));
//     let mut p: Array2<f64> = Array::eye(n);
//     let mut q: Array2<f64> = Array::eye(n);
//     let mut at = a.to_owned();
//
//     for j in 0..n {
//         // iterate diagonally through
//         let kr = argabsmax(at.slice(s![j.., j])) + j;
//         let kc = argabsmax(at.slice(s![j, j..])) + j;
//
//         match (kr, kc) {
//             (kr, kc) if kr > j && kc > j => {
//                 if at[[kr, j]].abs() > at[[j, kc]].abs() {
//                     row_swap(&mut p, &j, &kr);
//                     row_swap(&mut at, &j, &kr);
//                 } else {
//                     col_swap(&mut q, &j, &kc);
//                     col_swap(&mut at, &j, &kc);
//                 }
//             }
//             (kr, kc) if kr > j && kc == j => {
//                 // MODIFIER as explained:
//                 // if !(at[[j, j]].abs() > 1e-8) {
//                     row_swap(&mut p, &j, &kr);
//                     row_swap(&mut at, &j, &kr);
//                 // }
//             }
//             (kr, kc) if kr == j && kc > j => {
//                 col_swap(&mut q, &j, &kc);
//                 col_swap(&mut at, &j, &kc);
//             }
//             _ => {}
//         }
//     }
//     (p, q, at)
// }
//
// pub enum PivotMethod {
//     Partial,
//     Complete,
//     Rook,
// }

// pub fn pluq_decomp<T>(
//     a: &ArrayView2<T>,
//     pivot: PivotMethod,
// ) -> (Array2<f64>, Array2<T>, Array2<T>, Array2<f64>)
// where
//     T: Signed + Num + PartialOrd + Clone + One + Zero + Sum + for<'a> Div<&'a T, Output = T>,
//     for<'a> &'a T: Mul<&'a T, Output = T> + Sub<T, Output = T>,
// {
//     let n: usize = a.len_of(Axis(0));
//     let mut l: Array2<T> = Array2::zeros((n, n));
//     let mut u: Array2<T> = Array2::zeros((n, n));
//     let p;
//     let q;
//     let paq;
//     match pivot {
//         PivotMethod::Partial => (p, q, paq) = partial_pivot_matrix(a),
//         PivotMethod::Complete => (p, q, paq) = complete_pivot_matrix(a),
//         PivotMethod::Rook => {
//             (p, q, paq) = rook_pivot_matrix(a);
//         }
//     }
//
//     let one = T::one();
//     for j in 0..n {
//         l[[j, j]] = one.clone(); // all diagonal entries of L are set to unity
//
//         for i in 0..j + 1 {
//             // LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
//             let sx = dmul11_(&l.slice(s![i, ..i]), &u.slice(s![..i, j]));
//             u[[i, j]] = &paq[[i, j]] - sx;
//         }
//
//         for i in j..n {
//             // LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
//             let sy = dmul11_(&l.slice(s![i, ..j]), &u.slice(s![..j, j]));
//             l[[i, j]] = (&paq[[i, j]] - sy) / &u[[j, j]];
//         }
//     }
//     (p, l, u, q)
// }

// fn dsolve_lower21_<T>(l: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
// where
//     T: Clone + Sum + Zero,
//     for<'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T> + Div<&'a T, Output = T>
// {
//     let n: usize = l.len_of(Axis(0));
//     let mut x: Array1<T> = Array::zeros(n);
//     for i in 0..n {
//         let v = &b[i] - &dmul11_(&l.slice(s![i, ..i]), &x.slice(s![..i]));
//         x[i] = &v / &l[[i, i]]
//     }
//     x
// }


fn dsolve_upper21_<T>(u: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
where
    T: Clone + Sum + Zero,
    for<'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T> + Div<&'a T, Output = T>
{
    let n: usize = u.len_of(Axis(0));
    let mut x: Array1<T> = Array::zeros(n);
    for i in (0..n).rev() {
        let v = &b[i] - &dmul11_(&u.slice(s![i, (i+1)..]), &x.slice(s![(i+1)..]));
        x[i] = &v / &u[[i, i]]
    }
    x
}

fn dsolve21_<T>(a: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
where
  T: PartialOrd + Signed + Clone + Zero + Sum,
 for <'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T> + Div<&'a T, Output = T>
{
    assert!(a.is_square());
    let n = a.len_of(Axis(0));
    assert_eq!(b.len_of(Axis(0)), n);

    // a_ and b_ will be pivoted and amended throughout the solution
    let mut a_ = a.to_owned();
    let mut b_ = b.to_owned();

    for j in 0..n {
        let k = argabsmax(a_.slice(s![j.., j])) + j;
        if j != k {
            // define row swaps j <-> k  (note that k > j by definition)
            row_swap(&mut a_, &j, &k);
            el_swap(&mut b_, &j, &k);
        }
        // perform reduction on subsequent rows below j
        for l in (j+1)..n {
            let scl = &a_[[l, j]] / &a_[[j,j]];
            a_[[l, j]] = T::zero();
            for m in (j+1)..n {
                a_[[l, m]] = &a_[[l, m]] - &(&scl * &a_[[j, m]]);
            }
            b_[l] = &b_[l] - &(&scl * &b_[j]);
        }
    }
    dsolve_upper21_(&a_.view(), &b_.view())
}


// fn dsolve_upper_1d<T>(u: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
// where
//     T: Clone + Sum + Zero + for<'a> Div<&'a T, Output = T>,
//     for<'a> &'a T: Sub<T, Output = T> + Mul<&'a T, Output = T>,
// {
//     // reverse all dimensions and solve as lower triangular
//     dsolve_lower_1d(&u.slice(s![..;-1, ..;-1]), &b.slice(s![..;-1]))
//         .slice(s![..;-1])
//         .to_owned()
// }

// fn dsolve21_<T>(a: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
// where
//     T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a T, Output = T>,
//     for<'a> &'a T: Mul<&'a f64, Output = T> + Sub<T, Output = T> + Mul<&'a T, Output = T>,
//     for<'a> &'a f64: Mul<&'a T, Output = T>,
// {
//     let (p, l, u, q) = pluq_decomp::<T>(&a.view(), PivotMethod::Complete);
//     let pb: Array1<T> = fdmul21_(&p.view(), &b.view());
//     let z: Array1<T> = dsolve_lower_1d(&l.view(), &pb.view());
//     let y: Array1<T> = dsolve_upper_1d(&u.view(), &z.view());
//     let x: Array1<T> = fdmul21_(&q.view(), &y.view());
//     x
// }

/// Solve a linear system of equations, ax = b, using Gaussian elimination and partial pivoting.
///
/// - `a` is a 2d-array.
/// - `b` is a 1d-array.
/// - `allow_lsq` can be set to `true` if the number of rows in `a` is greater than its number of columns.
pub fn dsolve<T>(a: &ArrayView2<T>, b: &ArrayView1<T>, allow_lsq: bool) -> Array1<T>
where T: PartialOrd + Signed + Clone + Sum + Zero,
 for <'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T> + Div<&'a T, Output = T>
{
    if allow_lsq {
        let a_ = dmul22_(&a.t(), a);
        let b_ = dmul21_(&a.t(), b);
        dsolve21_(&a_.view(), &b_.view())
    } else {
        dsolve21_(a, b)
    }
}

// UNIT TESTS

//

//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::dual1::{Dual, Vars};
    use std::sync::Arc;

    fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
        // used rather than equality for float numbers
        (a - b).abs() < abs_tol.unwrap_or(1e-8)
    }

    #[test]
    fn argabsmx_i32() {
        let a: Array1<i32> = arr1(&[1, 4, 2, -5, 2]);
        let result = argabsmax(a.view());
        let expected: usize = 3;
        assert_eq!(result, expected);
    }

//     #[test]
//     fn argabsmx2_i32() {
//         let a: Array2<i32> = arr2(&[[-1, 2, 100], [-5, -2000, 0], [0, 0, 0]]);
//         let result = argabsmax2(a.view());
//         let expected: (usize, usize) = (1, 1);
//         assert_eq!(result, expected);
//     }

    #[test]
    fn argabsmx_dual() {
        let a: Array1<Dual> = arr1(&[
            Dual::new(1.0, Vec::new()),
            Dual::try_new(-2.5, Vec::from(["a".to_string()]), Vec::from([2.0])).unwrap(),
        ]);
        let result = argabsmax(a.view());
        let expected: usize = 1;
        assert_eq!(result, expected);
    }

//     #[test]
//     fn lower_tri_dual() {
//         let a = arr2(&[
//             [
//                 Dual::new(1.0, Vec::new()),
//                 Dual::new(0.0, Vec::new()),
//             ],
//             [
//                 Dual::new(2.0, Vec::new()),
//                 Dual::new(1.0, Vec::new()),
//             ],
//         ]);
//         let b = arr1(&[
//             Dual::new(2.0, Vec::new()),
//             Dual::new(5.0, Vec::new()),
//         ]);
//         let x = dsolve_lower21_(&a.view(), &b.view());
//         let expected_x = arr1(&[
//             Dual::new(2.0, Vec::new()),
//             Dual::new(1.0, Vec::new()),
//         ]);
//         assert_eq!(x, expected_x);
//     }

    #[test]
    fn upper_tri_dual() {
        let a = arr2(&[
            [
                Dual::new(1.0, Vec::new()),
                Dual::new(2.0, Vec::new()),
            ],
            [
                Dual::new(0.0, Vec::new()),
                Dual::new(1.0, Vec::new()),
            ],
        ]);
        let b = arr1(&[
            Dual::new(2.0, Vec::new()),
            Dual::new(5.0, Vec::new()),
        ]);
        let x = dsolve_upper21_(&a.view(), &b.view());
        let expected_x = arr1(&[
            Dual::new(-8.0, Vec::new()),
            Dual::new(5.0, Vec::new()),
        ]);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn dsolve_dual() {
        let a: Array2<Dual> = Array2::eye(2);
        let b: Array1<Dual> = arr1(&[
            Dual::new(2.0, vec!["x".to_string()]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()]),
        ]);
        let result = dsolve(&a.view(), &b.view(), false);
        let expected = arr1(&[
            Dual::new(2.0, vec!["x".to_string()]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()]),
        ]);
        assert_eq!(result, expected);
        assert!(Arc::ptr_eq(&result[0].vars(), &result[1].vars()));
    }

    #[test]
    #[should_panic]
    fn dmul11_p() {
        dmul11_(&arr1(&[1.0, 2.0]).view(), &arr1(&[1.0]).view());
    }

    #[test]
    #[should_panic]
    fn dmul22_p() {
        dmul22_(
            &arr2(&[[1.0, 2.0], [2.0, 3.0]]).view(),
            &arr2(&[[1.0, 2.0]]).view(),
        );
    }

    #[test]
    #[should_panic]
    fn dmul21_p() {
        dmul21_(
            &arr2(&[[1.0, 2.0], [2.0, 3.0]]).view(),
            &arr1(&[1.0]).view(),
        );
    }
}
