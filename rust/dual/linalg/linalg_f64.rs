//! Perform linear algebraic operations between arrays of generic type and arrays of f64.

use crate::dual::linalg::linalg_dual::{argabsmax, dmul22_, el_swap, row_swap};
use itertools::Itertools;
use ndarray::prelude::*;
use num_traits::identities::Zero;
use num_traits::Signed;
use std::cmp::PartialOrd;
use std::iter::Sum;
use std::ops::{Mul, Sub};

/// Outer product of two 1d-arrays containing f64s.
pub fn fouter11_(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array2<f64> {
    Array1::from_vec(
        a.iter()
            .cartesian_product(b.iter())
            .map(|(x, y)| x * y)
            .collect(),
    )
    .into_shape((a.len(), b.len()))
    .expect("Pre checked dimensions")
}

// F64 Crossover

/// Inner product of two 1d-arrays.
///
/// The LHS contains f64s and the RHS is generic.
pub fn fdmul11_<T>(a: &ArrayView1<f64>, b: &ArrayView1<T>) -> T
where
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix multiplication of a 2d-array with a 1d-array.
///
/// The LHS contains f64s and the RHS is generic.
pub fn fdmul21_<T>(a: &ArrayView2<f64>, b: &ArrayView1<T>) -> Array1<T>
where
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len_of(Axis(1)), b.len_of(Axis(0)));
    Array1::from_vec(a.axis_iter(Axis(0)).map(|row| fdmul11_(&row, b)).collect())
}

/// Matrix multiplication of a 2d-array with a 1d-array.
///
/// The LHS is generic and the RHS contains f64s.
pub fn dfmul21_<T>(a: &ArrayView2<T>, b: &ArrayView1<f64>) -> Array1<T>
where
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len_of(Axis(1)), b.len_of(Axis(0)));
    Array1::from_vec(a.axis_iter(Axis(0)).map(|row| fdmul11_(b, &row)).collect())
}

/// Matrix multiplication of two 2d-arrays.
///
/// The LHS contains f64s and the RHS is generic.
pub fn fdmul22_<T>(a: &ArrayView2<f64>, b: &ArrayView2<T>) -> Array2<T>
where
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len_of(Axis(1)), b.len_of(Axis(0)));
    Array1::<T>::from_vec(
        a.axis_iter(Axis(0))
            .cartesian_product(b.axis_iter(Axis(1)))
            .map(|(row, col)| fdmul11_(&row, &col))
            .collect(),
    )
    .into_shape((a.len_of(Axis(0)), b.len_of(Axis(1))))
    .expect("Dim are pre-checked")
}

/// Matrix multiplication of two 2d-arrays.
///
/// The LHS is generic and the RHS contains f64s.
pub fn dfmul22_<T>(a: &ArrayView2<T>, b: &ArrayView2<f64>) -> Array2<T>
where
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum,
{
    assert_eq!(a.len_of(Axis(1)), b.len_of(Axis(0)));
    Array1::<T>::from_vec(
        a.axis_iter(Axis(0))
            .cartesian_product(b.axis_iter(Axis(1)))
            .map(|(row, col)| fdmul11_(&col, &row))
            .collect(),
    )
    .into_shape((a.len_of(Axis(0)), b.len_of(Axis(1))))
    .expect("Dim are pre-checked")
}

fn fdsolve_upper21_<T>(u: &ArrayView2<f64>, b: &ArrayView1<T>) -> Array1<T>
where
    T: Sum + Zero + Clone,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    let n: usize = u.len_of(Axis(0));
    let mut x: Array1<T> = Array::zeros(n);
    for i in (0..n).rev() {
        let v = &b[i] - &fdmul11_(&u.slice(s![i, (i + 1)..]), &x.slice(s![(i + 1)..]));
        x[i] = &(1.0_f64 / &u[[i, i]]) * &v
    }
    x
}

fn fdsolve21_<T>(a: &ArrayView2<f64>, b: &ArrayView1<T>) -> Array1<T>
where
    T: PartialOrd + Signed + Clone + Zero + Sum,
    for<'a> &'a f64: Mul<&'a T, Output = T> + Mul<&'a f64, Output = f64>,
    for<'a> &'a T: Sub<&'a T, Output = T>,
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
        for l in (j + 1)..n {
            let scl: f64 = a_[[l, j]] / a_[[j, j]];
            a_[[l, j]] = 0.0_f64;
            for m in (j + 1)..n {
                a_[[l, m]] -= scl * a_[[j, m]];
            }
            b_[l] = &b_[l] - &(&scl * &b_[j]);
        }
    }
    fdsolve_upper21_(&a_.view(), &b_.view())
}

/// Solve a linear system, ax = b, using Gaussian elimination and partial pivoting.
///
/// The LHS contains f64s and the RHS is generic. `allow_lsq` can be `true` is the number of
/// rows in `a` is greater than the number of columns.
pub fn fdsolve<T>(a: &ArrayView2<f64>, b: &ArrayView1<T>, allow_lsq: bool) -> Array1<T>
where
    T: PartialOrd + Signed + Clone + Zero + Sum,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    if allow_lsq {
        let a_: Array2<f64> = dmul22_(&a.t(), a);
        let b_: Array1<T> = fdmul21_(&a.t(), b);
        fdsolve21_(&a_.view(), &b_.view())
    } else {
        fdsolve21_(a, b)
    }
}

// UNIT TESTS

//

//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::dual::{Dual, Vars};
    use std::sync::Arc;

    // fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
    //     // used rather than equality for float numbers
    //     (a - b).abs() < abs_tol.unwrap_or(1e-8)
    // }

    #[test]
    fn outer_prod() {
        let a = arr1(&[1.0, 2.0]);
        let b = arr1(&[2.0, 1.0, 3.0]);
        let c = fouter11_(&a.view(), &b.view());
        let result = arr2(&[[2., 1., 3.], [4., 2., 6.]]);
        assert_eq!(result, c)
    }

    #[test]
    fn fdupper_tri_dual() {
        let a = arr2(&[[1., 2.], [0., 1.]]);
        let b = arr1(&[Dual::new(2.0, Vec::new()), Dual::new(5.0, Vec::new())]);
        let x = fdsolve_upper21_(&a.view(), &b.view());
        let expected_x = arr1(&[Dual::new(-8.0, Vec::new()), Dual::new(5.0, Vec::new())]);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn fdsolve_dual() {
        let a: Array2<f64> = Array2::eye(2);
        let b: Array1<Dual> = arr1(&[
            Dual::new(2.0, vec!["x".to_string()]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()]),
        ]);
        let result: Array1<Dual> = fdsolve(&a.view(), &b.view(), false);
        let expected = arr1(&[
            Dual::new(2.0, vec!["x".to_string()]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()]),
        ]);
        assert_eq!(result, expected);
        assert!(Arc::ptr_eq(&result[0].vars(), &result[1].vars()));
    }

    #[test]
    #[should_panic]
    fn fdmul11_p() {
        fdmul11_(&arr1(&[1.0, 2.0]).view(), &arr1(&[1.0]).view());
    }

    #[test]
    #[should_panic]
    fn fdmul22_p() {
        fdmul22_(
            &arr2(&[[1.0, 2.0], [2.0, 3.0]]).view(),
            &arr2(&[[1.0, 2.0]]).view(),
        );
    }

    #[test]
    #[should_panic]
    fn dfmul22_p() {
        dfmul22_(
            &arr2(&[[1.0, 2.0], [2.0, 3.0]]).view(),
            &arr2(&[[1.0, 2.0]]).view(),
        );
    }
}
