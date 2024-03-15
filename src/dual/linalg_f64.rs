use crate::dual::dual1::Dual;
use crate::dual::linalg::{dmul22_, pluq_decomp, PivotMethod};
use ndarray::prelude::*;
use ndarray::Zip;
use num_traits::identities::{One, Zero};
use num_traits::{Num, Signed};
use std::cmp::PartialOrd;
use std::iter::Sum;
use std::ops::{Div, Mul, Sub};
use std::sync::Arc;

pub fn outer11_(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array2<f64> {
    // TODO make this more efficient without looping
    let mut c: Array2<f64> = Array::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            c[[i, j]] = &a[i] * &b[j];
        }
    }
    c
}

// F64 Crossover

pub fn fdmul11_<T>(a: &ArrayView1<f64>, b: &ArrayView1<T>) -> T
where
    for<'a> &'a T: Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum,
{
    if a.len() != b.len() {
        panic!("Lengths of LHS and RHS do not match.")
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn fdmul22_<T>(a: &ArrayView2<f64>, b: &ArrayView2<T>) -> Array2<T>
where
    for<'a> &'a T: Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum + Zero + Clone,
{
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out: Array2<T> = Array2::zeros((a.len_of(Axis(0)), b.len_of(Axis(1))));
    for r in 0..a.len_of(Axis(0)) {
        for c in 0..b.len_of(Axis(1)) {
            out[[r, c]] = fdmul11_(&a.row(r), &b.column(c))
        }
    }
    out
}

pub fn dfmul22_<T>(a: &ArrayView2<T>, b: &ArrayView2<f64>) -> Array2<T>
where
    for<'a> &'a T: Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum + Zero + Clone,
{
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out: Array2<T> = Array2::zeros((a.len_of(Axis(0)), b.len_of(Axis(1))));
    for r in 0..a.len_of(Axis(0)) {
        for c in 0..b.len_of(Axis(1)) {
            out[[r, c]] = fdmul11_(&b.column(c), &a.row(r))
        }
    }
    out
}

pub fn fdmul21_<T>(a: &ArrayView2<f64>, b: &ArrayView1<T>) -> Array1<T>
where
    for<'a> &'a T: Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
    T: Sum + Zero + Clone,
{
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out = Array1::zeros(b.len_of(Axis(0)));
    for r in 0..a.len_of(Axis(0)) {
        out[[r]] = fdmul11_(&a.row(r), b)
    }
    out
}

fn fdsolve_lower_1d<T>(l: &ArrayView2<f64>, b: &ArrayView1<T>) -> Array1<T>
where
    T: Clone + Sum + Zero + for<'a> Div<&'a f64, Output = T>,
    for<'a> &'a T: Sub<T, Output = T> + Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
{
    let n: usize = l.len_of(Axis(0));
    let mut x: Array1<T> = Array::zeros(n);
    for i in 0..n {
        let s = fdmul11_(&l.slice(s![i, ..i]), &x.slice(s![..i]));
        let v = &b[i] - s;
        x[i] = v / &l[[i, i]]
    }
    x
}

fn fdsolve_upper_1d<T>(u: &ArrayView2<f64>, b: &ArrayView1<T>) -> Array1<T>
where
    T: Clone + Sum + Zero + for<'a> Div<&'a f64, Output = T>,
    for<'a> &'a T: Sub<T, Output = T> + Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
{
    // reverse all dimensions and solve as lower triangular
    fdsolve_lower_1d(&u.slice(s![..;-1, ..;-1]), &b.slice(s![..;-1]))
        .slice(s![..;-1])
        .to_owned()
}

fn fdsolve21_<T>(a: &ArrayView2<f64>, b: &ArrayView1<T>) -> Array1<T>
where
    T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a f64, Output = T>,
    for<'a> &'a T: Mul<&'a f64, Output = T> + Sub<T, Output = T> + Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
{
    let (p, l, u, q) = pluq_decomp::<f64>(&a.view(), PivotMethod::Rook);
    let pb = fdmul21_(&p.view(), &b.view());
    let z = fdsolve_lower_1d(&l.view(), &pb.view());
    let y = fdsolve_upper_1d(&u.view(), &z.view());
    let x = fdmul21_(&q.view(), &y.view());
    x
}

pub fn fdsolve<T>(a: &ArrayView2<f64>, b: &ArrayView1<T>, allow_lsq: bool) -> Array1<T>
where
    T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a f64, Output = T>,
    for<'a> &'a T: Sub<T, Output = T> + Mul<&'a f64, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
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
    use crate::dual::dual1::Gradient1;

    fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
        // used rather than equality for float numbers
        (a - b).abs() < abs_tol.unwrap_or(1e-8)
    }

    #[test]
    fn outer_prod() {
        let a = arr1(&[1.0, 2.0]);
        let b = arr1(&[2.0, 1.0, 3.0]);
        let mut c = outer11_(&a.view(), &b.view());
        let result = arr2(&[[2., 1., 3.], [4., 2., 6.]]);
        assert_eq!(result, c)
    }

    #[test]
    fn lower_tri_dual() {
        let a = arr2(&[[1., 0.], [2., 1.]]);
        let b = arr1(&[
            Dual::new(2.0, Vec::new(), Vec::new()),
            Dual::new(5.0, Vec::new(), Vec::new()),
        ]);
        let x = fdsolve_lower_1d(&a.view(), &b.view());
        let expected_x = arr1(&[
            Dual::new(2.0, Vec::new(), Vec::new()),
            Dual::new(1.0, Vec::new(), Vec::new()),
        ]);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn fdupper_tri_dual() {
        let a = arr2(&[[1., 2.], [0., 1.]]);
        let b = arr1(&[
            Dual::new(2.0, Vec::new(), Vec::new()),
            Dual::new(5.0, Vec::new(), Vec::new()),
        ]);
        let x = fdsolve_upper_1d(&a.view(), &b.view());
        let expected_x = arr1(&[
            Dual::new(-8.0, Vec::new(), Vec::new()),
            Dual::new(5.0, Vec::new(), Vec::new()),
        ]);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn fdsolve_dual() {
        let a: Array2<f64> = Array2::eye(2);
        let b: Array1<Dual> = arr1(&[
            Dual::new(2.0, vec!["x".to_string()], vec![1.0]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 1.0]),
        ]);
        let result: Array1<Dual> = fdsolve(&a.view(), &b.view(), false);
        let expected = arr1(&[
            Dual::new(2.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 1.0]),
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
