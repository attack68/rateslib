use crate::dual::dual1::Dual;
use crate::dual::linalg_f64::{dfmul22_, fdmul21_, fdmul22_};
use ndarray::prelude::*;
use ndarray::Zip;
use num_traits::identities::{One, Zero};
use num_traits::{Num, Signed};
use std::cmp::PartialOrd;
use std::iter::Sum;
use std::ops::{Div, Mul, Sub};
use std::sync::Arc;

// Tensor ops

pub fn dmul11_<T>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> T
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
    T: Sum,
{
    if a.len() != b.len() {
        panic!("Lengths of LHS and RHS do not match.")
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dmul22_<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
    T: Sum + Zero + Clone,
{
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out = Array2::zeros((a.len_of(Axis(0)), b.len_of(Axis(1))));
    for r in 0..a.len_of(Axis(0)) {
        for c in 0..b.len_of(Axis(1)) {
            out[[r, c]] = dmul11_(&a.row(r), &b.column(c))
        }
    }
    out
}

pub fn dmul21_<T>(a: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
    T: Sum + Zero + Clone,
{
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out = Array1::zeros(b.len_of(Axis(0)));
    for r in 0..a.len_of(Axis(0)) {
        out[[r]] = dmul11_(&a.row(r), b)
    }
    out
}

// Linalg solver

fn argabsmax<T>(a: ArrayView1<T>) -> usize
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

fn argabsmax2<T>(a: ArrayView2<T>) -> (usize, usize)
where
    T: Signed + PartialOrd,
{
    let vi: (&T, usize) = a
        .iter()
        .zip(0..)
        .max_by(|x, y| x.0.abs().partial_cmp(&y.0.abs()).unwrap())
        .unwrap();
    let n = a.len_of(Axis(0));
    (vi.1 / n, vi.1 % n)
}

fn row_swap<T>(p: &mut Array2<T>, j: &usize, kr: &usize)
where
    T: Signed + Num + PartialOrd + Clone,
{
    let (mut pt, mut pb) = p.slice_mut(s![.., ..]).split_at(Axis(0), *kr);
    let (r1, r2) = (pt.row_mut(*j), pb.row_mut(0));
    Zip::from(r1).and(r2).for_each(std::mem::swap);
}

fn col_swap<T>(p: &mut Array2<T>, j: &usize, kc: &usize)
where
    T: Signed + Num + PartialOrd + Clone,
{
    let (mut pl, mut pr) = p.slice_mut(s![.., ..]).split_at(Axis(1), *kc);
    let (c1, c2) = (pl.column_mut(*j), pr.column_mut(0));
    Zip::from(c1).and(c2).for_each(std::mem::swap);
}

fn partial_pivot_matrix<T>(a: &ArrayView2<T>) -> (Array2<f64>, Array2<f64>, Array2<T>)
where
    T: Signed + Num + PartialOrd + Clone,
{
    // pivot square matrix
    let n = a.len_of(Axis(0));
    let mut p: Array2<f64> = Array::eye(n);
    let q: Array2<f64> = Array::eye(n);
    let mut pa = a.to_owned();
    for j in 0..n {
        let k = argabsmax(pa.slice(s![j.., j])) + j;
        if j != k {
            // define row swaps j <-> k  (note that k > j by definition)
            let (mut pt, mut pb) = p.slice_mut(s![.., ..]).split_at(Axis(0), k);
            let (r1, r2) = (pt.row_mut(j), pb.row_mut(0));
            Zip::from(r1).and(r2).for_each(std::mem::swap);

            let (mut pt, mut pb) = pa.slice_mut(s![.., ..]).split_at(Axis(0), k);
            let (r1, r2) = (pt.row_mut(j), pb.row_mut(0));
            Zip::from(r1).and(r2).for_each(std::mem::swap);
        }
    }
    (p, q, pa)
}

fn complete_pivot_matrix<T>(a: &ArrayView2<T>) -> (Array2<f64>, Array2<f64>, Array2<T>)
where
    T: Signed + Num + PartialOrd + Clone,
{
    // pivot square matrix
    let n = a.len_of(Axis(0));
    let mut p: Array2<f64> = Array::eye(n);
    let mut q: Array2<f64> = Array::eye(n);
    let mut at = a.to_owned();

    for j in 0..n {
        // iterate diagonally through
        let (mut kr, mut kc) = argabsmax2(at.slice(s![j.., j..]));
        kr += j;
        kc += j; // align with out scope array indices

        match (kr, kc) {
            (kr, kc) if kr > j && kc > j => {
                row_swap(&mut p, &j, &kr);
                row_swap(&mut at, &j, &kr);
                col_swap(&mut q, &j, &kc);
                col_swap(&mut at, &j, &kc);
            }
            (kr, kc) if kr > j && kc == j => {
                row_swap(&mut p, &j, &kr);
                row_swap(&mut at, &j, &kr);
            }
            (kr, kc) if kr == j && kc > j => {
                col_swap(&mut q, &j, &kc);
                col_swap(&mut at, &j, &kc);
            }
            _ => {}
        }
    }
    (p, q, at)
}

fn rook_pivot_matrix<T>(a: &ArrayView2<T>) -> (Array2<f64>, Array2<f64>, Array2<T>)
where
    T: Signed + Num + PartialOrd + Clone,
{
    // pivot square matrix
    let n = a.len_of(Axis(0));
    let mut p: Array2<f64> = Array::eye(n);
    let mut q: Array2<f64> = Array::eye(n);
    let mut at = a.to_owned();

    for j in 0..n {
        // iterate diagonally through
        let kr = argabsmax(at.slice(s![j.., j])) + j;
        let kc = argabsmax(at.slice(s![j, j..])) + j;

        match (kr, kc) {
            (kr, kc) if kr > j && kc > j => {
                if at[[kr, j]].abs() > at[[j, kc]].abs() {
                    row_swap(&mut p, &j, &kr);
                    row_swap(&mut at, &j, &kr);
                } else {
                    col_swap(&mut q, &j, &kc);
                    col_swap(&mut at, &j, &kc);
                }
            }
            (kr, kc) if kr > j && kc == j => {
                row_swap(&mut p, &j, &kr);
                row_swap(&mut at, &j, &kr);
            }
            (kr, kc) if kr == j && kc > j => {
                col_swap(&mut q, &j, &kc);
                col_swap(&mut at, &j, &kc);
            }
            _ => {}
        }
    }
    (p, q, at)
}

pub enum PivotMethod {
    Partial,
    Complete,
    Rook,
}

pub fn pluq_decomp<T>(
    a: &ArrayView2<T>,
    pivot: PivotMethod,
) -> (Array2<f64>, Array2<T>, Array2<T>, Array2<f64>)
where
    T: Signed + Num + PartialOrd + Clone + One + Zero + Sum + for<'a> Div<&'a T, Output = T>,
    for<'a> &'a T: Mul<&'a T, Output = T> + Sub<T, Output = T>,
{
    let n: usize = a.len_of(Axis(0));
    let mut l: Array2<T> = Array2::zeros((n, n));
    let mut u: Array2<T> = Array2::zeros((n, n));
    let p;
    let q;
    let paq;
    match pivot {
        PivotMethod::Partial => (p, q, paq) = partial_pivot_matrix(a),
        PivotMethod::Complete => (p, q, paq) = complete_pivot_matrix(a),
        PivotMethod::Rook => {
            (p, q, paq) = rook_pivot_matrix(a);
        }
    }

    let one = T::one();
    for j in 0..n {
        l[[j, j]] = one.clone(); // all diagonal entries of L are set to unity

        for i in 0..j + 1 {
            // LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
            let sx = dmul11_(&l.slice(s![i, ..i]), &u.slice(s![..i, j]));
            u[[i, j]] = &paq[[i, j]] - sx;
        }

        for i in j..n {
            // LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
            let sy = dmul11_(&l.slice(s![i, ..j]), &u.slice(s![..j, j]));
            l[[i, j]] = (&paq[[i, j]] - sy) / &u[[j, j]];
        }
    }
    (p, l, u, q)
}

fn dsolve_lower_1d<T>(l: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
where
    T: Clone + Sum + Zero + for<'a> Div<&'a T, Output = T>,
    for<'a> &'a T: Sub<T, Output = T> + Mul<&'a T, Output = T>,
{
    let n: usize = l.len_of(Axis(0));
    let mut x: Array1<T> = Array::zeros(n);
    for i in 0..n {
        let v = &b[i] - dmul11_(&l.slice(s![i, ..i]), &x.slice(s![..i]));
        x[i] = v / &l[[i, i]]
    }
    x
}

fn dsolve_upper_1d<T>(u: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
where
    T: Clone + Sum + Zero + for<'a> Div<&'a T, Output = T>,
    for<'a> &'a T: Sub<T, Output = T> + Mul<&'a T, Output = T>,
{
    // reverse all dimensions and solve as lower triangular
    dsolve_lower_1d(&u.slice(s![..;-1, ..;-1]), &b.slice(s![..;-1]))
        .slice(s![..;-1])
        .to_owned()
}

fn dsolve21_<T>(a: &ArrayView2<T>, b: &ArrayView1<T>) -> Array1<T>
where
    T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a T, Output = T>,
    for<'a> &'a T: Mul<&'a f64, Output = T> + Sub<T, Output = T> + Mul<&'a T, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
{
    let (p, l, u, q) = pluq_decomp::<T>(&a.view(), PivotMethod::Rook);
    let pb: Array1<T> = fdmul21_(&p.view(), &b.view());
    let z: Array1<T> = dsolve_lower_1d(&l.view(), &pb.view());
    let y: Array1<T> = dsolve_upper_1d(&u.view(), &z.view());
    let x: Array1<T> = fdmul21_(&q.view(), &y.view());
    x
}

pub fn dsolve<T>(a: &ArrayView2<T>, b: &ArrayView1<T>, allow_lsq: bool) -> Array1<T>
where
    T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a T, Output = T>,
    for<'a> &'a T: Mul<&'a f64, Output = T> + Sub<T, Output = T> + Mul<&'a T, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
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

    #[test]
    fn argabsmx2_i32() {
        let a: Array2<i32> = arr2(&[[-1, 2, 100], [-5, -2000, 0], [0, 0, 0]]);
        let result = argabsmax2(a.view());
        let expected: (usize, usize) = (1, 1);
        assert_eq!(result, expected);
    }

    #[test]
    fn argabsmx_dual() {
        let a: Array1<Dual> = arr1(&[
            Dual::new(1.0, Vec::new(), Vec::new()),
            Dual::new(-2.5, Vec::from(["a".to_string()]), Vec::from([2.0])),
        ]);
        let result = argabsmax(a.view());
        let expected: usize = 1;
        assert_eq!(result, expected);
    }

    #[test]
    fn pivot_f64_update() {
        let p: Array2<f64> = arr2(&[
            [1., 2., 3., 4.],
            [10., 2., 5., 6.],
            [7., 8., 1., 1.],
            [2., 2., 2., 9.],
        ]);
        let (result0, _, result1) = partial_pivot_matrix(&p.view());
        let expected0: Array2<f64> = arr2(&[
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ]);
        let expected1: Array2<f64> = arr2(&[
            [10., 2., 5., 6.],
            [7., 8., 1., 1.],
            [1., 2., 3., 4.],
            [2., 2., 2., 9.],
        ]);
        assert_eq!(result0, expected0);
        assert_eq!(result1, expected1);
    }

    #[test]
    fn pivot_i32_complete() {
        let i: Array2<i32> = arr2(&[[1, 2, 3, 4], [10, 2, 5, 6], [7, 8, 1, 1], [2, 2, 2, 9]]);
        let (_p, _q, at) = complete_pivot_matrix(&i.view());
        let expected2: Array2<i32> =
            arr2(&[[10, 6, 2, 5], [2, 9, 2, 2], [7, 1, 8, 1], [1, 4, 2, 3]]);
        assert_eq!(at, expected2);
    }

    #[test]
    fn pluq_decomp_dual() {
        let a = arr2(&[
            [
                Dual::new(1.0, Vec::new(), Vec::new()),
                Dual::new(2.0, Vec::new(), Vec::new()),
                Dual::new(3.0, Vec::new(), Vec::new()),
                Dual::new(4.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(10.0, Vec::new(), Vec::new()),
                Dual::new(2.0, Vec::new(), Vec::new()),
                Dual::new(5.0, Vec::new(), Vec::new()),
                Dual::new(6.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(7.0, Vec::new(), Vec::new()),
                Dual::new(8.0, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(2.0, Vec::new(), Vec::new()),
                Dual::new(2.0, Vec::new(), Vec::new()),
                Dual::new(2.0, Vec::new(), Vec::new()),
                Dual::new(9.0, Vec::new(), Vec::new()),
            ],
        ]);
        let (p, l, u, q) = pluq_decomp(&a.view(), PivotMethod::Complete);

        let expected_p = arr2(&[
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ]);
        assert_eq!(p, expected_p);

        let expected_l = arr2(&[
            [
                Dual::new(1.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(0.2, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(0.7000000000000001, Vec::new(), Vec::new()),
                Dual::new(-0.41025641025641035, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(0.1, Vec::new(), Vec::new()),
                Dual::new(0.43589743589743596, Vec::new(), Vec::new()),
                Dual::new(0.1519434628975265, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
            ],
        ]);
        assert_eq!(l, expected_l);

        let expected_u = arr2(&[
            [
                Dual::new(10.0, Vec::new(), Vec::new()),
                Dual::new(6.0, Vec::new(), Vec::new()),
                Dual::new(2.0, Vec::new(), Vec::new()),
                Dual::new(5.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(7.8, Vec::new(), Vec::new()),
                Dual::new(1.6, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(7.256410256410256, Vec::new(), Vec::new()),
                Dual::new(-2.0897435897435903, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(2.381625441696113, Vec::new(), Vec::new()),
            ],
        ]);
        assert_eq!(u, expected_u);

        let expected_q = arr2(&[
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
        ]);
        assert_eq!(q, expected_q);
    }

    #[test]
    fn lower_tri_dual() {
        let a = arr2(&[
            [
                Dual::new(1.0, Vec::new(), Vec::new()),
                Dual::new(0.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(2.0, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
            ],
        ]);
        let b = arr1(&[
            Dual::new(2.0, Vec::new(), Vec::new()),
            Dual::new(5.0, Vec::new(), Vec::new()),
        ]);
        let x = dsolve_lower_1d(&a.view(), &b.view());
        let expected_x = arr1(&[
            Dual::new(2.0, Vec::new(), Vec::new()),
            Dual::new(1.0, Vec::new(), Vec::new()),
        ]);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn upper_tri_dual() {
        let a = arr2(&[
            [
                Dual::new(1.0, Vec::new(), Vec::new()),
                Dual::new(2.0, Vec::new(), Vec::new()),
            ],
            [
                Dual::new(0.0, Vec::new(), Vec::new()),
                Dual::new(1.0, Vec::new(), Vec::new()),
            ],
        ]);
        let b = arr1(&[
            Dual::new(2.0, Vec::new(), Vec::new()),
            Dual::new(5.0, Vec::new(), Vec::new()),
        ]);
        let x = dsolve_upper_1d(&a.view(), &b.view());
        let expected_x = arr1(&[
            Dual::new(-8.0, Vec::new(), Vec::new()),
            Dual::new(5.0, Vec::new(), Vec::new()),
        ]);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn dsolve_dual() {
        let a: Array2<Dual> = Array2::eye(2);
        let b: Array1<Dual> = arr1(&[
            Dual::new(2.0, vec!["x".to_string()], vec![1.0]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 1.0]),
        ]);
        let result = dsolve(&a.view(), &b.view(), false);
        let expected = arr1(&[
            Dual::new(2.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]),
            Dual::new(5.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 1.0]),
        ]);
        assert_eq!(result, expected);
        assert!(Arc::ptr_eq(&result[0].vars, &result[1].vars));
    }

    #[test]
    fn pluq_dual_sparse() {
        fn d(f: f64) -> Dual {
            Dual::new(f, Vec::new(), Vec::new())
        }

        let a = arr2(&[
            [
                d(24.),
                d(-36.),
                d(12.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
            ],
            [
                d(1.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
            ],
            [
                d(0.),
                d(0.25),
                d(0.583333333333),
                d(0.16666666666),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
            ],
            [
                d(0.),
                d(0.),
                d(0.16666666666),
                d(0.66666666666),
                d(0.16666666666),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
            ],
            [
                d(0.),
                d(0.),
                d(0.),
                d(0.16666666666),
                d(0.66666666666),
                d(0.16666666666),
                d(0.),
                d(0.),
                d(0.),
            ],
            [
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.16666666666),
                d(0.66666666666),
                d(0.16666666666),
                d(0.),
                d(0.),
            ],
            [
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.16666666666),
                d(0.583333333333),
                d(0.25),
                d(0.),
            ],
            [
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(1.),
            ],
            [
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(0.),
                d(12.),
                d(-36.),
                d(24.),
            ],
        ]);
        let (p, l, u, q) = pluq_decomp(&a.view(), PivotMethod::Rook);
        println!("L: {:?}", l);
        println!("U: {:?}", u);

        let pa = fdmul22_(&p.view(), &a.view());
        let paq = dfmul22_(&pa.view(), &q.view());

        let lu = dmul22_(&l.view(), &u.view());
        println!("PAQ: {:?}", paq);
        println!("LU: {:?}", lu);

        let v: Vec<bool> = paq
            .into_raw_vec()
            .iter()
            .zip(lu.into_raw_vec().iter())
            .map(|(x, y)| is_close(&x.real, &y.real, None))
            .collect();

        assert!(v.iter().all(|x| *x));
    }

    #[test]
    fn pluq_dual_sparse_3x3() {
        fn d(f: f64) -> Dual {
            Dual::new(f, Vec::new(), Vec::new())
        }

        let a = arr2(&[
            [d(0.), d(2.), d(1.)],
            [d(1.), d(2.), d(4.)],
            [d(5.), d(3.0), d(0.)],
        ]);

        let b = arr1(&[d(1.), d(2.), d(3.)]);
        let expected = arr1(&[
            d(0.36363636363636365),
            d(0.393939393939394),
            d(0.2121212121212121),
        ]);
        let result = dsolve(&a.view(), &b.view(), false);

        let v: Vec<bool> = expected
            .into_raw_vec()
            .iter()
            .zip(result.into_raw_vec().iter())
            .map(|(x, y)| is_close(&x.real, &y.real, None))
            .collect();
        assert!(v.iter().all(|x| *x));
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