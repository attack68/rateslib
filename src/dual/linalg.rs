use crate::dual::dual1::Dual;
use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::{ArrayBase, Zip};
// use ndarray_linalg::Solve;
use num_traits::{Num, Signed};
use std::cmp::PartialOrd;
// use std::iter::Sum;
use std::ops::Mul;
use std::sync::Arc;

fn dmul11_(a: &ArrayView1<Dual>, b: &ArrayView1<Dual>) -> Dual {
    if a.len() != b.len() {
        panic!("Lengths of LHS and RHS do not match.")
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn fdmul11_(a: &ArrayView1<f64>, b: &ArrayView1<Dual>) -> Dual {
    if a.len() != b.len() {
        panic!("Lengths of LHS and RHS do not match.")
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dmul22_(a: &ArrayView2<Dual>, b: &ArrayView2<Dual>) -> Array2<Dual> {
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

fn fdmul22_(a: &ArrayView2<f64>, b: &ArrayView2<Dual>) -> Array2<Dual> {
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out = Array2::zeros((a.len_of(Axis(0)), b.len_of(Axis(1))));
    for r in 0..a.len_of(Axis(0)) {
        for c in 0..b.len_of(Axis(1)) {
            out[[r, c]] = fdmul11_(&a.row(r), &b.column(c))
        }
    }
    out
}

fn dmul21_(a: &ArrayView2<Dual>, b: &ArrayView1<Dual>) -> Array1<Dual> {
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out = Array1::zeros(b.len_of(Axis(0)));
    for r in 0..a.len_of(Axis(0)) {
        out[[r]] = dmul11_(&a.row(r), &b)
    }
    out
}

fn fdmul21_(a: &ArrayView2<f64>, b: &ArrayView1<Dual>) -> Array1<Dual> {
    if a.len_of(Axis(1)) != b.len_of(Axis(0)) {
        panic!("Columns of LHS do not match rows of RHS.")
    }
    let mut out = Array1::zeros(b.len_of(Axis(0)));
    for r in 0..a.len_of(Axis(0)) {
        out[[r]] = fdmul11_(&a.row(r), &b)
    }
    out
}

fn ddot(a: &Array1<Dual>, b: &Array1<Dual>) -> Dual {
    dmul11_(&a.view(), &b.view())
}

fn fdot(a: &Array1<f64>, b: &Array1<Dual>) -> Dual {
    fdmul11_(&a.view(), &b.view())
}

fn dmul(a: &Array2<Dual>, b: &Array2<Dual>) -> Array2<Dual> {
    dmul22_(&a.view(), &b.view())
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

fn partial_pivot_matrix<T>(A: &Array2<T>) -> (Array2<i32>, Array2<T>)
where
    T: Signed + Num + PartialOrd + Clone,
{
    // pivot square matrix
    let n = A.len_of(Axis(0));
    let mut P: Array2<i32> = Array::eye(n);
    let mut Pa = A.to_owned(); // initialise PA and Original (or)
                               // let Or = A.to_owned();
    for j in 0..n {
        let k = argabsmax(Pa.slice(s![j.., j])) + j;
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

fn complete_pivot_matrix<T>(A: &ArrayView2<T>) -> (Array2<f64>, Array2<f64>, Array2<T>)
where
    T: Signed + Num + PartialOrd + Clone,
{
    // pivot square matrix
    let n = A.len_of(Axis(0));
    let mut p: Array2<f64> = Array::eye(n);
    let mut q: Array2<f64> = Array::eye(n);
    let mut at = A.to_owned();

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

fn pluq_decomp(a: &ArrayView2<Dual>) -> (Array2<f64>, Array2<Dual>, Array2<Dual>, Array2<f64>) {
    let n: usize = a.len_of(Axis(0));
    let mut l: Array2<Dual> = Array2::zeros((n, n));
    let mut u: Array2<Dual> = Array2::zeros((n, n));
    let (p, q, paq) = complete_pivot_matrix(&a);

    let one = Dual::new(1.0, Vec::new(), Vec::new());
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

fn solve_lower_1d(l: &ArrayView2<Dual>, b: &ArrayView1<Dual>) -> Array1<Dual> {
    let n: usize = l.len_of(Axis(0));
    let mut x: Array1<Dual> = Array::zeros(n);
    for i in 0..n {
        let v = &b[i] - dmul11_(&l.slice(s![i, ..i]), &x.slice(s![..i]));
        x[i] = v / &l[[i, i]]
    }
    x
}

fn solve_upper_1d(u: &ArrayView2<Dual>, b: &ArrayView1<Dual>) -> Array1<Dual> {
    solve_lower_1d(&u.t(), &b.slice(s![..;-1]))
        .slice(s![..;-1])
        .to_owned()
}

fn dsolve21_(a: &ArrayView2<Dual>, b: &ArrayView1<Dual>) -> Array1<Dual> {
    let (p, l, u, q) = pluq_decomp(&a.view());
    let pb = fdmul21_(&p.view(), &b.view());
    let z = solve_lower_1d(&l.view(), &pb.view());
    let y = solve_upper_1d(&u.view(), &z.view());
    let x = fdmul21_(&q.view(), &y.view());
    x
}

pub fn dsolve(a: &Array2<Dual>, b: &Array1<Dual>, allow_lsq: bool) -> Array1<Dual> {
    if allow_lsq {
        let a_ = dmul22_(&a.t(), &a.view());
        let b_ = dmul21_(&a.t(), &b.view());
        dsolve21_(&a_.view(), &b_.view())
    } else {
        dsolve21_(&a.view(), &b.view())
    }
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
fn argabsmx2_i32() {
    let A: Array2<i32> = arr2(&[[-1, 2, 100], [-5, -2000, 0], [0, 0, 0]]);
    let result = argabsmax2(A.view());
    let expected: (usize, usize) = (1, 1);
    assert_eq!(result, expected);
}

#[test]
fn argabsmx_dual() {
    let A: Array1<Dual> = arr1(&[
        Dual::new(1.0, Vec::new(), Vec::new()),
        Dual::new(-2.5, Vec::from(["a".to_string()]), Vec::from([2.0])),
    ]);
    let result = argabsmax(A.view());
    let expected: usize = 1;
    assert_eq!(result, expected);
}

#[test]
fn pivot_i32_update() {
    let P: Array2<i32> = arr2(&[[1, 2, 3, 4], [10, 2, 5, 6], [7, 8, 1, 1], [2, 2, 2, 9]]);
    let (result0, result1) = partial_pivot_matrix(&P);
    let expected0: Array2<i32> = arr2(&[[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]);
    let expected1: Array2<i32> = arr2(&[[10, 2, 5, 6], [7, 8, 1, 1], [1, 2, 3, 4], [2, 2, 2, 9]]);
    assert_eq!(result0, expected0);
    assert_eq!(result1, expected1);
}

#[test]
fn pivot_i32_complete() {
    let i: Array2<i32> = arr2(&[[1, 2, 3, 4], [10, 2, 5, 6], [7, 8, 1, 1], [2, 2, 2, 9]]);
    let (p, q, at) = complete_pivot_matrix(&i.view());
    let expected2: Array2<i32> = arr2(&[[10, 6, 2, 5], [2, 9, 2, 2], [7, 1, 8, 1], [1, 4, 2, 3]]);
    assert_eq!(at, expected2);
}

#[test]
fn ddot_dual() {
    let a = arr1(&[
        Dual::new(1.0, Vec::new(), Vec::new()),
        Dual::new(2.0, Vec::new(), Vec::new()),
    ]);
    let b = arr1(&[
        Dual::new(3.0, Vec::new(), Vec::new()),
        Dual::new(3.0, Vec::new(), Vec::new()),
    ]);
    let result = ddot(&a, &b);
    let expected = Dual::new(9.0, Vec::new(), Vec::new());
    assert_eq!(result, expected);
}

#[test]
fn dmul_dual() {
    let a = arr2(&[
        [
            Dual::new(1.0, Vec::new(), Vec::new()),
            Dual::new(2.0, Vec::new(), Vec::new()),
        ],
        [
            Dual::new(3.0, Vec::new(), Vec::new()),
            Dual::new(4.0, Vec::new(), Vec::new()),
        ],
    ]);
    let b = arr2(&[
        [
            Dual::new(3.0, Vec::new(), Vec::new()),
            Dual::new(3.0, Vec::new(), Vec::new()),
        ],
        [
            Dual::new(4.0, Vec::new(), Vec::new()),
            Dual::new(5.0, Vec::new(), Vec::new()),
        ],
    ]);
    let result = dmul(&a, &b);
    println!("{:?}", result);
    let expected = arr2(&[
        [
            Dual::new(11.0, Vec::new(), Vec::new()),
            Dual::new(13.0, Vec::new(), Vec::new()),
        ],
        [
            Dual::new(25.0, Vec::new(), Vec::new()),
            Dual::new(29.0, Vec::new(), Vec::new()),
        ],
    ]);
    assert_eq!(result, expected);
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
    let (p, l, u, q) = pluq_decomp(&a.view());

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
    let x = solve_lower_1d(&a.view(), &b.view());
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
    let x = solve_upper_1d(&a.view(), &b.view());
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
    let result = dsolve(&a, &b, false);
    println!("{:?}", result);
    let expected = arr1(&[
        Dual::new(2.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]),
        Dual::new(5.0, vec!["x".to_string(), "y".to_string()], vec![1.0, 1.0]),
    ]);
    assert_eq!(result, expected);
    assert!(Arc::ptr_eq(&result[0].vars, &result[1].vars));
}

// #[test]
// fn ndarray_broadcast_dual() {
//     let a = arr1(&[
//         Dual::new(1.0, Vec::new(), Vec::new()),
//         Dual::new(2.0, Vec::new(), Vec::new()),
//     ]);
//     let b = Dual::new(2.5, Vec::new(), Vec::new());
//     let c = b * a;
//     println!("{:?}", c);
//     assert_eq!(1, 2);
// }
