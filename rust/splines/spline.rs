use crate::dual::linalg::{dmul11_, fdmul11_, fdsolve, fouter11_};
use crate::dual::{Dual, Dual2, Gradient1, Gradient2, Number, NumberMapping};
use ndarray::{Array1, Array2};
use num_traits::{Signed, Zero};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::PyErr;
use serde::{Deserialize, Serialize};
use std::{
    cmp::PartialEq,
    iter::{zip, Sum},
    ops::{Mul, Sub},
};

/// Evaluate the `x` value on the `i`'th B-spline with order `k` and knot sequence `t`.
///
/// Note `org_k` should be input as None, it is used internally for recursively calculating
/// spline derivatives, where it is set to the original `k` value from the outer scope.
pub fn bsplev_single_f64(x: &f64, i: usize, k: &usize, t: &Vec<f64>, org_k: Option<usize>) -> f64 {
    let org_k: usize = org_k.unwrap_or(*k);

    // Short circuit (positivity and support property)
    if *x < t[i] || *x > t[i + k] {
        return 0.0_f64;
    }

    // Right side end point support
    if *x == t[t.len() - 1] && i >= (t.len() - org_k - 1) {
        return 1.0_f64;
    }

    // Recursion
    if *k == 1_usize {
        if t[i] <= *x && *x < t[i + 1] {
            1.0_f64
        } else {
            0.0_f64
        }
    } else {
        let mut left: f64 = 0.0_f64;
        let mut right: f64 = 0.0_f64;
        if t[i] != t[i + k - 1] {
            left = (x - t[i]) / (t[i + k - 1] - t[i]) * bsplev_single_f64(x, i, &(k - 1), t, None);
        }
        if t[i + 1] != t[i + k] {
            right = (t[i + k] - x) / (t[i + k] - t[i + 1])
                * bsplev_single_f64(x, i + 1, &(k - 1), t, None);
        }
        left + right
    }
}

/// Evaluate the `x` value on the `i`'th B-spline with order `k` and knot sequence `t`.
///
/// Note `org_k` should be input as None, it is used internally for recursively calculating
/// spline derivatives, where it is set to the original `k` value from the outer scope.
pub fn bsplev_single_dual(
    x: &Dual,
    i: usize,
    k: &usize,
    t: &Vec<f64>,
    org_k: Option<usize>,
) -> Dual {
    let b_f64 = bsplev_single_f64(&x.real(), i, k, t, org_k);
    let dbdx_f64 = bspldnev_single_f64(&x.real(), i, k, t, 1, org_k);
    Dual::clone_from(x, b_f64, dbdx_f64 * x.dual())
}

/// Evaluate the `x` value on the `i`'th B-spline with order `k` and knot sequence `t`.
///
/// Note `org_k` should be input as None, it is used internally for recursively calculating
/// spline derivatives, where it is set to the original `k` value from the outer scope.
pub fn bsplev_single_dual2(
    x: &Dual2,
    i: usize,
    k: &usize,
    t: &Vec<f64>,
    org_k: Option<usize>,
) -> Dual2 {
    let b_f64 = bsplev_single_f64(&x.real(), i, k, t, org_k);
    let dbdx_f64 = bspldnev_single_f64(&x.real(), i, k, t, 1, org_k);
    let d2bdx2_f64 = bspldnev_single_f64(&x.real(), i, k, t, 2, org_k);
    let dual2 =
        dbdx_f64 * x.dual2() + 0.5 * d2bdx2_f64 * fouter11_(&x.dual().view(), &x.dual().view());
    Dual2::clone_from(x, b_f64, dbdx_f64 * x.dual(), dual2)
}

/// Evaluate the `m`'th order derivative of the `x` value on the `i`'th B-spline with
/// order `k` and knot sequence `t`.
///
/// Note `org_k` should be input as None, it is used internally for recursively calculating
/// spline derivatives, where it is set to the original `k` value from the outer scope.
pub fn bspldnev_single_f64(
    x: &f64,
    i: usize,
    k: &usize,
    t: &Vec<f64>,
    m: usize,
    org_k: Option<usize>,
) -> f64 {
    if m == 0 {
        return bsplev_single_f64(x, i, k, t, None);
    } else if *k == 1 || m >= *k {
        return 0.0_f64;
    }

    let org_k: usize = org_k.unwrap_or(*k);
    let mut r: f64 = 0.0;
    let div1: f64 = t[i + k - 1] - t[i];
    let div2: f64 = t[i + k] - t[i + 1];

    if m == 1 {
        if div1 != 0_f64 {
            r += bsplev_single_f64(x, i, &(k - 1), t, Some(org_k)) / div1;
        }
        if div2 != 0_f64 {
            r -= bsplev_single_f64(x, i + 1, &(k - 1), t, Some(org_k)) / div2;
        }
        r *= (k - 1) as f64;
    } else {
        if div1 != 0_f64 {
            r += bspldnev_single_f64(x, i, &(k - 1), t, m - 1, Some(org_k)) / div1;
        }
        if div2 != 0_f64 {
            r -= bspldnev_single_f64(x, i + 1, &(k - 1), t, m - 1, Some(org_k)) / div2;
        }
        r *= (k - 1) as f64
    }
    r
}

/// Evaluate the `m`'th order derivative of the `x` value on the `i`'th B-spline with
/// order `k` and knot sequence `t`.
///
/// Note `org_k` should be input as None, it is used internally for recursively calculating
/// spline derivatives, where it is set to the original `k` value from the outer scope.
pub fn bspldnev_single_dual(
    x: &Dual,
    i: usize,
    k: &usize,
    t: &Vec<f64>,
    m: usize,
    org_k: Option<usize>,
) -> Dual {
    let b_f64 = bspldnev_single_f64(&x.real(), i, k, t, m, org_k);
    let dbdx_f64 = bspldnev_single_f64(&x.real(), i, k, t, m + 1, org_k);
    Dual::clone_from(x, b_f64, dbdx_f64 * x.dual())
}

/// Evaluate the `m`'th order derivative of the `x` value on the `i`'th B-spline with
/// order `k` and knot sequence `t`.
///
/// Note `org_k` should be input as None, it is used internally for recursively calculating
/// spline derivatives, where it is set to the original `k` value from the outer scope.
pub fn bspldnev_single_dual2(
    x: &Dual2,
    i: usize,
    k: &usize,
    t: &Vec<f64>,
    m: usize,
    org_k: Option<usize>,
) -> Dual2 {
    let b_f64 = bspldnev_single_f64(&x.real(), i, k, t, m, org_k);
    let dbdx_f64 = bspldnev_single_f64(&x.real(), i, k, t, m + 1, org_k);
    let d2bdx2_f64 = bspldnev_single_f64(&x.real(), i, k, t, m + 2, org_k);
    let dual2 =
        dbdx_f64 * x.dual2() + 0.5 * d2bdx2_f64 * fouter11_(&x.dual().view(), &x.dual().view());
    Dual2::clone_from(x, b_f64, dbdx_f64 * x.dual(), dual2)
}

/// A piecewise polynomial spline of given order and knot sequence.
#[derive(Clone, Deserialize, Serialize)]
pub struct PPSpline<T> {
    k: usize,
    t: Vec<f64>,
    c: Option<Array1<T>>,
    n: usize,
}

impl<T> PPSpline<T> {
    pub fn k(&self) -> &usize {
        &self.k
    }

    pub fn t(&self) -> &Vec<f64> {
        &self.t
    }

    pub fn n(&self) -> &usize {
        &self.n
    }

    pub fn c(&self) -> &Option<Array1<T>> {
        &self.c
    }
}

impl<T> PPSpline<T>
where
    T: PartialOrd + Signed + Clone + Sum + Zero,
    for<'a> &'a T: Sub<&'a T, Output = T>,
    for<'a> &'a f64: Mul<&'a T, Output = T>,
{
    /// Create a PPSpline from its order `k`, knot sequence `t` and optional spline coefficents `c`.
    pub fn new(k: usize, t: Vec<f64>, c: Option<Vec<T>>) -> Self {
        // t is given and is non-decreasing
        assert!(t.len() > 1);
        assert!(zip(&t[1..], &t[..(t.len() - 1)]).all(|(a, b)| a >= b));
        let n = t.len() - k;
        let c_ = c.map(Array1::from_vec);
        PPSpline { k, t, n, c: c_ }
    }

    pub fn ppdnev_single(&self, x: &f64, m: usize) -> Result<T, PyErr> {
        let b: Array1<f64> = Array1::from_vec(
            (0..self.n)
                .map(|i| bspldnev_single_f64(x, i, &self.k, &self.t, m, None))
                .collect(),
        );
        match &self.c {
            Some(c) => Ok(fdmul11_(&b.view(), &c.view())),
            None => Err(PyValueError::new_err(
                "Must call `csolve` before evaluating PPSpline.",
            )),
        }
    }

    pub fn csolve(
        &mut self,
        tau: &[f64],
        y: &[T],
        left_n: usize,
        right_n: usize,
        allow_lsq: bool,
    ) -> Result<(), PyErr> {
        if tau.len() != self.n && !(allow_lsq && tau.len() > self.n) {
            return Err(PyValueError::new_err(
                "`csolve` cannot complete if length of `tau` < n or `allow_lsq` is false.",
            ));
        }
        if tau.len() != y.len() {
            return Err(PyValueError::new_err(
                "`tau` and `y` must have the same length.",
            ));
        }
        let b: Array2<f64> = self.bsplmatrix(tau, left_n, right_n);
        let ya: Array1<T> = Array1::from_vec(y.to_owned());
        let c: Array1<T> = fdsolve(&b.view(), &ya.view(), allow_lsq);
        self.c = Some(c);
        Ok(())
    }

    // pub fn bsplev(&self, x: &Vec<f64>, i: &usize) -> Vec<f64> {
    //     x.iter().map(|v| bsplev_single_f64(v, *i, self.k(), self.t(), None)).collect()
    // }

    pub fn bspldnev(&self, x: &[f64], i: &usize, m: &usize) -> Vec<f64> {
        x.iter()
            .map(|v| bspldnev_single_f64(v, *i, self.k(), self.t(), *m, None))
            .collect()
    }

    pub fn bsplmatrix(&self, tau: &[f64], left_n: usize, right_n: usize) -> Array2<f64> {
        let mut b = Array2::zeros((tau.len(), self.n));
        for i in 0..self.n {
            b[[0, i]] = bspldnev_single_f64(&tau[0], i, &self.k, &self.t, left_n, None);
            b[[tau.len() - 1, i]] =
                bspldnev_single_f64(&tau[tau.len() - 1], i, &self.k, &self.t, right_n, None);
            for j in 1..(tau.len() - 1) {
                b[[j, i]] = bsplev_single_f64(&tau[j], i, &self.k, &self.t, None)
            }
        }
        b
    }
}

impl NumberMapping for PPSpline<f64> {
    fn mapped_value(&self, x: &Number) -> Result<Number, PyErr> {
        match x {
            Number::F64(f) => Ok(Number::F64(self.ppdnev_single(f, 0_usize)?)),
            Number::Dual(d) => Ok(Number::Dual(self.ppdnev_single_dual(d, 0_usize)?)),
            Number::Dual2(d) => Ok(Number::Dual2(self.ppdnev_single_dual2(d, 0_usize)?)),
        }
    }
}

impl PPSpline<f64> {
    pub fn ppdnev_single_dual(&self, x: &Dual, m: usize) -> Result<Dual, PyErr> {
        let b: Array1<Dual> = Array1::from_vec(
            (0..self.n)
                .map(|i| bspldnev_single_dual(x, i, &self.k, &self.t, m, None))
                .collect(),
        );
        match &self.c {
            Some(c) => Ok(fdmul11_(&c.view(), &b.view())),
            None => Err(PyValueError::new_err(
                "Must call `csolve` before evaluating PPSpline.",
            )),
        }
    }

    pub fn ppdnev_single_dual2(&self, x: &Dual2, m: usize) -> Result<Dual2, PyErr> {
        let b: Array1<Dual2> = Array1::from_vec(
            (0..self.n)
                .map(|i| bspldnev_single_dual2(x, i, &self.k, &self.t, m, None))
                .collect(),
        );
        match &self.c {
            Some(c) => Ok(fdmul11_(&c.view(), &b.view())),
            None => Err(PyValueError::new_err(
                "Must call `csolve` before evaluating PPSpline.",
            )),
        }
    }
}

impl NumberMapping for PPSpline<Dual> {
    fn mapped_value(&self, x: &Number) -> Result<Number, PyErr> {
        match x {
            Number::F64(f) => Ok(Number::Dual(self.ppdnev_single(f, 0_usize)?)),
            Number::Dual(d) => Ok(Number::Dual(self.ppdnev_single_dual(d, 0_usize)?)),
            Number::Dual2(d) => Ok(Number::Dual2(self.ppdnev_single_dual2(d, 0_usize)?)),
        }
    }
}

impl PPSpline<Dual> {
    pub fn ppdnev_single_dual2(&self, _x: &Dual2, _m: usize) -> Result<Dual2, PyErr> {
        Err(PyTypeError::new_err(
            "Cannot index with type `Dual2` on PPSpline<Dual>`.",
        ))
    }

    pub fn ppdnev_single_dual(&self, x: &Dual, m: usize) -> Result<Dual, PyErr> {
        let b: Array1<Dual> = Array1::from_vec(
            (0..self.n)
                .map(|i| bspldnev_single_dual(x, i, &self.k, &self.t, m, None))
                .collect(),
        );
        match &self.c {
            Some(c) => Ok(dmul11_(&c.view(), &b.view())),
            None => Err(PyValueError::new_err(
                "Must call `csolve` before evaluating PPSpline.",
            )),
        }
    }
}

impl NumberMapping for PPSpline<Dual2> {
    fn mapped_value(&self, x: &Number) -> Result<Number, PyErr> {
        match x {
            Number::F64(f) => Ok(Number::Dual2(self.ppdnev_single(f, 0_usize)?)),
            Number::Dual(d) => Ok(Number::Dual(self.ppdnev_single_dual(d, 0_usize)?)),
            Number::Dual2(d) => Ok(Number::Dual2(self.ppdnev_single_dual2(d, 0_usize)?)),
        }
    }
}

impl PPSpline<Dual2> {
    pub fn ppdnev_single_dual(&self, _x: &Dual, _m: usize) -> Result<Dual, PyErr> {
        Err(PyTypeError::new_err(
            "Cannot index with type `Dual` on PPSpline<Dual2>.",
        ))
    }

    pub fn ppdnev_single_dual2(&self, x: &Dual2, m: usize) -> Result<Dual2, PyErr> {
        let b: Array1<Dual2> = Array1::from_vec(
            (0..self.n)
                .map(|i| bspldnev_single_dual2(x, i, &self.k, &self.t, m, None))
                .collect(),
        );
        match &self.c {
            Some(c) => Ok(dmul11_(&c.view(), &b.view())),
            None => Err(PyValueError::new_err(
                "Must call `csolve` before evaluating PPSpline.",
            )),
        }
    }
}

impl<T> PartialEq for PPSpline<T>
where
    T: PartialEq,
{
    /// Equality of `PPSpline` if

    fn eq(&self, other: &Self) -> bool {
        if self.k != other.k || self.n != other.n {
            return false;
        }
        if !self.t.eq(&other.t) {
            return false;
        }
        match (&self.c, &other.c) {
            (Some(c1), Some(c2)) => c1.eq(&c2),
            _ => false, // if any c is None then false
        }
    }
}

// UNIT TESTS

//

//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::Dual;
    use ndarray::{arr1, arr2};
    use num_traits::One;

    fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
        // used rather than equality for float numbers
        (a - b).abs() < abs_tol.unwrap_or(1e-8)
    }

    #[test]
    fn bsplev_single_f64_() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8)
            .map(|i| bsplev_single_f64(&x, i as usize, &k, &t, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[0.125, 0.375, 0.375, 0.125, 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bsplev_single_dual_() {
        let x: Dual = Dual::new(1.5, vec!["x".to_string()]);
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<Dual> = (0..8)
            .map(|i| bsplev_single_dual(&x, i as usize, &k, &t, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[0.125, 0.375, 0.375, 0.125, 0., 0., 0., 0.]);
        for i in 0..8 {
            assert_eq!(result[i].real(), expected[i])
        }
        // These are values from the bspldnev_single evaluation test
        let dual_expected: Vec<f64> = Vec::from(&[-0.75, -0.75, 0.75, 0.75, 0., 0., 0., 0.]);
        for i in 0..8 {
            assert_eq!(result[i].dual()[0], dual_expected[i])
        }
    }

    #[test]
    fn bsplev_single_f64_right() {
        let x: f64 = 4.0_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8)
            .map(|i| bsplev_single_f64(&x, i as usize, &k, &t, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 0., 0., 1.0]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_f64_() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8)
            .map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 1_usize, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[-0.75, -0.75, 0.75, 0.75, 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_f64_right() {
        let x: f64 = 4.0_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8)
            .map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 1_usize, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 0., -3., 3.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_shortcut() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8)
            .map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 6_usize, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_m() {
        let x: f64 = 4.0_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8)
            .map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 2_usize, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 3., -9., 6.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_m_zero() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8)
            .map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 0_usize, None))
            .collect();
        let expected: Vec<f64> = Vec::from(&[0.125, 0.375, 0.375, 0.125, 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn ppspline_new() {
        let _pps: PPSpline<f64> = PPSpline::new(
            4,
            vec![1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.],
            None,
        );
    }

    #[test]
    fn ppspline_bsplmatrix() {
        let pps: PPSpline<f64> = PPSpline::new(4, vec![1., 1., 1., 1., 2., 3., 3., 3., 3.], None);
        let result = pps.bsplmatrix(&vec![1., 1., 2., 3., 3.], 2_usize, 2_usize);
        let expected: Array2<f64> = arr2(&[
            [6., -9., 3., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0.25, 0.5, 0.25, 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 3., -9., 6.],
        ]);
        assert_eq!(result, expected)
    }

    #[test]
    fn csolve_() {
        let t = vec![0., 0., 0., 0., 4., 4., 4., 4.];
        let tau = vec![0., 1., 3., 4.];
        let val = vec![0., 0., 2., 2.];
        let mut pps: PPSpline<f64> = PPSpline::new(4, t, None);
        let _ = pps.csolve(&tau, &val, 0, 0, false);
        let expected = vec![0., -1.11111111, 3.111111111111, 2.0];
        let v: Vec<bool> = pps
            .c
            .expect("csolve")
            .into_raw_vec()
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| is_close(&x, &y, None))
            .collect();

        assert!(v.iter().all(|x| *x));
    }

    #[test]
    fn csolve_dual() {
        let t = vec![0., 0., 0., 0., 4., 4., 4., 4.];
        let tau = vec![0., 1., 3., 4.];
        let d1 = Dual::one();
        let val = vec![0. * &d1, 0. * &d1, 2. * &d1, 2. * &d1];
        let mut pps = PPSpline::new(4, t, None);
        let _ = pps.csolve(&tau, &val, 0, 0, false);
        let expected = vec![0. * &d1, -1.11111111 * &d1, 3.111111111111 * &d1, 2.0 * &d1];
        let v: Vec<bool> = pps
            .c
            .expect("csolve")
            .into_raw_vec()
            .iter()
            .zip(expected.iter())
            .map(|(x, y)| is_close(&x.real(), &y.real(), None))
            .collect();

        assert!(v.iter().all(|x| *x));
    }

    #[test]
    fn ppev_single_() {
        let t = vec![1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.];
        let mut pps = PPSpline::new(4, t, None);
        pps.c = Some(arr1(&[1., 2., -1., 2., 1., 1., 2., 2.]));
        let r1 = pps.ppdnev_single(&1.1, 0).unwrap();
        assert!(is_close(&r1, &1.19, None));
        let r2 = pps.ppdnev_single(&1.8, 0).unwrap();
        assert!(is_close(&r2, &0.84, None));
        let r3 = pps.ppdnev_single(&2.8, 0).unwrap();
        assert!(is_close(&r3, &1.136, None));
    }

    #[test]
    fn partialeq_() {
        let pp1 = PPSpline::<f64>::new(2, vec![1., 1., 2., 2.], None);
        let pp2 = PPSpline::<f64>::new(2, vec![1., 1., 2., 2.], None);
        assert!(pp1 != pp2);
        let pp1 = PPSpline::new(2, vec![1., 1., 2., 2.], Some(vec![1.5, 0.2]));
        let pp2 = PPSpline::new(2, vec![1., 1., 2., 2.], Some(vec![1.5, 0.2]));
        assert!(pp1 == pp2);
    }

    #[test]
    #[should_panic]
    fn backwards_definition() {
        let _pp1 = PPSpline::<f64>::new(4, vec![3., 3., 3., 3., 2., 1., 1., 1., 1.], None);
    }
}
