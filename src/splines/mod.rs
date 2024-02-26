use ndarray::{Array1, Array2};
use crate::dual::linalg_f64::{fdsolve, fdmul11_};
use std::ops::{Mul, Div, Sub};
use num_traits::{Signed, Zero};
use std::iter::Sum;

fn bsplev_single_f64(x: &f64, i: usize, k: &usize, t: &Vec<f64>, org_k: Option<usize>) -> f64 {
    let org_k: usize = org_k.unwrap_or(*k);

    // Right side end point support
    if *x == t[t.len()-1] && i >= (t.len() - org_k - 1) {
        return 1.0_f64
    }

    // Recursion
    if *k == 1_usize {
        if t[i] <= *x && *x < t[i+1] {
            1.0_f64
        } else {
            0.0_f64
        }
    } else {
        let mut left: f64 = 0.0_f64;
        let mut right: f64 = 0.0_f64;
        if t[i] != t[i + k -1] {
            left = (x - t[i]) / (t[i+k-1] - t[i]) * bsplev_single_f64(x, i, &(k-1), t, None);
        }
        if t[i+1] != t[i+k] {
            right = (t[i+k] - x) / (t[i+k] - t[i+1]) * bsplev_single_f64(x, i+1, &(k-1), t, None);
        }
        left + right
    }
}

fn bspldnev_single_f64(x: &f64, i: usize, k: &usize, t: &Vec<f64>, m: usize, org_k: Option<usize>) -> f64{
    if m == 0 {
        return bsplev_single_f64(x, i, k, t, None)
    } else if *k == 1 || m >= *k {
        return 0.0_f64
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

fn ppev_single<T>(pps: &PPSpline<T>, x: &f64) -> T
where T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a T, Output=T>,
  for<'a> &'a T: Mul<&'a f64, Output=T> + Sub<T, Output=T> + Mul<&'a T, Output=T>,
  for<'a> &'a f64: Mul<&'a T, Output=T>
{
    let b: Array1<f64> = Array1::from_vec(
        (0..pps.n).map(|i| bsplev_single_f64(x, i, &pps.k, &pps.t, None)).collect()
    );
    match &pps.c {
        Some(c) => { fdmul11_(&b.view(), &c.view()) }
        None => { panic!("Must call csolve before attempting to evaluate spline.") }
    }
}

pub struct PPSpline<T>
// where T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a T, Output=T>,
//   for<'a> &'a T: Mul<&'a f64, Output=T> + Sub<T, Output=T> + Mul<&'a T, Output=T>,
//   for<'a> &'a f64: Mul<&'a T, Output=T>
{
    k: usize,
    t: Vec<f64>,
    c: Option<Array1<T>>,
    n: usize
}

impl<T> PPSpline<T>
where T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a T, Output=T>,
  for<'a> &'a T: Mul<&'a f64, Output=T> + Sub<T, Output=T> + Mul<&'a T, Output=T>,
  for<'a> &'a f64: Mul<&'a T, Output=T>
{
    pub fn new(k: usize, t: Vec<f64>) -> Self {
        let n = t.len() - k;
        // let c: Array1<T> = Array1::zeros(0);
        PPSpline {k, t, n, c: None}
    }

    pub fn ppev_single(&self, x: &f64) -> T {
        ppev_single(&self, &x)
    }

    pub fn ppdnev_single(&self, x: &f64, m: usize) -> T {
        let b: Array1<f64> = Array1::from_vec(
             (0..self.n).map(|i| bspldnev_single_f64(x, i, &self.k, &self.t, m, None)).collect()
         );
        match &self.c {
            Some(c) => { fdmul11_(&b.view(), &c.view()) }
            None => { panic!("Must call csolve before attempting to evaluate spline.") }
        }
    }

    pub fn csolve(&mut self, tau: &Vec<f64>, y: &Vec<T>, left_n: usize, right_n: usize, allow_lsq: bool)
    where T: PartialOrd + Signed + Clone + Sum + Zero + for<'a> Div<&'a f64, Output=T>,
       for<'a> &'a T: Mul<&'a f64, Output=T> + Sub<T, Output=T> + Mul<&'a f64, Output=T>,
       for<'a> &'a f64: Mul<&'a T, Output=T>
    {
        if tau.len() != self.n && !(allow_lsq && tau.len() > self.n){
            panic!("`csolve` cannot complete if length of `tau` < n or `allow_lsq` is false.")
        }
        if tau.len() != y.len() {
            panic!("`tau` and `y` must have the same length.")
        }
        let b: Array2<f64> = self.bsplmatrix(&tau, left_n, right_n);
        let ya: Array1<T> = Array1::from_vec(y.clone());
        let c: Array1<T> = fdsolve(&b.view(), &ya.view(), allow_lsq);
        self.c = Some(c);
    }

    pub fn bsplmatrix(&self, tau: &Vec<f64>, left_n: usize, right_n: usize) -> Array2<f64> {
        let mut b = Array2::zeros((tau.len(), self.n));
        for i in 0..self.n {
            b[[0, i]] = bspldnev_single_f64(&tau[0], i, &self.k, &self.t, left_n, None);
            b[[tau.len()-1, i]] = bspldnev_single_f64(&tau[tau.len()-1], i, &self.k, &self.t, right_n, None);
            for j in 1..(tau.len()-1) {
                b[[j, i]] = bsplev_single_f64(&tau[j], i, &self.k, &self.t,  None )
            }
        }
        b
    }
}


// UNIT TESTS

//

//

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn is_close(a: &f64, b: &f64, abs_tol: Option<f64>) -> bool {
        // used rather than equality for float numbers
        (a-b).abs() < abs_tol.unwrap_or(1e-8)
    }

    #[test]
    fn bsplev_single_f64_() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8).map(|i| bsplev_single_f64(&x, i as usize, &k, &t, None)).collect();
        let expected: Vec<f64> = Vec::from(&[0.125, 0.375, 0.375, 0.125, 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bsplev_single_f64_right() {
        let x: f64 = 4.0_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8).map(|i| bsplev_single_f64(&x, i as usize, &k, &t, None)).collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 0., 0., 1.0]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_f64_() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8).map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 1_usize, None)).collect();
        let expected: Vec<f64> = Vec::from(&[-0.75, -0.75, 0.75, 0.75, 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_f64_right() {
        let x: f64 = 4.0_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8).map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 1_usize, None)).collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 0., -3., 3.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_shortcut() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8).map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 6_usize, None)).collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_m() {
        let x: f64 = 4.0_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8).map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 2_usize, None)).collect();
        let expected: Vec<f64> = Vec::from(&[0., 0., 0., 0., 0., 3., -9., 6.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn bspldnev_single_m_zero() {
        let x: f64 = 1.5_f64;
        let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let k: usize = 4;
        let result: Vec<f64> = (0..8).map(|i| bspldnev_single_f64(&x, i as usize, &k, &t, 0_usize, None)).collect();
        let expected: Vec<f64> = Vec::from(&[0.125, 0.375, 0.375, 0.125, 0., 0., 0., 0.]);
        assert_eq!(result, expected)
    }

    #[test]
    fn ppspline_new() {
        let pps: PPSpline<f64> = PPSpline::new(4, vec![1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
    }

    #[test]
    fn ppspline_bsplmatrix() {
        let pps: PPSpline<f64> = PPSpline::new(4, vec![1., 1., 1., 1., 2., 3., 3., 3., 3.]);
        let result = pps.bsplmatrix(&vec![1., 1., 2., 3., 3.], 2_usize, 2_usize);
        let expected: Array2<f64> = arr2(&[
            [6., -9., 3., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0.25, 0.5, 0.25, 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 3., -9., 6.]
        ]);
        assert_eq!(result, expected)
    }

    #[test]
    fn csolve_() {
        let t = vec![0., 0., 0., 0., 4., 4., 4., 4.];
        let tau = vec![0., 1., 3., 4.,];
        let val = vec![0., 0., 2., 2.];
        let mut pps = PPSpline::new(4, t);
        pps.csolve(&tau, &val, 0, 0, false);
        let expected = vec![0., -1.11111111, 3.111111111111, 2.0];
        let v: Vec<bool> = pps.c.expect("csolve")
                                .into_raw_vec().iter()
                                .zip(expected.iter())
                                .map(|(x,y)| is_close(&x, &y, None))
                                .collect();

        assert!(v.iter().all(|x| *x));
    }

    #[test]
    fn ppev_single_(){
        let t = vec![1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.];
        let mut pps = PPSpline::new(4, t);
        pps.c = Some(arr1(&[1., 2., -1., 2., 1., 1., 2., 2.]));
        let r1 = pps.ppev_single(&1.1);
        assert!(is_close(&r1, &1.19, None));
        let r2 = pps.ppev_single(&1.8);
        assert!(is_close(&r2, &0.84, None));
        let r3 = pps.ppev_single(&2.8);
        assert!(is_close(&r3, &1.136, None));
    }

}