
use crate::splines::{PPSpline, bspldnev_single_f64, bsplev_single_dual, bsplev_single_f64};
use crate::dual::{Dual, Gradient1};
use ndarray::{arr1, arr2, Array2};
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
        .into_raw_vec_and_offset()
        .0
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
        .into_raw_vec_and_offset()
        .0
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
    assert!(pp1 == pp2);
    let pp3 = PPSpline::new(2, vec![1., 1., 2., 2.], Some(vec![1.5, 0.2]));
    let pp4 = PPSpline::new(2, vec![1., 1., 2., 2.], Some(vec![1.5, 0.2]));
    assert!(pp3 == pp4);
    assert!(pp3 != pp2);
    assert!(pp1 != pp4);
}

#[test]
#[should_panic]
fn backwards_definition() {
    let _pp1 = PPSpline::<f64>::new(4, vec![3., 3., 3., 3., 2., 1., 1., 1., 1.], None);
}
