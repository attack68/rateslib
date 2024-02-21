use ndarray::Array1;

fn bsplev_single_f64(x: &f64, i: usize, k: usize, t: &Vec<f64>, org_k: Option<usize>) -> f64 {
    let org_k: usize = org_k.unwrap_or(k);

    // Right side end point support
    if *x == t[t.len()-1] && i >= (t.len() - org_k - 1) {
        return 1.0_f64
    }

    // Recursion
    if k == 1_usize {
        if t[i] <= *x && *x < t[i+1] {
            1.0_f64
        } else {
            0.0_f64
        }
    } else {
        let mut left: f64 = 0.0_f64;
        let mut right: f64 = 0.0_f64;
        if t[i] != t[i + k -1] {
            left = (x - t[i]) / (t[i+k-1] - t[i]) * bsplev_single_f64(x, i, k-1, t, None);
        }
        if t[i+1] != t[i+k] {
            right = (t[i+k] - x) / (t[i+k] - t[i+1]) * bsplev_single_f64(x, i+1, k-1, t, None);
        }
        left + right
    }
}

fn bspldnev_single_f64(x: &f64, i: usize, k: usize, t: &Vec<f64>, m: usize, org_k: Option<usize>) -> f64{
    if m == 0 {
        return bsplev_single_f64(x, i, k, t, None)
    } else if k == 1 || m >= k {
        return 0.0_f64
    }

    let org_k: usize = org_k.unwrap_or(k);
    let mut r: f64 = 0.0;
    let div1: f64 = t[i + k - 1] - t[i];
    let div2: f64 = t[i + k] - t[i + 1];

    if m == 1 {
        if div1 != 0_f64 {
            r += bsplev_single_f64(x, i, k - 1, t, Some(org_k)) / div1;
        }
        if div2 != 0_f64 {
            r -= bsplev_single_f64(x, i + 1, k - 1, t, Some(org_k)) / div2;
        }
        r *= (k - 1) as f64;
    } else {
        if div1 != 0_f64 {
            r += bspldnev_single_f64(x, i, k - 1, t, m - 1, Some(org_k)) / div1;
        }
        if div2 != 0_f64 {
            r -= bspldnev_single_f64(x, i + 1, k - 1, t, m - 1, Some(org_k)) / div2;
        }
        r *= (k - 1) as f64
    }
    r
}

pub struct PPSpline {
    k: usize,
    t: Vec<f64>,
    c: Array1<f64>,
    n: usize
}

impl PPSpline {
     pub fn new(k: usize, t: Vec<f64>) -> Self {
        let n = t.len() - k;
        PPSpline {k, t, n, c: Array1::zeros(0)}
     }

     pub fn ppev_single(&self, x: &f64) -> f64 {
         let b: Array1<f64> = Array1::from_vec(
             (0..self.n).map(|i| bsplev_single_f64(x, i, self.k, &self.t, None)).collect()
         );
         if self.c.len() != b.len() {
             panic!("Must call csolve before attempting to evaluate spline.")
         }
         return b.dot(&self.c)
     }
}



// UNIT TESTS

//

//

#[test]
fn bsplev_single_f64_(){
    let x: f64 = 1.5_f64;
    let t: Vec<f64> = Vec::from(&[1., 1., 1., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
    let k: usize = 4;
    let result: Vec<f64> = (0..8).map(|i| bsplev_single_f64(&x, i as usize, k, &t, None)).collect();
    let expected: Vec<f64> = Vec::from(&[0.125, 0.375, 0.375, 0.125, 0., 0., 0., 0.]);
    assert_eq!(result, expected)
}
