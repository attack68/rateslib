

fn bsplev_single(x: &f64, i: &usize, k: &usize, t: &Vec<f64>, org_k: Option<usize>) -> f64:
    let org_k: usize = org_k.unwrap_or(k);

    // Right side end point support
    if x == t[t.len()-1] and i >= (len(t) - org_k - 1) {
        return 1.0_f64
    }

    // Recursion
    if k == 1 {
        if t[i] <= x && x < t[i+1] {
            return 1.0_f64
        } else {
            return 0.0_f64
        }
    } else {
        let mut left: f64 = 0.0_f64;
        let mut right: f64 = 0.0_f64;
        if t[i] != t[i + k -1] {
            left = (x - t[i]) / (t[i+k-1] - t[i]) * bsplev_single(x, i, k-1, t, None);
        } else if t[i+1] != t[i+k] {
            right = (t[i+k] - x) / (t[i+k] - t[i+1]) * bsplev_single(x, i+1, k-1, t, None);
        }
        return left + right
    }
}

struct PPSpline {
    k: usize,
    t: Vec<f64>,
    c: Vec<f64>,
    n: usize
}

impl PPSpline {
     fn new(k: usize, t: Vec<f64>) {
        PPSpline {k, t, c: Vec::new(), n: t.len() - k}
     }

     fn ppev_single()
}