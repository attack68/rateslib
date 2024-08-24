use crate::dual::{MathFuncs, NumberOps};
use std::{
    cmp::{PartialEq, PartialOrd},
    ops::{Mul, Sub},
};

// pub(crate) fn linear_interp<T, U>(x1: &T, y1: &U, x2: &T, y2: &U, x: &T) -> U
// where
//     for<'a> &'a T: NumberOps<T>,
//     for<'a> &'a U: NumberOps<U>,
//     U: Mul<T, Output = U>,
// {
//     y1 + &((y2 - y1) * (&(x - x1) / &(x2 - x1)))
// }

/// Calculate the linear interpolation between two coordinates.
pub(crate) fn linear_interp<T>(x1: f64, y1: &T, x2: f64, y2: &T, x: f64) -> T
where
    for<'a> &'a T: NumberOps<T>,
    T: Mul<f64, Output = T>,
{
    y1 + &((y2 - y1) * ((x - x1) / (x2 - x1)))
}

/// Calculate the log-linear interpolation between two coordinates.
pub(crate) fn log_linear_interp<T>(x1: f64, y1: &T, x2: f64, y2: &T, x: f64) -> T
where
    for<'a> &'a T: NumberOps<T>,
    T: Mul<f64, Output = T> + MathFuncs,
{
    let (y1, y2) = (y1.log(), y2.log());
    let y = linear_interp(x1, &y1, x2, &y2, x);
    y.exp()
}

/// Calculate the linear zero rate interpolation between two coordinates.
pub(crate) fn linear_zero_interp<T>(x0: f64, x1: f64, y1: &T, x2: f64, y2: &T, x: f64) -> T
where
    for<'a> &'a T: NumberOps<T>,
    T: Mul<f64, Output = T> + MathFuncs + Sub + Clone,
{
    let t1: f64 = x1 - x0;
    let t2: f64 = x2 - x0;
    let t: f64 = x - x0;
    let r2: T = y2.log() * (-1_f64 / t2);
    let r: T = if t1 == 0.0_f64 {
        r2.clone() // Flat forward zero rate in first interval
    } else {
        let r1: T = y1.log() * (-1_f64 / t1);
        &r1 + &((&r2 - &r1) * ((t - t1) / (t2 - t1)))
    };
    (r * -t).exp()
}

/// Calculate the left sided index for a given value in a sorted list.
/// `left_count` is used recursively; it should always be entered as None intially.
/// Examples
/// --------
/// If `list_input` is [1.2, 1.7, 1.9, 2.8];
///
/// - 0.5: returns 0 (extrapolated out of range)
/// - 1.5: returns 0 (within first interval)
/// - 1.7: returns 0 (closed right side of first interval)
/// - 1.71: returns 1 (within second interval)
/// - 2.8: returns 2 (closed right side of third interval)
/// - 3.5: returns 2 (extrapolated out of range)
pub(crate) fn index_left<T>(list_input: &[T], value: &T, left_count: Option<usize>) -> usize
where
    for<'a> &'a T: PartialOrd + PartialEq,
{
    let lc = left_count.unwrap_or(0_usize);
    let n = list_input.len();
    match n {
        1 => panic!("`index_left` designed for intervals. Cannot index sequence of length 1."),
        2 => lc,
        _ => {
            let split = (n - 1_usize) / 2_usize; // this will take the floor of result
            if n == 3 && value == &list_input[split] {
                lc
            } else if value <= &list_input[split] {
                index_left(&list_input[..=split], value, Some(lc))
            } else {
                index_left(&list_input[split..], value, Some(lc + split))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::Dual;

    #[test]
    fn index_left_() {
        let a = [1.2, 1.7, 1.9, 2.8];
        assert_eq!(index_left(&a, &0.5, None), 0_usize);
        assert_eq!(index_left(&a, &1.5, None), 0_usize);
        assert_eq!(index_left(&a, &1.7, None), 0_usize);
        assert_eq!(index_left(&a, &1.71, None), 1_usize);
        assert_eq!(index_left(&a, &2.8, None), 2_usize);
        assert_eq!(index_left(&a, &3.5, None), 2_usize);
    }

    #[test]
    fn test_linear_interp() {
        // float linear_interp
        let result = linear_interp(1.0, &10.0, 2.0, &30.0, 1.5);
        assert_eq!(result, 20.0_f64);

        // Dual linear_interp
        let result = linear_interp(
            1.0,
            &Dual::new(10.0, vec!["x".to_string()]),
            2.0,
            &Dual::new(30.0, vec!["y".to_string()]),
            1.5,
        );
        assert_eq!(
            result,
            Dual::try_new(20.0, vec!["x".to_string(), "y".to_string()], vec![0.5, 0.5]).unwrap()
        );
    }

    #[test]
    fn test_log_linear_interp() {
        // float linear_interp
        let result = log_linear_interp(1.0, &10.0, 2.0, &30.0, 1.5);
        let expected = (0.5 * 10.0.log() + 0.5 * 30.0.log()).exp();
        assert_eq!(result, expected);

        // Dual linear_interp
        let y1 = Dual::new(10.0, vec!["x".to_string()]);
        let y2 = Dual::new(30.0, vec!["y".to_string()]);
        let result = log_linear_interp(1.0, &y1, 2.0, &y2, 1.5);
        let expected = (0.5 * y1.log() + 0.5 * y2.log()).exp();
        assert_eq!(result, expected);
    }
}
