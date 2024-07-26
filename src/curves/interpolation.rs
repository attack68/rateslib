
use crate::dual::dual::FieldOps;
use std::ops::Mul;

pub enum LocalInterpolation {
    LogLinear,
    Linear,
    LinearIndex,
    LinearZeroRate,
    FlatForward,
    FlatBackward,
}

pub(crate) fn linear_interp<T, U>(x1: &T, y1: &U, x2: &T, y2: &U, x: &T) -> U
where
    for<'a> &'a T: FieldOps<T>,
    for<'a> &'a U: FieldOps<U>,
    U: Mul<T, Output = U>,
{
    y1 + &((y2 - y1) * (&(x - x1) / &(x2 - x1)))
}

pub fn index_left<T>(list_input: &[T], value: &T, left_count: Option<usize>) -> usize
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
    use crate::dual::dual::Dual;

    #[test]
    fn index_left_() {
        let a = [1.2, 1.7, 1.9, 2.8];
        let result = index_left(&a, &1.8, None);
        assert_eq!(result, 1_usize);
    }

    #[test]
    fn test_linear_interp() {
        // float linear_interp
        let result = linear_interp(&1.0, &10.0, &2.0, &30.0, &1.5);
        assert_eq!(result, 20.0_f64);

        // Dual linear_interp
        let result = linear_interp(
            &1.0, &Dual::new(10.0, vec!["x".to_string()]), &2.0, &Dual::new(30.0, vec!["y".to_string()]), &1.5
        );
        assert_eq!(result, Dual::try_new(20.0, vec!["x".to_string(), "y".to_string()], vec![0.5, 0.5]).unwrap());
    }

}
