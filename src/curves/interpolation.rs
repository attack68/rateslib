
pub fn index_left<T>(list_input: &[T], value: &T, left_count: Option<usize>) -> usize
where for<'a> &'a T: PartialOrd + PartialEq
{
    let lc = left_count.unwrap_or(0_usize);
    let n = list_input.len();
    match n {
        1 => panic!("`index_left` designed for intervals. Cannot index sequence of length 1."),
        2 => lc,
        _ => {
            let split = (n - 1_usize) / 2_usize;  // this will take the floor of result
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

    #[test]
    fn index_left_() {
        let a = [1.2, 1.7, 1.9, 2.8];
        let result = index_left(&a, &1.8, None);
        assert_eq!(result, 1_usize);
    }

}

