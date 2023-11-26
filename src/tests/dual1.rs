#[cfg(test)]
mod tests {

    use crate::dual_rebuild::dual1::Dual1;

    #[test]
    fn new_dual1() {
        let result = Dual1::new(2.3, Vec::from([String::from("a")]), Vec::new());
    }

    #[test]
    #[should_panic]
    fn new_dual1_panic() {
        let result = Dual1::new(
            2.3, Vec::from([String::from("a"), String::from("b")]), Vec::from([1.0])
        );
    }
}