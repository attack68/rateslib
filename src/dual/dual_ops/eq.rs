use crate::dual::dual::{Dual, Dual2, Vars, VarsRelationship};

/// Measures value equivalence of `Dual`.
///
/// Returns `true` if:
///
/// - `real` components are equal: `lhs.real == rhs.real`.
/// - `dual` components are equal after aligning `vars`.
impl PartialEq<Dual> for Dual {
    fn eq(&self, other: &Dual) -> bool {
        if self.real != other.real {
            false
        } else {
            let state = self.vars_cmp(other.vars());
            match state {
                VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
                    self.dual.iter().eq(other.dual.iter())
                }
                _ => {
                    let (x, y) = self.to_union_vars(other, Some(state));
                    x.dual.iter().eq(y.dual.iter())
                }
            }
        }
    }
}

impl PartialEq<f64> for Dual {
    fn eq(&self, other: &f64) -> bool {
        Dual::new(*other, [].to_vec()) == *self
    }
}

impl PartialEq<f64> for Dual2 {
    fn eq(&self, other: &f64) -> bool {
        Dual2::new(*other, Vec::new()) == *self
    }
}

impl PartialEq<Dual> for f64 {
    fn eq(&self, other: &Dual) -> bool {
        Dual::new(*self, [].to_vec()) == *other
    }
}

impl PartialEq<Dual2> for f64 {
    fn eq(&self, other: &Dual2) -> bool {
        Dual2::new(*self, Vec::new()) == *other
    }
}

impl PartialEq<Dual2> for Dual2 {
    fn eq(&self, other: &Dual2) -> bool {
        if self.real != other.real {
            false
        } else {
            let state = self.vars_cmp(other.vars());
            match state {
                VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
                    self.dual.iter().eq(other.dual.iter())
                        && self.dual2.iter().eq(other.dual2.iter())
                }
                _ => {
                    let (x, y) = self.to_union_vars(other, Some(state));
                    x.dual.iter().eq(y.dual.iter()) && x.dual2.iter().eq(y.dual2.iter())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_ne() {
        // Dual with vars - f64
        assert!(Dual::new(0.0, Vec::from([String::from("a")])) != 0.0);
        // Dual with no vars - f64 (+reverse)
        assert!(Dual::new(2.0, Vec::new()) == 2.0);
        assert!(2.0 == Dual::new(2.0, Vec::new()));
        // Dual - Dual (various real, vars, gradient mismatch)
        let d = Dual::try_new(2.0, Vec::from([String::from("a")]), Vec::from([2.3])).unwrap();
        assert!(d == Dual::try_new(2.0, Vec::from([String::from("a")]), Vec::from([2.3])).unwrap());
        assert!(d != Dual::try_new(2.0, Vec::from([String::from("b")]), Vec::from([2.3])).unwrap());
        assert!(d != Dual::try_new(3.0, Vec::from([String::from("a")]), Vec::from([2.3])).unwrap());
        assert!(d != Dual::try_new(2.0, Vec::from([String::from("a")]), Vec::from([1.3])).unwrap());
        // Dual - Dual (missing Vars are zero and upcasted)
        assert!(
            d == Dual::try_new(
                2.0,
                Vec::from([String::from("a"), String::from("b")]),
                Vec::from([2.3, 0.0])
            )
                .unwrap()
        );
    }

    #[test]
    fn eq_ne2() {
        // Dual with vars - f64
        assert!(Dual2::new(0.0, Vec::from([String::from("a")])) != 0.0);
        // Dual with no vars - f64 (+reverse)
        assert!(Dual2::new(2.0, Vec::new()) == 2.0);
        assert!(2.0 == Dual2::new(2.0, Vec::new()));
        // Dual - Dual (various real, vars, gradient mismatch)
        let d = Dual2::try_new(
            2.0,
            Vec::from([String::from("a")]),
            Vec::from([2.3]),
            Vec::new(),
        )
            .unwrap();
        assert!(
            d == Dual2::try_new(
                2.0,
                Vec::from([String::from("a")]),
                Vec::from([2.3]),
                Vec::new()
            )
                .unwrap()
        );
        assert!(
            d != Dual2::try_new(
                2.0,
                Vec::from([String::from("b")]),
                Vec::from([2.3]),
                Vec::new()
            )
                .unwrap()
        );
        assert!(
            d != Dual2::try_new(
                3.0,
                Vec::from([String::from("a")]),
                Vec::from([2.3]),
                Vec::new()
            )
                .unwrap()
        );
        assert!(
            d != Dual2::try_new(
                2.0,
                Vec::from([String::from("a")]),
                Vec::from([1.3]),
                Vec::new()
            )
                .unwrap()
        );
        // Dual - Dual (missing Vars are zero and upcasted)
        assert!(
            d == Dual2::try_new(
                2.0,
                Vec::from([String::from("a"), String::from("b")]),
                Vec::from([2.3, 0.0]),
                Vec::new()
            )
                .unwrap()
        );
    }

}
