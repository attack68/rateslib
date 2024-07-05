use crate::dual::dual::{Dual, Dual2};
use std::cmp::Ordering;

/// Compares `Dual` by `real` component only.
impl PartialOrd<Dual> for Dual {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl PartialOrd<f64> for Dual {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.real.partial_cmp(other)
    }
}

impl PartialOrd<f64> for Dual2 {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.real.partial_cmp(other)
    }
}

impl PartialOrd<Dual2> for Dual2 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl PartialOrd<Dual> for f64 {
    fn partial_cmp(&self, other: &Dual) -> Option<Ordering> {
        self.partial_cmp(&other.real)
    }
}

impl PartialOrd<Dual2> for f64 {
    fn partial_cmp(&self, other: &Dual2) -> Option<Ordering> {
        self.partial_cmp(&other.real)
    }
}
