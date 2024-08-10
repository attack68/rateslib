use crate::curves::{Curve, CurveInterpolation};
use crate::json::JSON;
use serde::{Deserialize, Serialize};

impl<T: CurveInterpolation + for<'a> Deserialize<'a> + Serialize> JSON for Curve<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;
    use crate::curves::{LogLinearInterpolator, Nodes};
    use indexmap::IndexMap;

    fn curve_fixture() -> Curve<LogLinearInterpolator> {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        let interpolator = LogLinearInterpolator::new();
        Curve::try_new(nodes, interpolator, "crv").unwrap()
    }

    #[test]
    fn test_curve_json() {
        let curve = curve_fixture();
        let js = curve.to_json().unwrap();
        println!("{}", js);
        let curve2 = Curve::from_json(&js).unwrap();
        assert_eq!(curve, curve2);
    }
}
