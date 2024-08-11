use crate::curves::curve_py::PyCurve;
use crate::curves::{Curve, CurveInterpolation};
use crate::json::JSON;
use serde::{Deserialize, Serialize};

impl<T: CurveInterpolation + for<'a> Deserialize<'a> + Serialize> JSON for Curve<T> {}

impl JSON for PyCurve {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::ndt;
    use crate::curves::curve_py::CurveInterpolator;
    use crate::curves::{
        LinearInterpolator, LinearZeroRateInterpolator, LogLinearInterpolator, Nodes,
    };
    use indexmap::IndexMap;

    fn curve_fixture<T: CurveInterpolation>(interpolator: T) -> Curve<T> {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        Curve::try_new(nodes, interpolator, "crv").unwrap()
    }

    #[test]
    fn test_curve_json_loglinear() {
        let interpolator = LogLinearInterpolator::new();
        let curve = curve_fixture(interpolator);
        let js = curve.to_json().unwrap();
        println!("{}", js);
        let curve2 = Curve::from_json(&js).unwrap();
        assert_eq!(curve, curve2);
    }

    #[test]
    fn test_curve_json_linear() {
        let interpolator = LinearInterpolator::new();
        let curve = curve_fixture(interpolator);
        let js = curve.to_json().unwrap();
        println!("{}", js);
        let curve2 = Curve::from_json(&js).unwrap();
        assert_eq!(curve, curve2);
    }

    #[test]
    fn test_curve_json_linear_zero_rate() {
        let interpolator = LinearZeroRateInterpolator::new();
        let curve = curve_fixture(interpolator);
        let js = curve.to_json().unwrap();
        println!("{}", js);
        let curve2 = Curve::from_json(&js).unwrap();
        assert_eq!(curve, curve2);
    }

    #[test]
    fn test_curve_json_py_enum() {
        let interpolator = CurveInterpolator::Linear(LinearInterpolator::new());
        let curve = curve_fixture(interpolator);
        let js = curve.to_json().unwrap();
        println!("{}", js);
        let curve2 = Curve::from_json(&js).unwrap();
        assert_eq!(curve, curve2);
    }
}
