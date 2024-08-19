use crate::calendars::DateRoll;
use crate::curves::curve_py::Curve;
use crate::curves::{CurveDF, CurveInterpolation};
use crate::json::JSON;
use serde::{Deserialize, Serialize};

impl<T, U> JSON for CurveDF<T, U>
where
    T: CurveInterpolation + for<'a> Deserialize<'a> + Serialize,
    U: DateRoll + for<'a> Deserialize<'a> + Serialize,
{
}

impl JSON for Curve {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::{ndt, Convention, Modifier, NamedCal};
    use crate::curves::curve_py::CurveInterpolator;
    use crate::curves::{
        FlatBackwardInterpolator, FlatForwardInterpolator, LinearInterpolator,
        LinearZeroRateInterpolator, LogLinearInterpolator, Nodes,
    };
    use indexmap::IndexMap;

    fn curve_fixture<T: CurveInterpolation>(interpolator: T) -> CurveDF<T, NamedCal> {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        let convention = Convention::Act360;
        let modifier = Modifier::ModF;
        let cal = NamedCal::try_new("all").unwrap();
        CurveDF::try_new(nodes, interpolator, "crv", convention, modifier, None, cal).unwrap()
    }

    #[test]
    fn test_curve_json_all_interpolators() {
        macro_rules! test_interpolator {
            ($Variant: ident) => {
                let interpolator = $Variant::new();
                let curve = curve_fixture(interpolator);
                let js = curve.to_json().unwrap();
                let curve2 = CurveDF::from_json(&js).unwrap();
                assert_eq!(curve, curve2);
            };
        }

        test_interpolator!(FlatBackwardInterpolator);
        test_interpolator!(FlatForwardInterpolator);
        test_interpolator!(LogLinearInterpolator);
        test_interpolator!(LinearInterpolator);
        test_interpolator!(LinearZeroRateInterpolator);
    }

    #[test]
    fn test_curve_json_py_enum() {
        let interpolator = CurveInterpolator::Linear(LinearInterpolator::new());
        let curve = curve_fixture(interpolator);
        let js = curve.to_json().unwrap();
        println!("{}", js);
        let curve2 = CurveDF::from_json(&js).unwrap();
        assert_eq!(curve, curve2);
    }
}
