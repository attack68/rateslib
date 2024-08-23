use crate::calendars::DateRoll;
use crate::calendars::{Convention, Modifier};
use crate::curves::interpolation::utils::index_left;
use crate::curves::nodes::{Nodes, NodesTimestamp};
use crate::dual::{get_variable_tags, ADOrder, Dual, Dual2, Number};
use chrono::NaiveDateTime;
use indexmap::IndexMap;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;

/// Default struct for storing datetime indexed discount factors (DFs).
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct CurveDF<T: CurveInterpolation, U: DateRoll> {
    pub(crate) nodes: NodesTimestamp,
    pub(crate) interpolator: T,
    pub(crate) id: String,
    pub(crate) convention: Convention,
    pub(crate) modifier: Modifier,
    pub(crate) index_base: Option<f64>,
    pub(crate) calendar: U,
}

/// Assigns methods for returning values from datetime indexed Curves.
pub trait CurveInterpolation {
    /// Get a value from the curve's `Nodes` expressed in its input form, i.e. discount factor or value.
    fn interpolated_value(&self, nodes: &NodesTimestamp, date: &NaiveDateTime) -> Number;

    /// Get the left side node key index of the given datetime
    fn node_index(&self, nodes: &NodesTimestamp, date_timestamp: i64) -> usize {
        // let timestamp = date.and_utc().timestamp();
        index_left(&nodes.keys(), &date_timestamp, None)
    }
}

impl<T: CurveInterpolation, U: DateRoll> CurveDF<T, U> {
    pub fn try_new(
        nodes: Nodes,
        interpolator: T,
        id: &str,
        convention: Convention,
        modifier: Modifier,
        index_base: Option<f64>,
        calendar: U,
    ) -> Result<Self, PyErr> {
        let mut nodes = NodesTimestamp::from(nodes);
        nodes.sort_keys();
        Ok(Self {
            nodes,
            interpolator,
            id: id.to_string(),
            convention,
            modifier,
            index_base,
            calendar,
        })
    }

    /// Get the `ADOrder` of the `Curve`.
    pub fn ad(&self) -> ADOrder {
        match self.nodes {
            NodesTimestamp::F64(_) => ADOrder::Zero,
            NodesTimestamp::Dual(_) => ADOrder::One,
            NodesTimestamp::Dual2(_) => ADOrder::Two,
        }
    }

    pub fn interpolated_value(&self, date: &NaiveDateTime) -> Number {
        self.interpolator.interpolated_value(&self.nodes, date)
    }

    pub fn node_index(&self, date_timestamp: i64) -> usize {
        self.interpolator.node_index(&self.nodes, date_timestamp)
    }

    pub fn set_ad_order(&mut self, ad: ADOrder) -> Result<(), PyErr> {
        let vars: Vec<String> = get_variable_tags(&self.id, self.nodes.keys().len());
        match (ad, &self.nodes) {
            (ADOrder::Zero, NodesTimestamp::F64(_))
            | (ADOrder::One, NodesTimestamp::Dual(_))
            | (ADOrder::Two, NodesTimestamp::Dual2(_)) => {
                // leave unchanged.
                Ok(())
            }
            (ADOrder::One, NodesTimestamp::F64(i)) => {
                // rebuild the derivatives
                self.nodes = NodesTimestamp::Dual(IndexMap::from_iter(
                    i.into_iter()
                        .enumerate()
                        .map(|(i, (k, v))| (*k, Dual::new(*v, vec![vars[i].clone()]))),
                ));
                Ok(())
            }
            (ADOrder::Two, NodesTimestamp::F64(i)) => {
                // rebuild the derivatives
                self.nodes = NodesTimestamp::Dual2(IndexMap::from_iter(
                    i.into_iter()
                        .enumerate()
                        .map(|(i, (k, v))| (*k, Dual2::new(*v, vec![vars[i].clone()]))),
                ));
                Ok(())
            }
            (ADOrder::One, NodesTimestamp::Dual2(i)) => {
                self.nodes = NodesTimestamp::Dual(IndexMap::from_iter(
                    i.into_iter().map(|(k, v)| (*k, Dual::from(v))),
                ));
                Ok(())
            }
            (ADOrder::Zero, NodesTimestamp::Dual(i)) => {
                // covert dual into f64
                self.nodes = NodesTimestamp::F64(IndexMap::from_iter(
                    i.into_iter().map(|(k, v)| (*k, f64::from(v))),
                ));
                Ok(())
            }
            (ADOrder::Zero, NodesTimestamp::Dual2(i)) => {
                // covert dual into f64
                self.nodes = NodesTimestamp::F64(IndexMap::from_iter(
                    i.into_iter().map(|(k, v)| (*k, f64::from(v))),
                ));
                Ok(())
            }
            (ADOrder::Two, NodesTimestamp::Dual(i)) => {
                // rebuild derivatives
                self.nodes = NodesTimestamp::Dual2(IndexMap::from_iter(
                    i.into_iter().map(|(k, v)| (*k, Dual2::from(v))),
                ));
                Ok(())
            }
        }
    }

    pub fn index_value(&self, date: &NaiveDateTime) -> Result<Number, PyErr> {
        match self.index_base {
            None => Err(PyValueError::new_err("Can only calculate `index_value` for a Curve which has been initialised with `index_base`.")),
            Some(ib) => {
                if date.and_utc().timestamp() < self.nodes.first_key() {
                    Ok(Number::F64(0.0))
                } else {
                    Ok(Number::F64(ib) / self.interpolated_value(date))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::{ndt, Convention, NamedCal};
    use crate::curves::LogLinearInterpolator;
    use indexmap::IndexMap;

    fn curve_fixture() -> CurveDF<LogLinearInterpolator, NamedCal> {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        let interpolator = LogLinearInterpolator::new();
        let convention = Convention::Act360;
        let modifier = Modifier::ModF;
        let cal = NamedCal::try_new("all").unwrap();
        CurveDF::try_new(nodes, interpolator, "crv", convention, modifier, None, cal).unwrap()
    }

    fn index_curve_fixture() -> CurveDF<LogLinearInterpolator, NamedCal> {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        let interpolator = LogLinearInterpolator::new();
        let convention = Convention::Act360;
        let modifier = Modifier::ModF;
        let cal = NamedCal::try_new("all").unwrap();
        CurveDF::try_new(
            nodes,
            interpolator,
            "crv",
            convention,
            modifier,
            Some(100.0),
            cal,
        )
        .unwrap()
    }

    fn curve_dual_fixture() -> CurveDF<LogLinearInterpolator, NamedCal> {
        let nodes = Nodes::Dual(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), Dual::new(1.0, vec!["x".to_string()])),
            (ndt(2001, 1, 1), Dual::new(0.99, vec!["y".to_string()])),
            (ndt(2002, 1, 1), Dual::new(0.98, vec!["z".to_string()])),
        ]));
        let interpolator = LogLinearInterpolator::new();
        let convention = Convention::Act360;
        let modifier = Modifier::ModF;
        let cal = NamedCal::try_new("all").unwrap();
        CurveDF::try_new(nodes, interpolator, "crv", convention, modifier, None, cal).unwrap()
    }

    #[test]
    fn test_get_index() {
        let c = curve_fixture();
        let result = c.node_index(ndt(2001, 7, 30).and_utc().timestamp());
        assert_eq!(result, 1_usize)
    }

    #[test]
    fn test_get_value() {
        let c = curve_fixture();
        let result = c.interpolated_value(&ndt(2000, 7, 1));
        assert_eq!(result, Number::F64(0.9950147597711371))
    }

    fn nodes_timestamp_fixture() -> NodesTimestamp {
        let nodes = Nodes::F64(IndexMap::from_iter(vec![
            (ndt(2000, 1, 1), 1.0_f64),
            (ndt(2001, 1, 1), 0.99_f64),
            (ndt(2002, 1, 1), 0.98_f64),
        ]));
        NodesTimestamp::from(nodes)
    }

    #[test]
    fn test_log_linear() {
        let nts = nodes_timestamp_fixture();
        let ll = LogLinearInterpolator::new();
        let result = ll.interpolated_value(&nts, &ndt(2000, 7, 1));
        // expected = exp(0 + (182 / 366) * (ln(0.99) - ln(1.0)) = 0.995015
        assert_eq!(result, Number::F64(0.9950147597711371));
    }

    #[test]
    fn test_set_order() {
        // converts the input f64 nodes to dual with ordered variables tagged by id
        let mut curve = curve_fixture();
        let _ = curve.set_ad_order(ADOrder::One);
        let result = curve.interpolated_value(&ndt(2001, 1, 1));
        assert_eq!(
            result,
            Number::Dual(Dual::new(0.99, vec!["crv1".to_string()]))
        );
    }

    #[test]
    fn test_set_order_no_change() {
        // asserts no change in values when AD order remains same
        let mut curve = curve_dual_fixture();
        let _ = curve.set_ad_order(ADOrder::One);
        let result = curve.interpolated_value(&ndt(2001, 1, 1));
        assert_eq!(result, Number::Dual(Dual::new(0.99, vec!["y".to_string()])));
    }

    #[test]
    fn test_set_order_vars_remain() {
        // asserts no change in variables transitioning ADone to ADtwo
        let mut curve = curve_dual_fixture();
        let _ = curve.set_ad_order(ADOrder::Two);
        let result = curve.interpolated_value(&ndt(2001, 1, 1));
        assert_eq!(
            result,
            Number::Dual2(Dual2::new(0.99, vec!["y".to_string()]))
        );
    }

    #[test]
    fn test_index_value() {
        let index_curve = index_curve_fixture();
        let result = index_curve.index_value(&ndt(2001, 1, 1)).unwrap();
        assert_eq!(result, Number::F64(100.0 / 0.99))
    }

    #[test]
    fn test_index_value_prior_to_first() {
        let index_curve = index_curve_fixture();
        let result = index_curve.index_value(&ndt(1980, 1, 1)).unwrap();
        assert_eq!(result, Number::F64(0.0))
    }
}
