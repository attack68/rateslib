//! Create objects related to the management and valuation of monetary amounts in different
//! currencies, measured at different settlement dates in time.

use crate::dual::dual1::Dual;
use crate::dual::dual2::Dual2;
use crate::dual::dual_py::DualsOrF64;
use crate::dual::linalg::{argabsmax, douter11_, dsolve};
use chrono::prelude::*;
use indexmap::set::IndexSet;
use internment::Intern;
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis, arr2};
use num_traits::identities::One;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, PyErr};
use std::collections::HashSet;
use std::fmt;

/// Struct to define a currency.
#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ccy {
    pub(crate) name: Intern<String>,
}

impl Ccy {
    /// Constructs a new `Ccy`.
    ///
    /// Use **only** 3-ascii names. e.g. *"usd"*, aligned with ISO representation. `name` is converted
    /// to lowercase to promote performant equality between "USD" and "usd".
    ///
    /// Panics if `name` is not 3 bytes in length.
    pub fn try_new(name: &str) -> Result<Self, PyErr> {
        let ccy: String = name.to_string().to_lowercase();
        if ccy.len() != 3 {
            return Err(PyValueError::new_err(
                "`Ccy` must be 3 ascii character in length, e.g. 'usd'.",
            ));
        }
        Ok(Ccy {
            name: Intern::new(ccy),
        })
    }
}

/// Struct to define a currency cross.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FXPair(Ccy, Ccy);

impl FXPair {
    /// Constructs a new `FXCross`, as a combination of two distinct `Ccy`s.
    pub fn try_new(lhs: &str, rhs: &str) -> Result<Self, PyErr> {
        let lhs_ = Ccy::try_new(lhs)?;
        let rhs_ = Ccy::try_new(rhs)?;
        if lhs_ == rhs_ {
            return Err(PyValueError::new_err(
                "`FXPair` must be created from two distinct currencies, not same.",
            ));
        }
        Ok(FXPair(lhs_, rhs_))
    }
}

impl fmt::Display for FXPair {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.0.name, self.1.name)
    }
}

/// Struct to define an FXRate via FXCross, rate and settlement info.
#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone)]
pub struct FXRate {
    pub(crate) pair: FXPair,
    pub(crate) rate: DualsOrF64,
    pub(crate) settlement: Option<NaiveDateTime>,
    pub(crate) ad: u8,
}

impl FXRate {
    pub fn try_new(
        lhs: &str,
        rhs: &str,
        rate: DualsOrF64,
        settlement: Option<NaiveDateTime>,
    ) -> Result<Self, PyErr> {
        let ad = match rate {
            DualsOrF64::F64(_) => 0_u8,
            DualsOrF64::Dual(_) => 1_u8,
            DualsOrF64::Dual2(_) => 2_u8,
        };
        Ok(FXRate {
            pair: FXPair::try_new(lhs, rhs)?,
            rate,
            settlement,
            ad,
        })
    }
}

#[derive(Debug, Clone)]
pub enum FXVector {
    Dual(Array1<Dual>),
    Dual2(Array1<Dual2>),
}

// impl for FXVector {
//     fn get_index(&self, index: usize) -> DualsOrF64 {
//         match self {
//             FXVector::Dual(arr) => DualsOrF64::Dual(arr[index]),
//             FXVector::Dual2(arr) => DualsOrF64::Dual2(arr[index]),
//         }
//     }
// }

#[derive(Debug, Clone)]
pub enum FXArray {
    Dual(Array2<Dual>),
    Dual2(Array2<Dual2>),
}

/// Struct to define a global FX market with multiple currencies.
#[pyclass(module = "rateslib.rs")]
#[derive(Debug, Clone)]
pub struct FXRates {
    pub(crate) fx_rates: Vec<FXRate>,
    pub(crate) currencies: IndexSet<Ccy>,
    pub(crate) arr_dual: Array2<Dual>,
    pub(crate) arr_dual2: Option<Array2<Dual2>>,
    pub(crate) base: Ccy,
    pub(crate) ad: u8,
    // settlement : Option<NaiveDateTime>,
    // pairs : Vec<(Ccy, Ccy)>,
}

impl FXRates {
    pub fn try_new(
        fx_rates: Vec<FXRate>,
        settlement: NaiveDateTime,
        base: Option<Ccy>,
    ) -> Result<Self, PyErr> {
        // Validations:
        // 1. fx_rates is non-zero length
        // 2. currencies are not under or over overspecified
        // 3. settlement dates are all consistent.
        // 4. No Dual2 data types are provided as input

        // 1.
        if fx_rates.is_empty() {
            return Err(PyValueError::new_err(
                "`fx_rates` must contain at least on fx rate.",
            ));
        }

        let mut currencies: IndexSet<Ccy> = IndexSet::with_capacity(fx_rates.len() + 1_usize);
        for fxr in fx_rates.iter() {
            currencies.insert(fxr.pair.0);
            currencies.insert(fxr.pair.1);
        }
        let q = currencies.len();

        // 2.
        if q > (fx_rates.len() + 1) {
            return Err(PyValueError::new_err("`fx_rates` is underspecified."));
        } else if q < (fx_rates.len() + 1) {
            return Err(PyValueError::new_err("`fx_rates` is overspecified."));
        }

        // 3.
        if !(&fx_rates
            .iter()
            .all(|d| d.settlement.map_or(true, |v| v == settlement)))
        {
            return Err(PyValueError::new_err(
                "`fx_rates` must have consistent `settlement` dates across all rates.",
            ));
        }

        // aggregate information from FXPairs
        // let pairs: Vec<(Ccy, Ccy)> = fx_rates.iter().map(|fxp| fxp.pair).collect();
        //         let variables: Vec<String> = fx_rates.iter().map(
        //                 |fxp| "fx_".to_string() + &fxp.pair.0.to_string() + &fxp.pair.1.to_string()
        //             ).collect();

        let base = base.unwrap_or(currencies[0]);

        let (mut fx_array, mut edges) = FXRates::_populate_initial_arrays(&currencies, &fx_rates);
        FXRates::_populate_remaining_arrays(fx_array.view_mut(), edges.view_mut(), HashSet::new());

        // let fx_vector = fx_array.row(currencies.get_index_of(&base).unwrap());

        Ok(FXRates {
            fx_rates,
            arr_dual: fx_array,
            arr_dual2: None,
            currencies,
            base,
            ad: 1_u8,
        })
    }

    fn _populate_initial_arrays(
        currencies: &IndexSet<Ccy>,
        fx_rates: &Vec<FXRate>,
    ) -> (Array2<Dual>, Array2<i16>) {
        let mut fx_array: Array2<Dual> = Array2::eye(currencies.len());
        let mut edges: Array2<i16> = Array2::eye(currencies.len());
        for fxr in fx_rates.iter() {
            let row = currencies.get_index_of(&fxr.pair.0).unwrap();
            let col = currencies.get_index_of(&fxr.pair.1).unwrap();
            edges[[row, col]] = 1_i16;
            edges[[col, row]] = 1_i16;
            match &fxr.rate {
                DualsOrF64::F64(f) => {
                    fx_array[[row, col]] = Dual::new(*f, vec!["fx_".to_string() + &format!("{}", fxr.pair)]);
                    fx_array[[col, row]] = 1_f64 / &fx_array[[row, col]];
                },
                DualsOrF64::Dual(d) => {
                    fx_array[[row, col]] = d.clone();
                    fx_array[[col, row]] = 1_f64 / &fx_array[[row, col]];
                },
                DualsOrF64::Dual2(_) => panic!("cannot construct from dual2 rates")
            }
        }
        (fx_array, edges)
    }

    fn _populate_remaining_arrays(
        mut fx_array: ArrayViewMut2<Dual>,
        mut edges: ArrayViewMut2<i16>,
        mut prev_value: HashSet<usize>,
    ) -> Result<bool, PyErr> {
        if prev_value.len() == edges.len_of(Axis(0)) {
            return Err(PyValueError::new_err(
                "FX Array cannot be solved. There are degenerate FX rate pairs.\n\
                For example ('eurusd' + 'usdeur') or ('usdeur', 'eurjpy', 'usdjpy').",
            ));
        }
        if edges.sum() == (edges.len_of(Axis(0)) * edges.len_of(Axis(1))) as i16 {
            return Ok(true); // all edges and values have already been populated.
        }
        let mut row_edges = edges.sum_axis(Axis(1));


        let mut node: usize = edges.len_of(Axis(1)) + 1_usize;
        let mut combinations_: Vec<Vec<usize>> = Vec::new();
        let mut start_flag = true;
        while start_flag || prev_value.contains(&node) {
            start_flag = false;

            // find node with most outgoing edges
            node = argabsmax(row_edges.view());
            row_edges[node] = 0_i16;

            // filter the node indices of the directly linked nodes to node
            // let linked_nodes = &edges
            //     .row(node)
            //     .into_iter()
            //     .zip(0_usize..)
            //     .filter(|(v, i)| **v == 1_i16 && *i != node)
            //     .map(|(v, i)| i);

            // filter by combinations that are not already populated
            // let node_view = node_graph.view();
            combinations_ = edges
                .row(node)
                .iter()
                .zip(0_usize..)
                .filter(|(v, i)| **v == 1_i16 && *i != node)
                .map(|(v, i)| i)
                .combinations(2)
                .filter(|v| edges[[v[0], v[1]]] == 0_i16)
                .collect()
            ;
        }

        let mut counter: i16 = 0;
        for c in combinations_ {
            counter += 1_i16;
            edges[[c[0], c[1]]] = 1_i16;
            edges[[c[1], c[0]]] = 1_i16;
            fx_array[[c[0], c[1]]] = &fx_array[[c[0], node]] * &fx_array[[node, c[1]]];
            fx_array[[c[1], c[0]]] = 1.0_f64 / &fx_array[[c[0], c[1]]];
        }

        if counter == 0 {
            prev_value.insert(node);
            return FXRates::_populate_remaining_arrays(
                fx_array.view_mut(),
                edges.view_mut(),
                prev_value,
            );
        } else {
            return FXRates::_populate_remaining_arrays(
                fx_array.view_mut(),
                edges.view_mut(),
                HashSet::from([node]),
            );
        }
    }

    fn calculate_array(&self) {
        // Setup node graph
        let mut node_graph: Array2<i16> = Array2::eye(self.currencies.len());
        for fxr in self.fx_rates.iter() {
            let row = self.currencies.get_index_of(&fxr.pair.0).unwrap();
            let col = self.currencies.get_index_of(&fxr.pair.1).unwrap();
            node_graph[[row, col]] = 1_i16;
            node_graph[[col, row]] = 1_i16;
        }
    }

    fn discover_evaluation_node_and_populate(
        &self,
        node_graph: Array2<i16>,
        fx_array: FXArray,
    ) -> (Array2<i16>, FXArray) {
        // discover the node with the most outgoing nodes
        let v = Array1::from_vec(
            node_graph
                .lanes(Axis(0))
                .into_iter()
                .map(|row| row.iter().sum::<i16>())
                .collect(),
        );
        let node = argabsmax(v.view());

        // filter the node indices of the directly linked nodes to node
        let linked_nodes = node_graph
            .row(node)
            .into_iter()
            .zip(0_usize..)
            .filter(|(v, i)| **v == 1_i16 && *i != node)
            .map(|(v, i)| i);

        // filter by combinations that are not already populated
        // let node_view = node_graph.view();
        let combinations_to_calculate = linked_nodes
            .combinations(2)
            .filter(|v| node_graph[[v[0], v[1]]] == 0_i16);

        // calculate the combinations and mutate the input
        let mut output_node_graph = node_graph.clone();
        match fx_array {
            FXArray::Dual(arr) => {
                let mut output_values = arr.clone();
                for c in combinations_to_calculate {
                    output_node_graph[[c[0], c[1]]] = 1_i16;
                    output_node_graph[[c[1], c[0]]] = 1_i16;
                    let value = &arr[[c[0], node]] / &arr[[node, c[1]]];
                    output_values[[c[0], c[1]]] = value.clone();
                    output_values[[c[1], c[0]]] = 1_f64 / value;
                }
                (output_node_graph, FXArray::Dual(output_values))
            }
            FXArray::Dual2(arr) => {
                let mut output_values = arr.clone();
                for c in combinations_to_calculate {
                    output_node_graph[[c[0], c[1]]] = 1_i16;
                    output_node_graph[[c[1], c[0]]] = 1_i16;
                    let value = &arr[[c[0], node]] / &arr[[node, c[1]]];
                    output_values[[c[0], c[1]]] = value.clone();
                    output_values[[c[1], c[0]]] = 1_f64 / value;
                }
                (output_node_graph, FXArray::Dual2(output_values))
            }
        }
    }

    pub fn get_ccy_index(&self, currency: &Ccy) -> Option<usize> {
        self.currencies.get_index_of(currency)
    }

    pub fn rate(&self, lhs: &Ccy, rhs: &Ccy) -> Option<DualsOrF64> {
        let dom_idx = self.currencies.get_index_of(lhs)?;
        let for_idx = self.currencies.get_index_of(rhs)?;
        match self.ad {
            1 => Some(DualsOrF64::Dual(self.arr_dual[[dom_idx, for_idx]].clone())),
            2 => match &self.arr_dual2 {
                Some(arr) => Some(DualsOrF64::Dual2(arr[[dom_idx, for_idx]].clone())),
                None => None,
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendars::calendar::ndt;
    use num_traits::Signed;

    #[test]
    fn ccy_creation() {
        let a = Ccy::try_new("usd").unwrap();
        let b = Ccy::try_new("USD").unwrap();
        assert_eq!(a, b)
    }

    #[test]
    fn ccy_creation_error() {
        match Ccy::try_new("FOUR") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    #[test]
    fn pair_creation() {
        let a = FXPair::try_new("usd", "eur").unwrap();
        let b = FXPair::try_new("USD", "EUR").unwrap();
        assert_eq!(a, b)
    }

    #[test]
    fn pair_creation_error() {
        match FXPair::try_new("usd", "USD") {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        }
    }

    #[test]
    fn rate_creation() {
        FXRate::try_new("usd", "eur", DualsOrF64::F64(1.20), None).unwrap();
    }

    #[test]
    fn fxrates_rate() {
        let fxr = FXRates::try_new(
            vec![
                FXRate::try_new("eur", "usd", DualsOrF64::F64(1.08), None).unwrap(),
                FXRate::try_new("usd", "jpy", DualsOrF64::F64(110.0), None).unwrap(),
            ],
            ndt(2004, 1, 1),
            None,
        )
        .unwrap();

        let expected = arr2(&[[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]);

        let arr: Vec<f64> = fxr.arr_dual.iter().map(|x| x.real()).collect();
        println!("{:?}", arr);

        assert!(fxr.arr_dual.iter().zip(expected.iter()).all(|(x,y)| (x-y).abs() < 1e-8 ))
    }
}
