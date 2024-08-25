pub use crate::dual::dual_ops::convert::{set_order, set_order_clone};
pub use crate::dual::dual_ops::math_funcs::MathFuncs;
pub use crate::dual::dual_ops::numeric_ops::NumberOps;
use indexmap::set::IndexSet;
use ndarray::{Array, Array1, Array2, Axis};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, FromPyObject, PyErr};
use serde::{Deserialize, Serialize};
use std::cmp::PartialEq;
use std::sync::Arc;

/// A dual number data type supporting first order derivatives.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Default, Debug, Deserialize, Serialize)]
pub struct Dual {
    pub(crate) real: f64,
    pub(crate) vars: Arc<IndexSet<String>>,
    pub(crate) dual: Array1<f64>,
}

/// A dual number data type supporting second order derivatives.
#[pyclass(module = "rateslib.rs")]
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Dual2 {
    pub(crate) real: f64,
    pub(crate) vars: Arc<IndexSet<String>>,
    pub(crate) dual: Array1<f64>,
    pub(crate) dual2: Array2<f64>,
}

/// Container for the three core numeric types; [f64], [Dual] and [Dual2].
#[derive(Debug, Clone, FromPyObject, Serialize, Deserialize)]
pub enum Number {
    Dual(Dual),
    Dual2(Dual2),
    F64(f64),
}

/// Container for [Vec] of each core numeric type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NumberVec {
    F64(Vec<f64>),
    Dual(Vec<Dual>),
    Dual2(Vec<Dual2>),
}

/// Container for [Array1] of each core numeric type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NumberArray1 {
    F64(Array1<f64>),
    Dual(Array1<Dual>),
    Dual2(Array1<Dual2>),
}

/// Container for [Array2] of each core numeric type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NumberArray2 {
    F64(Array2<f64>),
    Dual(Array2<Dual>),
    Dual2(Array2<Dual2>),
}

/// Generic trait indicating a function exists to map one [Number] to another.
///
/// An example of this trait is used by certain [PPSpline] indicating that an x-value as
/// some [Number] can be mapped under spline interpolation to some y-value as another [Number].
pub trait NumberMapping {
    fn mapped_value(&self, x: &Number) -> Result<Number, PyErr>;
}

#[pyclass(module = "rateslib.rs")]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ADOrder {
    /// Floating point arithmetic only.
    Zero,
    /// Derivatives available to first order.
    One,
    /// Derivatives available to second order.
    Two,
}

/// The state of the `vars` measured between two dual number type structs; a LHS relative to a RHS.
#[derive(Clone, Debug, PartialEq)]
pub enum VarsRelationship {
    /// The two structs share the same Arc pointer for their `vars`.
    ArcEquivalent,
    /// The structs have the same `vars` in the same order but not a shared Arc pointer.
    ValueEquivalent,
    /// The `vars` of the compared RHS is contained within those of the LHS.
    Superset,
    /// The `vars` of the calling LHS are contained within those of the RHS.
    Subset,
    /// Both the LHS and RHS have different `vars`.
    Difference,
}

/// Manages the `vars` of the manifold associated with a dual number.
pub trait Vars
where
    Self: Clone,
{
    /// Get a reference to the Arc pointer for the `IndexSet` containing the struct's variables.
    fn vars(&self) -> &Arc<IndexSet<String>>;

    /// Create a new dual number with `vars` aligned with given new Arc pointer.
    ///
    /// This method compares the existing `vars` with the new and reshuffles manifold gradient
    /// values in memory. For large numbers of variables this is one of the least efficient
    /// operations relating different dual numbers and should be avoided where possible.
    fn to_new_vars(
        &self,
        arc_vars: &Arc<IndexSet<String>>,
        state: Option<VarsRelationship>,
    ) -> Self;

    /// Compare the `vars` on a `Dual` with a given Arc pointer.
    fn vars_cmp(&self, arc_vars: &Arc<IndexSet<String>>) -> VarsRelationship {
        if Arc::ptr_eq(self.vars(), arc_vars) {
            VarsRelationship::ArcEquivalent
        } else if self.vars().len() == arc_vars.len()
            && self.vars().iter().zip(arc_vars.iter()).all(|(a, b)| a == b)
        {
            VarsRelationship::ValueEquivalent
        } else if self.vars().len() >= arc_vars.len()
            && arc_vars.iter().all(|var| self.vars().contains(var))
        {
            VarsRelationship::Superset
        } else if self.vars().len() < arc_vars.len()
            && self.vars().iter().all(|var| arc_vars.contains(var))
        {
            VarsRelationship::Subset
        } else {
            VarsRelationship::Difference
        }
    }
    // fn vars_cmp(&self, arc_vars: &Arc<IndexSet<String>>) -> VarsRelationship;

    /// Construct a tuple of 2 `Self` types whose `vars` are linked by an Arc pointer.
    ///
    /// Gradient values contained in fields may be shuffled in memory if necessary
    /// according to the calculated `VarsRelationship`. Do not use `state` directly unless you have
    /// performed a pre-check.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual, Vars, VarsRelationship};
    /// let x = Dual::new(1.0, vec!["x".to_string()]);
    /// let y = Dual::new(1.5, vec!["y".to_string()]);
    /// let (a, b) = x.to_union_vars(&y, Some(VarsRelationship::Difference));
    /// // a: <Dual: 1.0, (x, y), [1.0, 0.0]>
    /// // b: <Dual: 1.5, (x, y), [0.0, 1.0]>
    /// ```
    fn to_union_vars(&self, other: &Self, state: Option<VarsRelationship>) -> (Self, Self)
    where
        Self: Sized,
    {
        let state_ = state.unwrap_or_else(|| self.vars_cmp(other.vars()));
        match state_ {
            VarsRelationship::ArcEquivalent => (self.clone(), other.clone()),
            VarsRelationship::ValueEquivalent => {
                (self.clone(), other.to_new_vars(self.vars(), Some(state_)))
            }
            VarsRelationship::Superset => (
                self.clone(),
                other.to_new_vars(self.vars(), Some(VarsRelationship::Subset)),
            ),
            VarsRelationship::Subset => {
                (self.to_new_vars(other.vars(), Some(state_)), other.clone())
            }
            VarsRelationship::Difference => self.to_combined_vars(other),
        }
    }

    /// Construct a tuple of 2 `Self` types whose `vars` are linked by the explicit union
    /// of their own variables.
    ///
    /// Gradient values contained in fields will be shuffled in memory.
    fn to_combined_vars(&self, other: &Self) -> (Self, Self)
    where
        Self: Sized,
    {
        let comb_vars = Arc::new(IndexSet::from_iter(
            self.vars().union(other.vars()).cloned(),
        ));
        (
            self.to_new_vars(&comb_vars, Some(VarsRelationship::Difference)),
            other.to_new_vars(&comb_vars, Some(VarsRelationship::Difference)),
        )
    }

    /// Compare if two `Dual` structs share the same `vars` by Arc pointer equivalence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual, Vars};
    /// let x1 = Dual::new(1.5, vec!["x".to_string()]);
    /// let x2 = Dual::new(2.5, vec!["x".to_string()]);
    /// assert_eq!(x1.ptr_eq(&x2), false); // Vars are the same but not a shared Arc pointer
    /// ```
    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.vars(), other.vars())
    }
}

impl Vars for Dual {
    /// Get a reference to the Arc pointer for the `IndexSet` containing the struct's variables.
    fn vars(&self) -> &Arc<IndexSet<String>> {
        &self.vars
    }

    /// Construct a new `Dual` with `vars` set as the given Arc pointer and gradients shuffled in memory.
    ///
    /// Examples
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual, Vars};
    /// let x = Dual::new(1.5, vec!["x".to_string()]);
    /// let xy = Dual::new(2.5, vec!["x".to_string(), "y".to_string()]);
    /// let x_y = x.to_new_vars(xy.vars(), None);
    /// // x_y: <Dual: 1.5, (x, y), [1.0, 0.0]>
    /// assert_eq!(x_y, Dual::try_new(1.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]).unwrap());
    fn to_new_vars(
        &self,
        arc_vars: &Arc<IndexSet<String>>,
        state: Option<VarsRelationship>,
    ) -> Self {
        let match_val = state.unwrap_or_else(|| self.vars_cmp(arc_vars));
        let dual_: Array1<f64> = match match_val {
            VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
                self.dual.clone()
            }
            _ => {
                let lookup_or_zero = |v| match self.vars.get_index_of(v) {
                    Some(idx) => self.dual[idx],
                    None => 0.0_f64,
                };
                Array1::from_vec(arc_vars.iter().map(lookup_or_zero).collect())
            }
        };
        Self {
            real: self.real,
            vars: Arc::clone(arc_vars),
            dual: dual_,
        }
    }
}

impl Vars for Dual2 {
    /// Get a reference to the Arc pointer for the `IndexSet` containing the struct's variables.
    fn vars(&self) -> &Arc<IndexSet<String>> {
        &self.vars
    }

    /// Construct a new `Dual2` with `vars` set as the given Arc pointer and gradients shuffled in memory.
    ///
    /// Examples
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual2, Vars};
    /// let x = Dual2::new(1.5, vec!["x".to_string()]);
    /// let xy = Dual2::new(2.5, vec!["x".to_string(), "y".to_string()]);
    /// let x_y = x.to_new_vars(xy.vars(), None);
    /// // x_y: <Dual2: 1.5, (x, y), [1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]>
    /// assert_eq!(x_y, Dual2::try_new(1.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0], vec![]).unwrap());
    fn to_new_vars(
        &self,
        arc_vars: &Arc<IndexSet<String>>,
        state: Option<VarsRelationship>,
    ) -> Self {
        let dual_: Array1<f64>;
        let mut dual2_: Array2<f64> = Array2::zeros((arc_vars.len(), arc_vars.len()));
        let match_val = state.unwrap_or_else(|| self.vars_cmp(arc_vars));
        match match_val {
            VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
                dual_ = self.dual.clone();
                dual2_.clone_from(&self.dual2);
            }
            _ => {
                let lookup_or_zero = |v| match self.vars.get_index_of(v) {
                    Some(idx) => self.dual[idx],
                    None => 0.0_f64,
                };
                dual_ = Array1::from_vec(arc_vars.iter().map(lookup_or_zero).collect());

                let indices: Vec<Option<usize>> =
                    arc_vars.iter().map(|x| self.vars.get_index_of(x)).collect();
                for (i, row_index) in indices.iter().enumerate() {
                    match row_index {
                        Some(row_value) => {
                            for (j, col_index) in indices.iter().enumerate() {
                                match col_index {
                                    Some(col_value) => {
                                        dual2_[[i, j]] = self.dual2[[*row_value, *col_value]]
                                    }
                                    None => {}
                                }
                            }
                        }
                        None => {}
                    }
                }
            }
        }
        Self {
            real: self.real,
            vars: Arc::clone(arc_vars),
            dual: dual_,
            dual2: dual2_,
        }
    }
}

/// Provides calculations of first order gradients to all, or a set of provided, `vars`.
pub trait Gradient1: Vars {
    /// Get a reference to the Array containing the first order gradients.
    fn dual(&self) -> &Array1<f64>;

    /// Return a set of first order gradients ordered by the given vector.
    ///
    /// Duplicate `vars` are dropped before parsing.
    fn gradient1(&self, vars: Vec<String>) -> Array1<f64> {
        let arc_vars = Arc::new(IndexSet::from_iter(vars));
        let state = self.vars_cmp(&arc_vars);
        match state {
            VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
                self.dual().clone()
            }
            _ => {
                let mut dual_ = Array1::<f64>::zeros(arc_vars.len());
                for (i, index) in arc_vars
                    .iter()
                    .map(|x| self.vars().get_index_of(x))
                    .enumerate()
                {
                    if let Some(value) = index {
                        dual_[i] = self.dual()[value]
                    }
                }
                dual_
            }
        }
    }
}

impl Gradient1 for Dual {
    fn dual(&self) -> &Array1<f64> {
        &self.dual
    }
}

impl Gradient1 for Dual2 {
    fn dual(&self) -> &Array1<f64> {
        &self.dual
    }
}

/// Provides calculations of second order gradients to all, or a set of provided, `vars`.
pub trait Gradient2: Gradient1 {
    /// Get a reference to the Array containing the second order gradients.
    fn dual2(&self) -> &Array2<f64>;

    /// Return a set of first order gradients ordered by the given vector.
    ///
    /// Duplicate `vars` are dropped before parsing.
    fn gradient2(&self, vars: Vec<String>) -> Array2<f64> {
        let arc_vars = Arc::new(IndexSet::from_iter(vars));
        let state = self.vars_cmp(&arc_vars);
        match state {
            VarsRelationship::ArcEquivalent | VarsRelationship::ValueEquivalent => {
                2.0_f64 * self.dual2()
            }
            _ => {
                let indices: Vec<Option<usize>> = arc_vars
                    .iter()
                    .map(|x| self.vars().get_index_of(x))
                    .collect();
                let mut dual2_ = Array::zeros((arc_vars.len(), arc_vars.len()));
                for (i, row_index) in indices.iter().enumerate() {
                    for (j, col_index) in indices.iter().enumerate() {
                        match row_index {
                            Some(row_value) => match col_index {
                                Some(col_value) => {
                                    dual2_[[i, j]] = self.dual2()[[*row_value, *col_value]]
                                }
                                None => {}
                            },
                            None => {}
                        }
                    }
                }
                2_f64 * dual2_
            }
        }
    }

    fn gradient1_manifold(&self, vars: Vec<String>) -> Array1<Dual2> {
        let indices: Vec<Option<usize>> =
            vars.iter().map(|x| self.vars().get_index_of(x)).collect();

        let default_zero = Dual2::new(0., vars.clone());
        let mut grad: Array1<Dual2> = Array1::zeros(vars.len());
        for (i, i_idx) in indices.iter().enumerate() {
            match i_idx {
                Some(i_val) => {
                    let mut dual: Array1<f64> = Array1::zeros(vars.len());
                    for (j, j_idx) in indices.iter().enumerate() {
                        match j_idx {
                            Some(j_val) => dual[j] = self.dual2()[[*i_val, *j_val]] * 2.0,
                            None => {}
                        }
                    }
                    grad[i] = Dual2 {
                        real: self.dual()[*i_val],
                        vars: Arc::clone(&default_zero.vars),
                        dual2: Array2::zeros((vars.len(), vars.len())),
                        dual,
                    };
                }
                None => grad[i] = default_zero.clone(),
            }
        }
        grad
    }
}

impl Gradient2 for Dual2 {
    fn dual2(&self) -> &Array2<f64> {
        &self.dual2
    }
}

impl Dual {
    /// Constructs a new `Dual`.
    ///
    /// - `vars` should be **unique**; duplicates will be removed by the `IndexSet`.
    ///
    /// Gradient values for each of the provided `vars` is set to 1.0_f64.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::Dual;
    /// let x = Dual::new(2.5, vec!["x".to_string()]);
    /// // x: <Dual: 2.5, (x), [1.0]>
    /// ```
    pub fn new(real: f64, vars: Vec<String>) -> Self {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        Self {
            real,
            dual: Array1::ones(unique_vars_.len()),
            vars: unique_vars_,
        }
    }

    /// Constructs a new `Dual`.
    ///
    /// - `vars` should be **unique**; duplicates will be removed by the `IndexSet`.
    /// - `dual` can be empty; if so each gradient with respect to each `vars` is set to 1.0_f64.
    ///
    /// `try_new` should be used instead of `new` when gradient values other than 1.0_f64 are to
    /// be initialised.
    ///
    /// # Errors
    ///
    /// If the length of `dual` and of `vars` are not the same after parsing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::Dual;
    /// let x = Dual::try_new(2.5, vec!["x".to_string()], vec![4.2]).unwrap();
    /// // x: <Dual: 2.5, (x), [4.2]>
    /// ```
    pub fn try_new(real: f64, vars: Vec<String>, dual: Vec<f64>) -> Result<Self, PyErr> {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        let dual_ = if dual.is_empty() {
            Array1::ones(unique_vars_.len())
        } else {
            Array1::from_vec(dual)
        };
        if unique_vars_.len() != dual_.len() {
            Err(PyValueError::new_err(
                "`vars` and `dual` must have the same length.",
            ))
        } else {
            Ok(Self {
                real,
                vars: unique_vars_,
                dual: dual_,
            })
        }
    }

    /// Construct a new `Dual` cloning the `vars` Arc pointer from another.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::Dual;
    /// let x = Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]).unwrap();
    /// let y1 = Dual::new_from(&x, 1.5, vec!["y".to_string()]);
    /// ```
    ///
    /// This is semantically the same as:
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual, Vars};
    /// # let x = Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]).unwrap();
    /// # let y1 = Dual::new_from(&x, 1.5, vec!["y".to_string()]);
    /// let y2 = Dual::new(1.5, vec!["y".to_string()]).to_new_vars(x.vars(), None);
    /// assert_eq!(y1, y2);
    /// ```
    pub fn new_from<T: Vars>(other: &T, real: f64, vars: Vec<String>) -> Self {
        let new = Self::new(real, vars);
        new.to_new_vars(other.vars(), None)
    }

    /// Construct a new `Dual` cloning the `vars` Arc pointer from another.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual, Vars};
    /// let x = Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]).unwrap();
    /// let y1 = Dual::try_new_from(&x, 1.5, vec!["y".to_string()], vec![3.2]).unwrap();
    /// ```
    ///
    /// This is semantically the same as:
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual, Vars};
    /// # let x = Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]).unwrap();
    /// # let y1 = Dual::try_new_from(&x, 1.5, vec!["y".to_string()], vec![3.2]).unwrap();
    /// let y2 = Dual::try_new(1.5, vec!["y".to_string()], vec![3.2]).unwrap().to_new_vars(x.vars(), None);
    /// assert_eq!(y1, y2);
    /// ```
    pub fn try_new_from<T: Vars>(
        other: &T,
        real: f64,
        vars: Vec<String>,
        dual: Vec<f64>,
    ) -> Result<Self, PyErr> {
        let new = Self::try_new(real, vars, dual)?;
        Ok(new.to_new_vars(other.vars(), None))
    }

    /// Construct a new `Dual` cloning the `vars` Arc pointer from another.
    ///
    pub fn clone_from<T: Vars>(other: &T, real: f64, dual: Array1<f64>) -> Self {
        assert_eq!(other.vars().len(), dual.len());
        Dual {
            real,
            vars: Arc::clone(other.vars()),
            dual,
        }
    }

    /// Get the real component value of the struct.
    pub fn real(&self) -> f64 {
        self.real
    }
}

impl Dual2 {
    /// Constructs a new `Dual2`.
    ///
    /// - `vars` should be **unique**; duplicates will be removed by the `IndexSet`.
    ///
    /// Gradient values for each of the provided `vars` is set to 1.0_f64.
    /// Second order gradient values for each combination of provided `vars` is set
    /// to 0.0_f64.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::Dual2;
    /// let x = Dual2::new(2.5, vec!["x".to_string()]);
    /// // x: <Dual2: 2.5, (x), [1.0], [[0.0]]>
    /// ```
    pub fn new(real: f64, vars: Vec<String>) -> Self {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        Self {
            real,
            dual: Array1::ones(unique_vars_.len()),
            dual2: Array2::zeros((unique_vars_.len(), unique_vars_.len())),
            vars: unique_vars_,
        }
    }

    /// Constructs a new `Dual2`.
    ///
    /// - `vars` should be **unique**; duplicates will be removed by the `IndexSet`.
    /// - `dual` can be empty; if so each gradient with respect to each `vars` is set to 1.0_f64.
    /// - `dual2` can be empty; if so each gradient with respect to each `vars` is set to 0.0_f64.
    ///   Input as a flattened 2d-array in row major order.
    ///
    /// # Errors
    ///
    /// If the length of `dual` and of `vars` are not the same after parsing.
    /// If the shape of two dimension `dual2` does not match `vars` after parsing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::Dual2;
    /// let x = Dual2::try_new(2.5, vec!["x".to_string()], vec![], vec![]).unwrap();
    /// // x: <Dual2: 2.5, (x), [1.0], [[0.0]]>
    /// ```
    pub fn try_new(
        real: f64,
        vars: Vec<String>,
        dual: Vec<f64>,
        dual2: Vec<f64>,
    ) -> Result<Self, PyErr> {
        let unique_vars_ = Arc::new(IndexSet::from_iter(vars));
        let dual_ = if dual.is_empty() {
            Array1::ones(unique_vars_.len())
        } else {
            Array1::from_vec(dual)
        };
        if unique_vars_.len() != dual_.len() {
            return Err(PyValueError::new_err(
                "`vars` and `dual` must have the same length.",
            ));
        }

        let dual2_ = if dual2.is_empty() {
            Array2::zeros((unique_vars_.len(), unique_vars_.len()))
        } else {
            if dual2.len() != (unique_vars_.len() * unique_vars_.len()) {
                return Err(PyValueError::new_err(
                    "`vars` and `dual2` must have compatible lengths.",
                ));
            }
            Array::from_vec(dual2)
                .into_shape((unique_vars_.len(), unique_vars_.len()))
                .expect("Reshaping failed, which should not occur because shape is pre-checked.")
        };
        Ok(Self {
            real,
            vars: unique_vars_,
            dual: dual_,
            dual2: dual2_,
        })
    }

    /// Construct a new `Dual2` cloning the `vars` Arc pointer from another.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::Dual2;
    /// let x = Dual2::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0], vec![]).unwrap();
    /// let y1 = Dual2::new_from(&x, 1.5, vec!["y".to_string()]);
    /// ```
    ///
    /// This is semantically the same as:
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual2, Vars};
    /// # let x = Dual2::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0], vec![]).unwrap();
    /// # let y1 = Dual2::new_from(&x, 1.5, vec!["y".to_string()]);
    /// let y = Dual2::new(1.5, vec!["y".to_string()]).to_new_vars(x.vars(), None);
    /// ```
    pub fn new_from<T: Vars>(other: &T, real: f64, vars: Vec<String>) -> Self {
        let new = Self::new(real, vars);
        new.to_new_vars(other.vars(), None)
    }

    /// Construct a new `Dual2` cloning the `vars` Arc pointer from another.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use rateslib::dual::Dual2;
    /// let x = Dual2::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0], vec![]).unwrap();
    /// let y1 = Dual2::new_from(&x, 1.5, vec!["y".to_string()]);
    /// ```
    ///
    /// This is semantically the same as:
    ///
    /// ```rust
    /// # use rateslib::dual::{Dual2, Vars};
    /// # let x = Dual2::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0], vec![]).unwrap();
    /// # let y1 = Dual2::new_from(&x, 1.5, vec!["y".to_string()]);
    /// let y2 = Dual2::new(1.5, vec!["y".to_string()]).to_new_vars(x.vars(), None);
    /// assert_eq!(y1, y2);
    /// ```
    pub fn try_new_from<T: Vars>(
        other: &T,
        real: f64,
        vars: Vec<String>,
        dual: Vec<f64>,
        dual2: Vec<f64>,
    ) -> Result<Self, PyErr> {
        let new = Self::try_new(real, vars, dual, dual2)?;
        Ok(new.to_new_vars(other.vars(), None))
    }

    /// Construct a new `Dual2` cloning the `vars` Arc pointer from another.
    ///
    pub fn clone_from<T: Vars>(
        other: &T,
        real: f64,
        dual: Array1<f64>,
        dual2: Array2<f64>,
    ) -> Self {
        assert_eq!(other.vars().len(), dual.len());
        assert_eq!(other.vars().len(), dual2.len_of(Axis(0)));
        assert_eq!(other.vars().len(), dual2.len_of(Axis(1)));
        Dual2 {
            real,
            vars: Arc::clone(other.vars()),
            dual,
            dual2,
        }
    }

    /// Get the real component value of the struct.
    pub fn real(&self) -> f64 {
        self.real
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::dual::Dual2;
    use std::ops::{Add, Div, Mul, Sub};
    use std::time::Instant;

    #[test]
    fn new() {
        let x = Dual::new(1.0, vec!["a".to_string(), "a".to_string()]);
        assert_eq!(x.real, 1.0_f64);
        assert_eq!(
            *x.vars,
            IndexSet::<String>::from_iter(vec!["a".to_string()])
        );
        assert_eq!(x.dual, Array1::from_vec(vec![1.0_f64]));
    }

    #[test]
    fn new_with_dual() {
        let x = Dual::try_new(1.0, vec!["a".to_string(), "a".to_string()], vec![2.5]).unwrap();
        assert_eq!(x.real, 1.0_f64);
        assert_eq!(
            *x.vars,
            IndexSet::<String>::from_iter(vec!["a".to_string()])
        );
        assert_eq!(x.dual, Array1::from_vec(vec![2.5_f64]));
    }

    #[test]
    fn new_len_mismatch() {
        let result =
            Dual::try_new(1.0, vec!["a".to_string(), "a".to_string()], vec![1.0, 2.0]).is_err();
        assert!(result);
    }

    #[test]
    fn ptr_eq() {
        let x = Dual::new(1.0, vec!["a".to_string()]);
        let y = Dual::new(1.0, vec!["a".to_string()]);
        assert!(x.ptr_eq(&y) == false);
    }

    #[test]
    fn to_new_vars() {
        let x = Dual::try_new(1.5, vec!["a".to_string(), "b".to_string()], vec![1., 2.]).unwrap();
        let y = Dual::try_new(2.0, vec!["a".to_string(), "c".to_string()], vec![3., 3.]).unwrap();
        let z = x.to_new_vars(&y.vars, None);
        assert_eq!(z.real, 1.5_f64);
        assert!(y.ptr_eq(&z));
        assert_eq!(z.dual, Array1::from_vec(vec![1.0, 0.0]));
        let u = x.to_new_vars(x.vars(), None);
        assert!(u.ptr_eq(&x))
    }

    #[test]
    fn new_from() {
        let x = Dual::try_new(2.0, vec!["a".to_string(), "b".to_string()], vec![3., 3.]).unwrap();
        let y = Dual::try_new_from(
            &x,
            2.0,
            vec!["a".to_string(), "c".to_string()],
            vec![3., 3.],
        )
        .unwrap();
        assert_eq!(y.real, 2.0_f64);
        assert!(y.ptr_eq(&x));
        assert_eq!(y.dual, Array1::from_vec(vec![3.0, 0.0]));
    }

    #[test]
    fn vars() {
        let x = Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]).unwrap();
        let y = Dual::new(1.5, vec!["y".to_string()]).to_new_vars(x.vars(), None);
        assert!(x.ptr_eq(&y));
        assert_eq!(y.dual, Array1::from_vec(vec![0.0, 1.0]));
    }

    #[test]
    fn vars_cmp() {
        let x = Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.0, 0.0]).unwrap();
        let y = Dual::new(1.5, vec!["y".to_string()]);
        let y2 = Dual::new(1.5, vec!["y".to_string()]);
        let z = x.to_new_vars(y.vars(), None);
        let u = Dual::new(1.5, vec!["u".to_string()]);
        assert_eq!(x.vars_cmp(y.vars()), VarsRelationship::Superset);
        assert_eq!(y.vars_cmp(z.vars()), VarsRelationship::ArcEquivalent);
        assert_eq!(y.vars_cmp(y2.vars()), VarsRelationship::ValueEquivalent);
        assert_eq!(y.vars_cmp(x.vars()), VarsRelationship::Subset);
        assert_eq!(y.vars_cmp(u.vars()), VarsRelationship::Difference);
    }

    #[test]
    fn default() {
        let x = Dual::default();
        assert_eq!(x.real, 0.0_f64);
        assert_eq!(x.vars.len(), 0_usize);
        assert_eq!(x.dual, Array1::<f64>::from_vec(vec![]));
    }

    // OPS TESTS

    #[test]
    fn unitialised_derivs_eq_1() {
        let d = Dual::new(2.3, Vec::from([String::from("a"), String::from("b")]));
        for (_, val) in d.dual.indexed_iter() {
            assert!(*val == 1.0)
        }
    }

    #[test]
    fn gradient1_no_equiv() {
        let d1 =
            Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.1, 2.2]).unwrap();
        let result = d1.gradient1(vec!["y".to_string(), "z".to_string(), "x".to_string()]);
        let expected = Array1::from_vec(vec![2.2, 0.0, 1.1]);
        assert_eq!(result, expected)
    }

    #[test]
    fn gradient1_equiv() {
        let d1 =
            Dual::try_new(2.5, vec!["x".to_string(), "y".to_string()], vec![1.1, 2.2]).unwrap();
        let result = d1.gradient1(vec!["x".to_string(), "y".to_string()]);
        let expected = Array1::from_vec(vec![1.1, 2.2]);
        assert_eq!(result, expected)
    }

    // PROFILING

    #[test]
    fn vars_cmp_profile() {
        // Setup
        let vars = 500_usize;
        let x = Dual::try_new(
            1.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (1..=vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let y = Dual::try_new(
            1.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (1..=vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let z = Dual::new_from(&x, 1.0, Vec::new());
        let u = Dual::try_new(
            1.5,
            (1..vars).map(|x| x.to_string()).collect(),
            (1..vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let s = Dual::try_new(
            1.5,
            (0..(vars - 1)).map(|x| x.to_string()).collect(), // 2..Vars+1 13us  0..Vars-1  48ns
            (1..vars).map(|x| x as f64).collect(),
        )
        .unwrap();

        println!("\nProfiling vars_cmp (VarsRelationship::ArcEquivalent):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&z.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100000);

        println!("\nProfiling vars_cmp (VarsRelationship::ValueEquivalent):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&y.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);

        println!("\nProfiling vars_cmp (VarsRelationship::Superset):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&u.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);

        println!("\nProfiling vars_cmp (VarsRelationship::Different):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.vars_cmp(&s.vars);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);
    }

    #[test]
    fn to_union_vars_profile() {
        // Setup
        let vars = 500_usize;
        let x = Dual::try_new(
            1.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (0..vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let y = Dual::try_new(
            1.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (0..vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let z = Dual::new_from(&x, 1.0, Vec::new());
        let u = Dual::try_new(
            1.5,
            (1..vars).map(|x| x.to_string()).collect(),
            (1..vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let s = Dual::try_new(
            1.5,
            (0..(vars - 1)).map(|x| x.to_string()).collect(),
            (0..(vars - 1)).map(|x| x as f64).collect(),
        )
        .unwrap();

        println!("\nProfiling to_union_vars (VarsRelationship::ArcEquivalent):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&z, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100000);

        println!("\nProfiling to_union_vars (VarsRelationship::ValueEquivalent):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&y, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 1000);

        println!("\nProfiling to_union_vars (VarsRelationship::Superset):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&u, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100);

        println!("\nProfiling to_union_vars (VarsRelationship::Different):");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..100 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                x.to_union_vars(&s, None);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.2?}", elapsed / 100);
    }

    #[test]
    fn std_ops_ref_profile() {
        fn four_ops<T>(a: &T, b: &T, c: &T, d: &T) -> T
        where
            for<'a> &'a T: Add<&'a T, Output = T>
                + Sub<&'a T, Output = T>
                + Div<&'a T, Output = T>
                + Mul<&'a T, Output = T>,
        {
            &(&(a + b) * &(c / d)) - a
        }

        let vars = 500_usize;
        let a = Dual::try_new(
            1.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (0..vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        // let b = Dual::new(
        //     3.5,
        //     (2..=(VARS+1)).map(|x| x.to_string()).collect(),
        //     (0..VARS).map(|x| x as f64).collect(),
        // );
        // let c = Dual::new(
        //     5.5,
        //     (3..=(VARS+2)).map(|x| x.to_string()).collect(),
        //     (0..VARS).map(|x| x as f64).collect(),
        // );
        // let d = Dual::new(
        //     6.5,
        //     (4..=(VARS+3)).map(|x| x.to_string()).collect(),
        //     (0..VARS).map(|x| x as f64).collect(),
        // );
        let b = Dual::try_new_from(
            &a,
            3.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (0..vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let c = Dual::try_new_from(
            &a,
            5.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (0..vars).map(|x| x as f64).collect(),
        )
        .unwrap();
        let d = Dual::try_new_from(
            &a,
            6.5,
            (1..=vars).map(|x| x.to_string()).collect(),
            (0..vars).map(|x| x as f64).collect(),
        )
        .unwrap();

        println!("\nProfiling f64 std ops:");
        let now = Instant::now();
        // Code block to measure.
        {
            for _i in 0..1000 {
                // Arc::ptr_eq(&x.vars, &y.vars);
                let _x = four_ops(&a, &b, &c, &d);
            }
        }
        let elapsed = now.elapsed();
        println!("\nElapsed: {:.9?}", elapsed / 1000);
    }

    // copied from old dual2.rs

    use ndarray::arr2;

    #[test]
    fn clone_arc2() {
        let d1 = Dual2::new(20.0, vec!["a".to_string()]);
        let d2 = d1.clone();
        assert!(Arc::ptr_eq(&d1.vars, &d2.vars))
    }

    #[test]
    fn default_dual2() {
        let result = Dual2::default();
        let expected = Dual2::new(0.0, Vec::new());
        assert_eq!(result, expected);
    }

    #[test]
    fn to_new_ordered_vars2() {
        let d1 = Dual2::new(20.0, vec!["a".to_string()]);
        let d2 = Dual2::new(20.0, vec!["a".to_string(), "b".to_string()]);
        let d3 = d1.to_new_vars(&d2.vars, None);
        assert!(Arc::ptr_eq(&d3.vars, &d2.vars));
        assert!(d3.dual.len() == 2);
        let d4 = d2.to_new_vars(&d1.vars, None);
        assert!(Arc::ptr_eq(&d4.vars, &d1.vars));
        assert!(d4.dual.len() == 1);
    }

    #[test]
    fn new_dual2() {
        Dual2::new(2.3, Vec::from([String::from("a")]));
    }

    #[test]
    fn new_dual_error2() {
        assert!(Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([1.0]),
            Vec::new(),
        )
        .is_err());
    }

    #[test]
    fn new_dual2_error() {
        assert!(Dual2::try_new(
            2.3,
            Vec::from([String::from("a"), String::from("b")]),
            Vec::from([1.0, 2.3]),
            Vec::from([1.0, 2.4, 3.4]),
        )
        .is_err());
    }

    #[test]
    fn try_new_from2() {
        let x = Dual2::new(1.2, vec!["x".to_string(), "y".to_string()]);
        let y = Dual2::try_new_from(&x, 3.2, vec!["y".to_string()], vec![1.9], vec![2.1]).unwrap();
        let z = Dual2::try_new(
            3.2,
            vec!["x".to_string(), "y".to_string()],
            vec![0., 1.9],
            vec![0., 0., 0., 2.1],
        )
        .unwrap();
        assert_eq!(y, z);
    }

    #[test]
    fn to_new_vars2() {
        let d1 = Dual2::new(2.5, vec!["x".to_string()]);
        let d2 = Dual2::new(3.5, vec!["x".to_string()]);
        let d3 = d1.to_new_vars(d2.vars(), None);
        assert!(d3.ptr_eq(&d2));
        assert_eq!(d3.real, 2.5);
        assert_eq!(d3.dual, Array1::from_vec(vec![1.0]));
    }

    #[test]
    fn gradient2_equivval() {
        let d1 = Dual2::try_new(
            2.5,
            vec!["x".to_string(), "y".to_string()],
            vec![2.3, 4.5],
            vec![1.0, 2.5, 2.5, 5.0],
        )
        .unwrap();
        let result = d1.gradient2(vec!["x".to_string(), "y".to_string()]);
        let expected = arr2(&[[2., 5.], [5., 10.]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn gradient2_diffvars2() {
        let d1 = Dual2::try_new(
            2.5,
            vec!["x".to_string(), "y".to_string()],
            vec![2.3, 4.5],
            vec![1.0, 2.5, 2.5, 5.0],
        )
        .unwrap();
        let result = d1.gradient2(vec!["z".to_string(), "y".to_string()]);
        let expected = arr2(&[[0., 0.], [0., 10.]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn uninitialised_derivs_eq_one2() {
        let d = Dual2::new(2.3, Vec::from([String::from("a"), String::from("b")]));
        for (_, val) in d.dual.indexed_iter() {
            assert!(*val == 1.0)
        }
    }

    #[test]
    fn ops_equiv2() {
        let d1 = Dual2::try_new(1.5, vec!["x".to_string()], vec![1.0], vec![0.0]).unwrap();
        let d2 = Dual2::try_new(2.5, vec!["x".to_string()], vec![2.0], vec![0.0]).unwrap();
        let result = &d1 + &d2;
        assert_eq!(
            result,
            Dual2::try_new(4.0, vec!["x".to_string()], vec![3.0], vec![0.0]).unwrap()
        );
        let result = &d1 - &d2;
        assert_eq!(
            result,
            Dual2::try_new(-1.0, vec!["x".to_string()], vec![-1.0], vec![0.0]).unwrap()
        );
    }

    #[test]
    fn grad_manifold() {
        let d1 = Dual2::try_new(
            2.0,
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            vec![1., 2., 3.],
            vec![2., 3., 4., 3., 5., 6., 4., 6., 7.],
        )
        .unwrap();
        let result = d1.gradient1_manifold(vec!["y".to_string(), "z".to_string()]);
        assert_eq!(result[0].real, 2.);
        assert_eq!(result[1].real, 3.);
        assert_eq!(result[0].dual, Array1::from_vec(vec![10., 12.]));
        assert_eq!(result[1].dual, Array1::from_vec(vec![12., 14.]));
        assert_eq!(result[0].dual2, Array2::<f64>::zeros((2, 2)));
        assert_eq!(result[1].dual2, Array2::<f64>::zeros((2, 2)));
    }

    // #[test]
    // #[should_panic]
    // fn no_dual_cross(){
    //     let a = Dual::new(2.0, Vec::new(), Vec::new());
    //     let b = Dual2::new(3.0, Vec::new(), Vec::new(), Vec::new());
    //     a + b
    // }
}
