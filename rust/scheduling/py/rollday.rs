use crate::json::{DeserializedObj, JSON};
use crate::scheduling::RollDay;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

enum RollDayNewArgs {
    U32(u32),
    NoArgs(),
}

impl<'py> IntoPyObject<'py> for RollDayNewArgs {
    type Target = PyTuple;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            RollDayNewArgs::U32(a) => Ok((a,).into_pyobject(py).unwrap()),
            RollDayNewArgs::NoArgs() => Ok(PyTuple::empty(py)),
        }
    }
}

impl<'py> FromPyObject<'py, 'py> for RollDayNewArgs {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let ext: PyResult<(u32,)> = obj.extract();
        match ext {
            Ok(v) => Ok(RollDayNewArgs::U32(v.0)),
            Err(_) => Ok(RollDayNewArgs::NoArgs()),
        }
    }
}

#[pymethods]
impl RollDay {
    pub(crate) fn __str__(&self) -> String {
        match self {
            RollDay::Day(n) => format!("{n}"),
            RollDay::IMM() => "IMM".to_string(),
        }
    }

    fn __getnewargs__(&self) -> RollDayNewArgs {
        match self {
            RollDay::Day(n) => RollDayNewArgs::U32(*n),
            RollDay::IMM() => RollDayNewArgs::NoArgs(),
        }
    }

    #[new]
    fn new_py(args: RollDayNewArgs) -> RollDay {
        match args {
            RollDayNewArgs::U32(n) => RollDay::Day(n),
            RollDayNewArgs::NoArgs() => RollDay::IMM(),
        }
    }

    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::RollDay(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `RollDay` to JSON.",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!("<rl.RollDay.{:?} at {:p}>", self, self)
    }
}
