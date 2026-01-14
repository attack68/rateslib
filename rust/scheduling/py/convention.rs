use crate::json::{DeserializedObj, JSON};
use crate::scheduling::{Adjuster, Calendar, Convention, Frequency, PyAdjuster};
use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl Convention {
    // Pickling
    #[new]
    fn new_py(variant: u8) -> PyResult<Convention> {
        match variant {
            0_u8 => Ok(Convention::Act365F),
            1_u8 => Ok(Convention::Act360),
            2_u8 => Ok(Convention::Thirty360),
            3_u8 => Ok(Convention::ThirtyU360),
            4_u8 => Ok(Convention::ThirtyE360),
            5_u8 => Ok(Convention::ThirtyE360ISDA),
            6_u8 => Ok(Convention::YearsAct365F),
            7_u8 => Ok(Convention::YearsAct360),
            8_u8 => Ok(Convention::YearsMonths),
            9_u8 => Ok(Convention::One),
            10_u8 => Ok(Convention::ActActISDA),
            11_u8 => Ok(Convention::ActActICMA),
            12_u8 => Ok(Convention::Bus252),
            13_u8 => Ok(Convention::ActActICMAStubAct365F),
            14_u8 => Ok(Convention::Act365_25),
            15_u8 => Ok(Convention::Act364),
            _ => Err(PyValueError::new_err(
                "unreachable code on Convention pickle.",
            )),
        }
    }

    /// Calculate the day count fraction of a period.
    ///
    /// Parameters
    /// ----------
    /// start : datetime
    ///     The adjusted start date of the calculation period.
    /// end : datetime
    ///     The adjusted end date of the calculation period.
    /// termination : datetime, optional
    ///     The adjusted termination date of the leg. Required only for some ``convention``.
    /// frequency : Frequency, str, optional
    ///     The frequency of the period. Required only for some ``convention``.
    /// stub : bool, optional
    ///    Indicates whether the period is a stub or not. Required only for some ``convention``.
    /// roll : str, int, optional
    ///     Used only if ``frequency`` is given in string form. Required only for some ``convention``.
    /// calendar: str, Calendar, optional
    ///     Used only of ``frequency`` is given in string form. Required only for some ``convention``.
    /// adjuster: Adjuster, str, optional
    ///     The :class:`~rateslib.scheduling.Adjuster` used to convert unadjusted dates to
    ///     adjusted accrual dates on the period. Required only for some ``convention``.
    ///
    /// Returns
    /// --------
    /// float
    ///
    /// Notes
    /// -----
    /// Further details on the required arguments can be found under ``Convention`` at the
    /// lower level Rust docs, see :rust:`rateslib-rs: Scheduling <scheduling>`.
    #[pyo3(name = "dcf", signature=(start, end, termination=None, frequency=None, stub=None, calendar=None, adjuster=None   ))]
    fn dcf_py(
        &self,
        start: NaiveDateTime,
        end: NaiveDateTime,
        termination: Option<NaiveDateTime>,
        frequency: Option<Frequency>,
        stub: Option<bool>,
        calendar: Option<Calendar>,
        adjuster: Option<PyAdjuster>,
    ) -> PyResult<f64> {
        let adjuster_opt: Option<Adjuster> = match adjuster {
            Some(val) => Some(val.into()),
            None => None,
        };
        self.dcf(
            &start,
            &end,
            termination.as_ref(),
            frequency.as_ref(),
            stub,
            calendar.as_ref(),
            adjuster_opt.as_ref(),
        )
    }
    fn __getnewargs__<'py>(&self) -> PyResult<(usize,)> {
        Ok((*self as usize,))
    }

    fn __repr__(&self) -> String {
        format!("<rl.Convention.{:?} at {:p}>", self, self)
    }

    fn __str__(&self) -> String {
        match self {
            Convention::Act360 => "Act360".to_string(),
            Convention::Act365F => "Act365F".to_string(),
            Convention::YearsAct365F => "YearsAct365F".to_string(),
            Convention::YearsAct360 => "YearsAct360".to_string(),
            Convention::YearsMonths => "YearsMonths".to_string(),
            Convention::Thirty360 => "30360".to_string(),
            Convention::ThirtyU360 => "30u360".to_string(),
            Convention::ThirtyE360 => "30e360".to_string(),
            Convention::ThirtyE360ISDA => "30e360ISDA".to_string(),
            Convention::One => "One".to_string(),
            Convention::ActActISDA => "ActActISDA".to_string(),
            Convention::ActActICMA => "ActActICMA".to_string(),
            Convention::Bus252 => "Bus252".to_string(),
            Convention::ActActICMAStubAct365F => "ActActICMAStubAct365F".to_string(),
            Convention::Act365_25 => "Act365_25".to_string(),
            Convention::Act364 => "Act364".to_string(),
        }
    }

    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::Convention(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `Convention` to JSON.",
            )),
        }
    }
}
