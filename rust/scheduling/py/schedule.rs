use crate::json::{DeserializedObj, JSON};
use crate::scheduling::{Calendar, Frequency, PyAdjuster, Schedule, StubInference};

use chrono::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl StubInference {
    // Pickling
    #[new]
    fn new_py(item: usize) -> StubInference {
        match item {
            _ if item == StubInference::ShortFront as usize => StubInference::ShortFront,
            _ if item == StubInference::ShortBack as usize => StubInference::ShortBack,
            _ if item == StubInference::LongFront as usize => StubInference::LongFront,
            _ if item == StubInference::LongBack as usize => StubInference::LongBack,
            _ => panic!("Reportable issue: must map this enum variant for serialization."),
        }
    }
    fn __getnewargs__<'py>(&self) -> PyResult<(usize,)> {
        Ok((*self as usize,))
    }
    fn __repr__(&self) -> String {
        format!("<rl.StubInference.{:?} at {:p}>", self, self)
    }
    // JSON
    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::StubInference(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `StubInference` to JSON.",
            )),
        }
    }
}

#[pymethods]
impl Schedule {
    #[new]
    #[pyo3(signature = (effective, termination, frequency, calendar, accrual_adjuster, payment_adjuster, payment_adjuster2, eom, front_stub=None, back_stub=None, stub_inference=None, payment_adjuster3=None))]
    fn new_py(
        effective: NaiveDateTime,
        termination: NaiveDateTime,
        frequency: Frequency,
        calendar: Calendar,
        accrual_adjuster: PyAdjuster,
        payment_adjuster: PyAdjuster,
        payment_adjuster2: PyAdjuster,
        eom: bool,
        front_stub: Option<NaiveDateTime>,
        back_stub: Option<NaiveDateTime>,
        stub_inference: Option<StubInference>,
        payment_adjuster3: Option<PyAdjuster>,
    ) -> PyResult<Self> {
        Schedule::try_new_inferred(
            effective,
            termination,
            frequency,
            front_stub,
            back_stub,
            calendar,
            accrual_adjuster.into(),
            payment_adjuster.into(),
            payment_adjuster2.into(),
            payment_adjuster3.map(Into::into),
            eom,
            stub_inference,
        )
    }

    #[pyo3(name = "is_regular")]
    fn is_regular_py(&self) -> bool {
        self.is_regular()
    }

    #[getter]
    #[pyo3(name = "ueffective")]
    fn ueffective_py(&self) -> NaiveDateTime {
        self.ueffective
    }

    #[getter]
    #[pyo3(name = "utermination")]
    fn utermination_py(&self) -> NaiveDateTime {
        self.utermination
    }

    #[getter]
    #[pyo3(name = "frequency")]
    fn frequency_py(&self) -> Frequency {
        self.frequency.clone()
    }

    #[getter]
    #[pyo3(name = "accrual_adjuster")]
    fn accrual_adjuster_py(&self) -> PyAdjuster {
        self.accrual_adjuster.into()
    }

    #[getter]
    #[pyo3(name = "calendar")]
    fn calendar_py(&self) -> Calendar {
        self.calendar.clone()
    }

    #[getter]
    #[pyo3(name = "payment_adjuster")]
    fn payment_adjuster_py(&self) -> PyAdjuster {
        self.payment_adjuster.into()
    }

    #[getter]
    #[pyo3(name = "payment_adjuster2")]
    fn payment_adjuster2_py(&self) -> PyAdjuster {
        self.payment_adjuster2.into()
    }

    #[getter]
    #[pyo3(name = "payment_adjuster3")]
    fn payment_adjuster3_py(&self) -> Option<PyAdjuster> {
        self.payment_adjuster3.map(Into::into)
    }

    #[getter]
    #[pyo3(name = "ufront_stub")]
    fn ufront_stub_py(&self) -> Option<NaiveDateTime> {
        self.ufront_stub
    }

    #[getter]
    #[pyo3(name = "uback_stub")]
    fn uback_stub_py(&self) -> Option<NaiveDateTime> {
        self.uback_stub
    }

    #[getter]
    #[pyo3(name = "uschedule")]
    fn uschedule_py(&self) -> Vec<NaiveDateTime> {
        self.uschedule.clone()
    }

    #[getter]
    #[pyo3(name = "aschedule")]
    fn aschedule_py(&self) -> Vec<NaiveDateTime> {
        self.aschedule.clone()
    }

    #[getter]
    #[pyo3(name = "pschedule")]
    fn pschedule_py(&self) -> Vec<NaiveDateTime> {
        self.pschedule.clone()
    }

    #[getter]
    #[pyo3(name = "pschedule2")]
    fn pschedule2_py(&self) -> Vec<NaiveDateTime> {
        self.pschedule2.clone()
    }

    #[getter]
    #[pyo3(name = "pschedule3")]
    fn pschedule3_py(&self) -> Vec<NaiveDateTime> {
        self.pschedule3.clone()
    }

    // Pickling
    fn __getnewargs__(
        &self,
    ) -> PyResult<(
        NaiveDateTime,
        NaiveDateTime,
        Frequency,
        Calendar,
        PyAdjuster,
        PyAdjuster,
        PyAdjuster,
        bool,
        Option<NaiveDateTime>,
        Option<NaiveDateTime>,
        Option<StubInference>,
        Option<PyAdjuster>,
    )> {
        Ok((
            self.ueffective,
            self.utermination,
            self.frequency.clone(),
            self.calendar.clone(),
            self.accrual_adjuster.into(),
            self.payment_adjuster.into(),
            self.payment_adjuster2.into(),
            false,
            self.ufront_stub,
            self.uback_stub,
            None,
            self.payment_adjuster3.map(Into::into),
        ))
    }

    /// Return a JSON representation of the object.
    ///
    /// Returns
    /// -------
    /// str
    #[pyo3(name = "to_json")]
    fn to_json_py(&self) -> PyResult<String> {
        match DeserializedObj::Schedule(self.clone()).to_json() {
            Ok(v) => Ok(v),
            Err(_) => Err(PyValueError::new_err(
                "Failed to serialize `Schedule` to JSON.",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!("<rl.Schedule at {:p}>", self)
    }
}
