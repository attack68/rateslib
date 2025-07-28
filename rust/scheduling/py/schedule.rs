use crate::scheduling::{Calendar, Frequency, Schedule, StubInference, PyAdjuster};

use chrono::prelude::*;
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
    pub fn __getnewargs__<'py>(&self) -> PyResult<(usize,)> {
        Ok((*self as usize,))
    }
}

#[pymethods]
impl Schedule {
    #[new]
    #[pyo3(signature = (effective, termination, frequency, calendar, accrual_adjuster, payment_adjuster, eom, front_stub=None, back_stub=None, stub_inference=None))]
    fn new_py(
        effective: NaiveDateTime,
        termination: NaiveDateTime,
        frequency: Frequency,
        calendar: Calendar,
        accrual_adjuster: PyAdjuster,
        payment_adjuster: PyAdjuster,
        eom: bool,
        front_stub: Option<NaiveDateTime>,
        back_stub: Option<NaiveDateTime>,
        stub_inference: Option<StubInference>,
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
        bool,
        Option<NaiveDateTime>,
        Option<NaiveDateTime>,
        Option<StubInference>,
    )> {
        Ok((
            self.ueffective,
            self.utermination,
            self.frequency.clone(),
            self.calendar.clone(),
            self.accrual_adjuster.into(),
            self.payment_adjuster.into(),
            false,
            self.ufront_stub,
            self.uback_stub,
            None,
        ))
    }
}
