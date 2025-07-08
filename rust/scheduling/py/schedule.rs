use crate::scheduling::{
    try_new_regular_from_adjusted, Adjuster, Calendar, Frequency, Schedule, StubInference,
};

use chrono::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyfunction]
pub(crate) fn try_schedules(
    effective: NaiveDateTime,
    termination: NaiveDateTime,
    frequency: Frequency,
    calendar: Calendar,
    accrual_adjuster: Adjuster,
    payment_adjuster: Adjuster,
) -> Result<Vec<Schedule>, PyErr> {
    try_new_regular_from_adjusted(
        effective,
        termination,
        frequency,
        calendar,
        accrual_adjuster,
        payment_adjuster,
    )
}

#[pymethods]
impl Schedule {
    #[new]
    #[pyo3(signature = (ueffective, utermination, frequency, calendar, accrual_adjuster, payment_adjuster, ufront_stub=None, uback_stub=None, stub_inference=None))]
    fn new_py(
        ueffective: NaiveDateTime,
        utermination: NaiveDateTime,
        frequency: Frequency,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        ufront_stub: Option<NaiveDateTime>,
        uback_stub: Option<NaiveDateTime>,
        stub_inference: Option<StubInference>,
    ) -> PyResult<Self> {
        Schedule::try_new_uschedule_inferred(
            ueffective,
            utermination,
            frequency,
            ufront_stub,
            uback_stub,
            calendar,
            accrual_adjuster,
            payment_adjuster,
            stub_inference,
        )
    }

    #[classmethod]
    #[pyo3(name = "new", signature = (effective, termination, frequency, calendar, accrual_adjuster, payment_adjuster, eom, front_stub=None, back_stub=None))]
    fn new_regular_py(
        _cls: &Bound<'_, PyType>,
        effective: NaiveDateTime,
        termination: NaiveDateTime,
        frequency: Frequency,
        calendar: Calendar,
        accrual_adjuster: Adjuster,
        payment_adjuster: Adjuster,
        eom: bool,
        front_stub: Option<NaiveDateTime>,
        back_stub: Option<NaiveDateTime>,
    ) -> PyResult<Self> {
        Schedule::try_new_schedule(
            effective,
            termination,
            frequency,
            front_stub,
            back_stub,
            calendar,
            accrual_adjuster,
            payment_adjuster,
            eom,
        )
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
    fn accrual_adjuster_py(&self) -> Adjuster {
        self.accrual_adjuster
    }

    #[getter]
    #[pyo3(name = "calendar")]
    fn calendar_py(&self) -> Calendar {
        self.calendar.clone()
    }

    #[getter]
    #[pyo3(name = "payment_adjuster")]
    fn payment_adjuster_py(&self) -> Adjuster {
        self.payment_adjuster
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
}
