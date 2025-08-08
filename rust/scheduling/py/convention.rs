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
            13_u8 => Ok(Convention::ActActICMA_Stub_Act365F),
            _ => Err(PyValueError::new_err(
                "unreachable code on Convention pickle.",
            )),
        }
    }

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
}
