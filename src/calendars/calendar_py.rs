use crate::calendars::calendar::HolCal;
use chrono::prelude::*;

#[pymethods]
impl HolCal {
    #[new]
    fn new_py(holidays: Vec<NaiveDateTime>, week_mask: Vec<u8>) -> PyResult<Self> {
        OK(HolCal::new(holidays, week_mask))
    }
}
