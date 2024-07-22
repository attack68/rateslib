use crate::calendars::calendar_py::Cals;
use crate::calendars::dcfs::DCF;

enum Nodes {
    F64(IndexMap<datetime, f64>),
    Dual(IndexMap<datetime, Dual>),
    Dual2(IndexMap<datetime, Dual2>),
}
pub struct Curve {
    nodes: Nodes,
    calendar: Cals,
    convention: Convention,
}