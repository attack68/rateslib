//! Static data for pre-existing named holiday calendars.
//!

pub mod all;
pub mod bus;
pub mod fed;
pub mod ldn;
pub mod mex;
pub mod mum;
pub mod nsw;
pub mod nyc;
pub mod osl;
pub mod stk;
pub mod syd;
pub mod tgt;
pub mod tro;
pub mod tyo;
pub mod wlg;
pub mod zur;

use chrono::NaiveDateTime;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::collections::HashMap;

pub(crate) fn get_weekmask_by_name(name: &str) -> Result<Vec<u8>, PyErr> {
    let hmap: HashMap<&str, &[u8]> = HashMap::from([
        ("all", all::WEEKMASK),
        ("bus", bus::WEEKMASK),
        ("nyc", nyc::WEEKMASK),
        ("fed", fed::WEEKMASK),
        ("tgt", tgt::WEEKMASK),
        ("ldn", ldn::WEEKMASK),
        ("stk", stk::WEEKMASK),
        ("osl", osl::WEEKMASK),
        ("zur", zur::WEEKMASK),
        ("tro", tro::WEEKMASK),
        ("tyo", tyo::WEEKMASK),
        ("syd", syd::WEEKMASK),
        ("nsw", nsw::WEEKMASK),
        ("wlg", wlg::WEEKMASK),
        ("mum", mum::WEEKMASK),
        ("mex", mex::WEEKMASK),
    ]);
    match hmap.get(name) {
        None => Err(PyValueError::new_err(format!(
            "'{}' is not found in list of existing calendars.",
            name
        ))),
        Some(value) => Ok(value.to_vec()),
    }
}

pub(crate) fn get_holidays_by_name(name: &str) -> Result<Vec<NaiveDateTime>, PyErr> {
    let hmap: HashMap<&str, &[&str]> = HashMap::from([
        ("all", all::HOLIDAYS),
        ("bus", bus::HOLIDAYS),
        ("nyc", nyc::HOLIDAYS),
        ("fed", fed::HOLIDAYS),
        ("tgt", tgt::HOLIDAYS),
        ("ldn", ldn::HOLIDAYS),
        ("stk", stk::HOLIDAYS),
        ("osl", osl::HOLIDAYS),
        ("zur", zur::HOLIDAYS),
        ("tro", tro::HOLIDAYS),
        ("tyo", tyo::HOLIDAYS),
        ("syd", syd::HOLIDAYS),
        ("nsw", nsw::HOLIDAYS),
        ("wlg", wlg::HOLIDAYS),
        ("mum", mum::HOLIDAYS),
        ("mex", mex::HOLIDAYS),
    ]);
    match hmap.get(name) {
        None => Err(PyValueError::new_err(format!(
            "'{}' is not found in list of existing calendars.",
            name
        ))),
        Some(value) => Ok(value
            .iter()
            .map(|x| NaiveDateTime::parse_from_str(x, "%Y-%m-%d %H:%M:%S").unwrap())
            .collect()),
    }
}

// fn get_rules_by_name(name: &str) -> Result<Vec<&str>, PyErr> {
//     let hmap: HashMap<&str, &[&str]> = HashMap::from([
//         ("all", all::RULES),
//         ("bus", bus::RULES),
//         ("nyc", nyc::RULES),
//         ("fed", fed::RULES),
//         ("tgt", tgt::RULES),
//         ("ldn", ldn::RULES),
//         ("stk", stk::RULES),
//         ("osl", osl::RULES),
//         ("zur", zur::RULES),
//         ("tro", tro::RULES),
//         ("tyo", tyo::RULES),
//         ("syd", syd::RULES),
//         ("nsw", nsw::RULES),
//         ("wlg", wlg::RULES),
//         ("mum", mum::RULES),
//         ("mex", mex::RULES),
//     ]);
//     match hmap.get(name) {
//         None => Err(PyValueError::new_err(format!(
//             "'{}' is not found in list of existing calendars.",
//             name
//         ))),
//         Some(value) => Ok(value.to_vec()),
//     }
// }

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_weekmask() {
        let result = get_weekmask_by_name("bus").unwrap();
        assert_eq!(result, vec![5, 6]);
    }

    #[test]
    fn test_get_holidays() {
        let result = get_holidays_by_name("bus").unwrap();
        assert_eq!(result, vec![]);
    }

    //     #[test]
    //     fn test_get_rules() {
    //         let result = get_rules_by_name("bus").unwrap();
    //         assert_eq!(result, Vec::<&str>::new());
    //     }
}
