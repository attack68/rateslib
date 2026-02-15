// SPDX-License-Identifier: LicenseRef-Rateslib-Dual
//
// Copyright (c) 2026 Siffrorna Technology Limited
// This code cannot be used or copied externally
//
// Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
// Source-available, not open source.
//
// See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
// and/or contact info (at) rateslib (dot) com
////////////////////////////////////////////////////////////////////////////////////////////////////

//! Static data for pre-existing named holiday calendars.
//!

pub mod all;
pub mod bjs;
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
use std::collections::HashMap;
use std::sync::LazyLock;

pub(crate) static WEEKMASKS: LazyLock<HashMap<&str, &[u8]>> = LazyLock::new(|| {
    HashMap::from([
        ("all", all::WEEKMASK),
        ("bus", bus::WEEKMASK),
        ("bjs", bjs::WEEKMASK),
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
    ])
});

pub(crate) static HOLIDAYS: LazyLock<HashMap<&str, Vec<NaiveDateTime>>> = LazyLock::new(|| {
    let temp = HashMap::<&str, &[&str]>::from([
        ("all", all::HOLIDAYS),
        ("bus", bus::HOLIDAYS),
        ("bjs", bjs::HOLIDAYS),
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
    let mut m: HashMap<&str, Vec<NaiveDateTime>> = HashMap::new();
    for (k, v) in temp.into_iter() {
        m.insert(
            k,
            v.iter()
                .map(|x| NaiveDateTime::parse_from_str(x, "%Y-%m-%d %H:%M:%S").unwrap())
                .collect(),
        );
    }
    m
});

// UNIT TESTS
#[cfg(test)]
mod tests {}
