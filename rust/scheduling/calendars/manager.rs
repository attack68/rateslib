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

use crate::scheduling::calendars::named::{HOLIDAYS, WEEKMASKS};
use crate::scheduling::calendars::{Cal, CalWrapper, Calendar, NamedCal, UnionCal};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::{pyclass, PyErr};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

// A single memory allocated space to maintain the UnionCal with an associated name.
static NAMED_CALENDARS: LazyLock<RwLock<HashMap<String, Arc<CalWrapper>>>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    for (k, _) in WEEKMASKS.iter() {
        m.insert(
            (*k).into(),
            Arc::new(CalWrapper::Cal(Cal::new(
                HOLIDAYS.get(k).unwrap().to_vec(),
                WEEKMASKS.get(k).unwrap().to_vec(),
            ))),
        );
    }
    RwLock::new(m)
});

/// A manager to add and mutate the core calendars from which [`NamedCal`] are constructed.
#[pyclass]
pub struct CalendarManager;

impl CalendarManager {
    /// Create an instance of the [`CalendarManager`] manager.
    ///
    /// This object interacts with the memory allocation for stored calendars. It returns
    /// objects with thread safe, shared memory access to the same objects for performance.
    pub fn new() -> Self {
        Self {}
    }

    /// Returns *true* if the set contains a specific key.
    pub fn contains_key(&self, key: &str) -> bool {
        let k: String = sort_calendar_names(key);
        let r = NAMED_CALENDARS.read().unwrap();
        r.contains_key(&k)
    }

    /// Return a list of keys.
    pub fn keys(&self) -> Vec<String> {
        let r = NAMED_CALENDARS.read().unwrap();
        r.iter().map(|(k, _)| k.to_string()).collect()
    }

    /// Add any [`Calendar`] to the calendar manager.
    ///
    /// Data will not be overwritten. It will error prior to that or clone existing data to a
    /// new key.
    pub fn add(&self, name: &str, calendar: Cal) -> Result<(), PyErr> {
        let k: String = sort_calendar_names(name);
        if k.chars().any(|c| c == ',' || c == '|') {
            return Err(PyValueError::new_err(
                "
            `name` cannot contain the comma (',') or pipe ('|') characters.\nThese are reserved
            to define calendar combinations (i.e. UnionCal) and only Cal objects are allowed to be
            populated directly to the calendar manager.",
            ));
        }

        let mut w = NAMED_CALENDARS.write().unwrap();
        if w.contains_key(&k) {
            return Err(PyKeyError::new_err(
                "`name` already exists in calendars.
            Cannot overwrite, first `pop` the existing calendar.",
            ));
        }
        w.insert(k, Arc::new(CalWrapper::Cal(calendar)));
        Ok(())
    }

    /// Remove an existing [`Calendar`] from the calendar manager.
    pub fn pop(&self, name: &str) -> Result<Calendar, PyErr> {
        let k: String = sort_calendar_names(name);
        let popped = remove_any_calendar(&k);
        match popped {
            Some(arc) => match &*arc {
                CalWrapper::Cal(c) => {
                    remove_all_combinations(&k);
                    Ok(Calendar::Cal(c.clone()))
                }
                CalWrapper::UnionCal(c) => Ok(Calendar::UnionCal(c.clone())),
            },
            None => Err(PyKeyError::new_err("`name` does not exist in calendars.")),
        }
    }

    /// Return a [`NamedCal`] matching the name that is stored in the calendar manager.
    ///
    /// If the name as a key does not exist then an error will result.
    pub fn get(&self, name: &str) -> Result<NamedCal, PyErr> {
        let k: String = sort_calendar_names(name);
        let r = NAMED_CALENDARS.read().unwrap();
        let v = r.get(&k);
        match v {
            Some(arc_ref) => Ok(NamedCal {
                name: k,
                inner: arc_ref.clone(),
            }),
            None => Err(PyKeyError::new_err("`name` does not exist in calendars.")),
        }
    }

    /// Return a [`NamedCal`] matching the name that is stored in the calendar manager.
    ///
    /// If the name as a key does not exist but a [`UnionCal`] as a combination of [`Cal`] can
    /// be created, the HashMap will be updated with a new entry and the relevant [`NamedCal`]
    /// returned.
    pub fn get_with_insert(&self, name: &str) -> Result<NamedCal, PyErr> {
        let k: String = sort_calendar_names(name);
        if !k.chars().any(|c| c == ',' || c == '|') {
            // then lookup is for a single calendar, no composition necessary
            self.get(&k)
        } else {
            let item = self.get(&k);
            match item {
                Ok(value) => Ok(value), // key is found pre-populated in HashMap
                Err(_) => {
                    // then the calendars might need to be composited and inserted
                    let data = extract_individual_calendars(&k)?;
                    let _ = insert_union_cal(
                        &k,
                        UnionCal {
                            calendars: data.0,
                            settlement_calendars: data.1,
                        },
                    );
                    self.get(&k)
                }
            }
        }
    }
}

// Take an input string (potentially with comma and pipe) and convert to lower case and
// order the specific calendar names. See test_sort_calendar_names.
fn sort_calendar_names(name: &str) -> String {
    let stripped: String = name.chars().filter(|c| !c.is_whitespace()).collect();
    let parts: Vec<String> = stripped
        .to_lowercase()
        .split("|")
        .map(String::from)
        .collect();
    let mut reordered_parts: Vec<String> = Vec::new();
    for part in parts {
        let mut cals: Vec<String> = part.split(",").map(String::from).collect();
        cals.sort();
        reordered_parts.push(cals.join(","))
    }
    reordered_parts.join("|")
}

// Take an input string (potentially with comma and pipe) and extract the ordered list
// of individual, expected [`Cal`] objects. `k` is expected to be cleaned (sorted, lowercase etc.)
fn extract_individual_calendars(k: &str) -> Result<(Vec<Cal>, Option<Vec<Cal>>), PyErr> {
    let nc = CalendarManager::new();
    let parts: Vec<String> = k.split("|").map(String::from).collect();
    let mut container: Vec<Vec<Cal>> = Vec::new();
    for part in &parts {
        let cal_names: Vec<String> = part.split(",").map(String::from).collect();

        let named_cals: Vec<NamedCal> = cal_names
            .iter()
            .map(|k| nc.get(k))
            .collect::<Result<Vec<_>, _>>()?;

        let cals: Vec<Cal> = named_cals
            .iter()
            .map(|n| match &*n.inner {
                CalWrapper::Cal(value) => Ok(value.clone()),
                _ => Err(PyValueError::new_err(
                    "Individual calendar name is not a Cal object.",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        container.push(cals);
    }
    if container.len() == 1 {
        Ok((container[0].clone(), None))
    } else if parts.len() == 2 {
        Ok((container[0].clone(), Some(container[1].clone())))
    } else {
        Err(PyValueError::new_err(
            "The calendar cannot be parsed. Is there more than one pipe character?",
        ))
    }
}

// Insert a named calendar to the HashMap
fn insert_union_cal(k: &str, u: UnionCal) -> Option<Arc<CalWrapper>> {
    // returns None when inserted correctly
    let mut w = NAMED_CALENDARS.write().unwrap();
    w.insert(k.to_string(), Arc::new(CalWrapper::UnionCal(u)))
}

// Remove a key and return the object
fn remove_any_calendar(k: &str) -> Option<Arc<CalWrapper>> {
    let mut w = NAMED_CALENDARS.write().unwrap();
    w.remove(&k.to_string())
}

// Remove all other combinations that is a UnionCal and contains the name 'k'.
fn remove_all_combinations(k: &str) -> () {
    let mut w = NAMED_CALENDARS.write().unwrap();
    let keys: Vec<String> = w
        .iter()
        .filter(|(key, v)| key.contains(k) && is_union_cal((*v).clone()))
        .map(|(key, _)| key.to_string())
        .collect();
    for key in keys.into_iter() {
        let _ = w.remove(&key);
    }
}

fn is_union_cal(v: Arc<CalWrapper>) -> bool {
    match *v {
        CalWrapper::Cal(_) => false,
        CalWrapper::UnionCal(_) => true,
    }
}

// UNIT TESTS
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_calendar_names() {
        let result = sort_calendar_names("tgt,NYC,  ldn|tyo, tro");
        assert_eq!(result, "ldn,nyc,tgt|tro,tyo");

        let result = sort_calendar_names("tgt,NYC,  ldn|tyo");
        assert_eq!(result, "ldn,nyc,tgt|tyo");

        let result = sort_calendar_names("tgt,NYC,  ldn  ");
        assert_eq!(result, "ldn,nyc,tgt");

        let result = sort_calendar_names("tgt|ldn  ");
        assert_eq!(result, "tgt|ldn");

        let result = sort_calendar_names("tgt  ");
        assert_eq!(result, "tgt");

        let result = sort_calendar_names("a2, a1 | a3 ");
        assert_eq!(result, "a1,a2|a3");
    }

    #[test]
    fn test_extract_individual_calendars() {
        let nc = CalendarManager::new();
        let result = extract_individual_calendars("ldn").unwrap();
        let expected = nc.get("ldn").unwrap();
        assert_eq!(result.0[0], expected);

        let a1 = Cal::new(vec![], vec![1]);
        let a2 = Cal::new(vec![], vec![2]);
        let a3 = Cal::new(vec![], vec![3]);
        let _ = nc.add("a1", a1);
        let _ = nc.add("a2", a2);
        let _ = nc.add("a3", a3);

        let result = extract_individual_calendars("a2, a1 | a3").unwrap();
        let expected = (
            vec![Cal::new(vec![], vec![2]), Cal::new(vec![], vec![1])],
            Some(vec![Cal::new(vec![], vec![3])]),
        );
        assert_eq!(result, expected)
    }

    #[test]
    fn test_get_with_insert() {
        let nc = CalendarManager::new();
        let result = nc.get_with_insert("ldn").unwrap();
        let result2 = nc.get("ldn").unwrap();
        assert_eq!(result, result2);
        assert!(Arc::ptr_eq(&result.inner, &result2.inner));
    }

    #[test]
    fn test_get_with_insert_composite() {
        let nc = CalendarManager::new();
        let result = nc.get_with_insert("ldn,tgt").unwrap();
        let result2 = nc.get("ldn,tgt").unwrap();
        let result3 = nc.get("tgt,ldn").unwrap();
        assert_eq!(result, result2);
        assert!(Arc::ptr_eq(&result.inner, &result2.inner));
        assert_eq!(result, result3);
        assert!(Arc::ptr_eq(&result.inner, &result3.inner));
    }

    #[test]
    fn test_remove_composites_calendars() {
        let nc = CalendarManager::new();

        let a1 = Cal::new(vec![], vec![1]);
        let a2 = Cal::new(vec![], vec![2]);
        let a3 = Cal::new(vec![], vec![3]);
        let _ = nc.add("a1", a1);
        let _ = nc.add("a2", a2);
        let _ = nc.add("a3", a3);
        let _ = nc.get_with_insert("a1,a2");
        let _ = nc.get_with_insert("a1,a3");
        let _ = nc.get_with_insert("a2,a3");
        let _ = nc.get_with_insert("a1,a2,a3");

        let _ = nc.pop("a1");
        assert!(!nc.keys().contains(&"a1,a2".to_string()));
        assert!(!nc.keys().contains(&"a1,a3".to_string()));
        assert!(nc.keys().contains(&"a2,a3".to_string()));
        assert!(!nc.keys().contains(&"a1,a2,a3".to_string()));
    }
}
