// use std::time::{Duration, SystemTime};
use chrono::prelude::*;
use chrono::Days;

fn main() {
    //     let list: Vec<f64> = (0..100).map(f64::from).collect();
    //     let now = SystemTime::now();
    //     let mut result = 0;
    //     for i in 0..1000000 {
    //         result = index_left(&list[..], &25_f64, None);
    //     }
    //     println!("{:.5?} seconds", now.elapsed());
    //     println!("{}", result);

    let date = NaiveDateTime::parse_from_str("2015-09-05 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let date2 = date + Days::new(1);
    println!("{:?}", date2)
}
