use std::time::{Duration, SystemTime};
use rateslibrs::curves::interpolation::index_left;

fn main() {

    let list: Vec<f64> = (0..100).map(f64::from).collect();

    let now = SystemTime::now();
    let mut result = 0;
    for i in 0..1000000 {
        result = index_left(&list[..], &25_f64, None);
    }
    println!("{:.5?} seconds", now.elapsed());
    println!("{}", result);
}
