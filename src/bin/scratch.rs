// use chrono::naive::NaiveDate;
// use rateslibrs::interpolate::{interpolate_with_method};
// use dual::Duals;
// use std::ops::Add;

// #[derive(Debug, Clone)]
// pub struct Test {
//     pub a: f64,
//     pub b: f64,
// }
//
#[derive(Debug, Clone)]
enum Fs {
    F64(f64),
    F32(f32),
}

fn add_one(x: Fs) -> Fs {
    match x {
        Fs::F64(v) => v + 1.0_f64,
        Fs::F32(v) => v + 1.0_f32
    }
}

fn main() {
//     let x_1 = NaiveDate::from_ymd_opt(2003, 12, 2).unwrap();

    let x = 32.5;
    let y = add_one(Fs(x));
    println!("{:?}", y)
}
