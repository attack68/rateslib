// use chrono::naive::NaiveDate;
// use rateslibrs::interpolate::{interpolate_with_method};
use rateslibrs::dual::Duals;
// use std::ops::Add;

// #[derive(Debug, Clone)]
// pub struct Test {
//     pub a: f64,
//     pub b: f64,
// }
//
// #[derive(Debug, Clone)]
// enum TestE {
//     A(f64),
//     B(Test),
// }
//
// impl Add for TestE {
//     type Output = Self;
//     fn add(self, other: Self) -> Self {
//         use TestE::*;
//         match self {
//             A(MT) => match other {
//                 A(MTO) => A(MT + MTO),
//                 B(MTO) => B(Test {a: 10.0, b: 0.0})
//             },
//             B(MT) => match other {
//                 A(MTO) => B(Test{a: 10.0, b: 20.0}),
//                 B(MTO) => B(Test{a: 12.0, b: 30.0})
//             }
//         }
//     }
// }
//
// fn add(a: TestE, b: TestE) -> TestE {
//     a + b
// }

fn main() {
    let x = Duals::Float(2.0);
    let y = Duals::Float(3.0);
    let z = x + y;
    println!("{:?}", z);
    // let x_1 = NaiveDate::from_ymd_opt(2003, 12, 2).unwrap();
    // let x_2 = NaiveDate::from_ymd_opt(2003, 12, 22).unwrap();
    // let x = NaiveDate::from_ymd_opt(2003, 12, 18).unwrap();
    //
    // let y_1 = Duals::Float(100.0);
    // let y_2 = Duals::Float(200.0);
    //
    // let z = interpolate_with_method(&x, &x_1, y_1, &x_2, y_2, "linear", None);
    // println!("{:?}", z);
    // println!("{}", x_1)
}
