use std::ops;
use ndarray::{arr1};
//
// #[derive(Clone, Debug)]
// pub struct MyType {
//     a : String,
//     b: f64
// }
//
// impl ops::Add<MyType> for MyType {
//     type Output = MyType;
//     fn add(self, other: MyType) -> MyType {
//         MyType {a: [self.a, other.a].join(""), b: self.b + other.b}
//     }
// }
//
// impl ops::Mul<MyType> for MyType {
//     type Output = MyType;
//     fn mul(self, other: MyType) -> MyType {
//         MyType {a: [self.a, other.a].join(""), b: self.b * other.b}
//     }
// }
//
fn main() {
    let arr_1 = arr1(&[1,2,3,4,5]);
    // println!("{:?}", arr_1[1..3]);
    // println!("array mul {:?}", arr_1.dot(&arr_2));
}
