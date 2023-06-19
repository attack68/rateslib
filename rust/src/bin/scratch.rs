use std::ops;
use ndarray::{arr1};

#[derive(Clone, Debug)]
pub struct MyType {
    a : String,
    b: f64
}

impl ops::Add<MyType> for MyType {
    type Output = MyType;
    fn add(self, other: MyType) -> MyType {
        MyType {a: [self.a, other.a].join(""), b: self.b + other.b}
    }
}

impl ops::Mul<MyType> for MyType {
    type Output = MyType;
    fn mul(self, other: MyType) -> MyType {
        MyType {a: [self.a, other.a].join(""), b: self.b * other.b}
    }
}

fn main() {
    let arr_1 = arr1(&[
        MyType {a: "A".to_string(), b: 2.0}, MyType {a: "B".to_string(), b: 3.0}
    ]);
    let arr_2 = arr1(&[
        MyType {a: "C".to_string(), b: 2.0}, MyType {a: "D".to_string(), b: 4.0}
    ]);
    println!("{:?}", arr_1.clone() + arr_2.clone());
    println!("{:?}", arr_1.sum());
    // println!("array mul {:?}", arr_1.dot(&arr_2));
}
