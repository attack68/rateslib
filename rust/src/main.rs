extern crate ndarray;

use std::ops;
use ndarray::{array, Array1, Array};
use indexmap::indexset;
use indexmap::set::IndexSet;

#[derive(Debug)]
struct Dual<'a> {
    vars : IndexSet<&'a str>,
    real : f64,
    dual : Array1<f64>,
}

impl Dual<'_> {
    fn upcast_combined<'a>(&'a self, other: &'a Dual) -> (Dual<'a>, Dual<'a>) {
        // check the set of vars in each Dual are equivalent
        if self.vars.intersection(&other.vars).count() == self.vars.len() {
            println!("maybe only need reorder 1");
            (self.clone(), other.upcast_vars(&self.vars))
        } else {
            println!("must reorder both");
            (
                Dual { vars: indexset!{"a", "b"}, real: 2.0, dual: array![2.0, 9.0]},
                Dual { vars: indexset!{"a", "b"}, real: 2.0, dual: array![2.0, 9.0]}
            )
        }
    }

    fn upcast_vars<'a>(&'a self, new_vars: &'a IndexSet<&'a str>) -> Dual<'a> {
        // if self.vars.intersection(new_vars).count() == self.vars.len()
        //     &&
        if self.vars.iter().zip(new_vars.iter()).filter(|&(a, b)| a == b).count() == self.vars.len()
        {
           self.clone()
        } else {
            let vars = new_vars.clone();
            let real = self.real.clone() + 1000.0;
            let mut dual = Array::zeros(vars.len());
            for (i, var) in new_vars.iter().enumerate() {
                let index = self.vars.get_index_of(var);
                match index {
                    Some(value) => {
                        dual[[i]] = self.dual[[value]]
                    }
                    None => {}
                }
            }
            Dual {vars: vars, real: real, dual: dual}
        }
    }

    fn clone<'a>(&'a self) -> Dual<'a> {
        Dual {
            vars: self.vars.clone(),
            real: self.real.clone(),
            dual: self.dual.clone(),
        }
    }
}

impl<'a> ops::Add<f64> for Dual<'a> {
    type Output = Dual<'a>;
    fn add(self, other: f64) -> Dual<'a> {
        Dual {
            vars: self.vars.clone(),
            real: self.real + other,
            dual: self.dual.clone()
        }
    }
}

impl<'a> ops::Sub<f64> for Dual<'a> {
    type Output = Dual<'a>;
    fn sub(self, other: f64) -> Dual<'a> {
        Dual {
            vars: self.vars.clone(),
            real: self.real - other,
            dual: self.dual.clone()
        }
    }
}

impl<'a> ops::Add<Dual<'_>> for Dual<'a> {
    type Output = Dual<'a>;
    fn add(self, other: Dual) -> Dual<'a> {
        Dual {
            real: self.real + other.real,
            dual: self.dual + other.dual,
            vars: self.vars.clone()
        }
    }
}

impl<'a> ops::Sub<Dual<'_>> for Dual<'a> {
    type Output = Dual<'a>;
    fn sub(self, other: Dual) -> Dual<'a> {
        Dual {
            real: self.real - other.real,
            dual: self.dual - other.dual,
            vars: self.vars.clone()
        }
    }
}

fn main() {
    let d1 = Dual { vars: indexset!{"a", "b"}, real: 2.0, dual: array![2.0, 9.0]};
    let d2 = Dual { vars: indexset!{"b", "a"}, real: 5.0, dual: array![3.0, 1.0]};

    let (d3, d4) = d1.upcast_combined(&d2);
    println!("{:?}", d3);
    println!("{:?}", d4)

    // let x = array![1,2,3];
    // let y = array![2,3,4];
    // let z = x.dot(&y);
    // println!("{}{}{}", x, y, z);
}

// enum Point {
//     Dual,
//     f64,
// }