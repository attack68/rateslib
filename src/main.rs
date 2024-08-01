
use std::time::SystemTime;
use rateslib::dual::{Dual, DualsOrF64, FieldOps};

fn ops<T>(a: &T, b: &T) -> T
where         T: FieldOps<T>,
  for<'a> &'a T: FieldOps<T>
{
    &(&(&(a + b) - a) * b) / a
}

fn ops2(a: f64, b: &Dual) -> Dual
{
    &(&(&(a + b) - a) * b) / a
}

fn main() {
    let a0 = 2.5_f64;
    let b0 = 3.5_f64;
    let a1 = Dual::new(2.5_f64, vec!["x".to_string(), "y".to_string()]);
    let b1 = Dual::new_from(&a1, 3.5_f64, vec!["x".to_string(), "y".to_string()]);
    let a2 = DualsOrF64::Dual(a1.clone());
    let b2 = DualsOrF64::Dual(b1.clone());
    let a3 = DualsOrF64::F64(2.5_f64);
    let b3 = DualsOrF64::F64(3.5_f64);

    let now = SystemTime::now();

    for i in 0..10000 {
        let _ = ops(&a0, &b0);
    }
    println!("{:.5?} time taken for f64", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a3, &b3);
    }
    println!("{:.5?} time taken for DualsOrF64 F64 wrapper", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a1, &b1);
    }
    println!("{:.5?} time taken for Dual", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a2, &b2);
    }
    println!("{:.5?} time taken for DualsOrF64 Dual wrapper", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a2, &a3);
    }
    println!("{:.5?} time taken for DualsOrF64 F64/Dual wrapper", now.elapsed());

    for i in 0..10000 {
        let _ = ops2(a0, &a1);
    }
    println!("{:.5?} time taken for F64/Dual special func", now.elapsed());

}
