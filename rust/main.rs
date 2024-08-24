use rateslib::dual::{Dual, Number, NumberOps};
use std::time::SystemTime;

fn ops<T>(a: &T, b: &T) -> T
where
    T: NumberOps<T>,
    for<'a> &'a T: NumberOps<T>,
{
    &(&(&(a + b) - a) * b) / a
}

fn ops2(a: f64, b: &Dual) -> Dual {
    &(&(&(a + b) - a) * b) / a
}

fn main() {
    let a0 = 2.5_f64;
    let b0 = 3.5_f64;
    let a1 = Dual::new(2.5_f64, vec!["x".to_string(), "y".to_string()]);
    let b1 = Dual::new_from(&a1, 3.5_f64, vec!["x".to_string(), "y".to_string()]);
    let a2 = Number::Dual(a1.clone());
    let b2 = Number::Dual(b1.clone());
    let a3 = Number::F64(2.5_f64);
    let b3 = Number::F64(3.5_f64);

    let now = SystemTime::now();

    for i in 0..10000 {
        let _ = ops(&a0, &b0);
    }
    println!("{:.5?} time taken for f64", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a3, &b3);
    }
    println!("{:.5?} time taken for Number F64 wrapper", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a1, &b1);
    }
    println!("{:.5?} time taken for Dual", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a2, &b2);
    }
    println!("{:.5?} time taken for Number Dual wrapper", now.elapsed());

    for i in 0..10000 {
        let _ = ops(&a2, &a3);
    }
    println!(
        "{:.5?} time taken for Number F64/Dual wrapper",
        now.elapsed()
    );

    for i in 0..10000 {
        let _ = ops2(a0, &a1);
    }
    println!("{:.5?} time taken for F64/Dual special func", now.elapsed());
}
