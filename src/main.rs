
pub mod dual;
use dual::dual1::Dual;
use ndarray::{Array1, Array};

fn main() {
    let d1 = Dual::new(
        1.1,
        Vec::from_iter((0..1000).map(|x| x.to_string())),
        Vec::from_iter((0..1000).map(|x| f64::from(x)))
    );

    use std::time::Instant;
    let now = Instant::now();

    // Code block to measure.
    {
        for i in 0..100000 {
            let y;
            y = &d1 + &d1;
        }
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed / 100000);
}
