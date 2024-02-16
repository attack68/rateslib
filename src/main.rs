
pub mod dual;
use dual::dual1::{Dual, DualOrF64};
// use dual::linalg::pivot_matrix;
use ndarray::{Array1, Array, Dimension};
use ndarray::{Array2, arr2, s, Zip, Axis};

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


    // let P: Array2<i32> = arr2(
    //     &[[1, 2, 3, 4],
    //       [10, 2, 5, 6],
    //       [7, 8, 1, 1],
    //       [2, 2, 2, 9]]
    // );
    // let (A, B) = pivot_matrix(&P);
    //
    // println!("{:?}", B);
}
