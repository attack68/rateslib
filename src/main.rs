pub mod dual;
use dual::dual1::{Dual};
// use dual::linalg::pivot_matrix;
// use ndarray::{arr2, s, Array2, Axis, Zip};
// use ndarray::{Array, Array1, Dimension};

fn main() {
    let d1 = Dual::new(
        1.1,
        Vec::from_iter((0..1000).map(|x| x.to_string())),
        Vec::from_iter((0..1000).map(f64::from)),
    );

    use std::time::Instant;
    let now = Instant::now();

    // Code block to measure.
    {
        for _i in 0..100000 {
            let _y = &d1 + &d1;
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
