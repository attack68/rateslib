use ndarray::arr2;

fn main() {
    let a = arr2(&[[1.2, 1.3], [2.3, 2.4]]);
    for v in a.iter() {
        println!("{}", v);
    }
}
