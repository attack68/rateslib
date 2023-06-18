use indexmap::{indexset, IndexSet};
use ndarray::array;

fn main() {
    let x: IndexSet<String> = IndexSet::from_iter([]);
    let y: IndexSet<String> = IndexSet::from_iter(["a".to_string()]);
    println!("{:?}", x.union(&y));
}