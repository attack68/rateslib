use std::collections::HashSet;

use indexmap::indexset;

fn main() {
    let set1 = indexset!{"a", "b"};
    let set2 = indexset!{"b", "c"};

    // // "a" is the first value
    // assert_eq!(set.iter().next(), Some(&"a"));

    println!("{:?}", set1.union(&set2))
}