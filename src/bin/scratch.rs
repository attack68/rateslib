
fn main() {

    let (x, y): (usize, usize) = (1, 2);

    match (x, y) {
        (x, y) if x > 2 => println!("done"),
        (x, y) if x < 2 => println!("other"),
        _ => println!("none")
    }
}
