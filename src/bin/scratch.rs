
struct A<T>
{
    x: Vec<T>,
}

impl<T> A<T>
where T: std::fmt::Debug
{
    fn print(&self, y: Vec<T>) {
        for (a, b) in self.x.iter().zip(y) {
            println!("{:?},{:?}", a, b);
        }
    }
}

fn main() {
    let a = A { x: vec![1.0, 2.0]};
    a.print(vec![4.2, 5.2]);
}
