use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rateslibrs::dual::dual1::Dual;
use rateslibrs::dual::Duals;

use indexmap::set::IndexSet;
use ndarray::Array;
use std::rc::Rc;

fn dual_add_bm(a: &Dual, b: &Dual) -> Dual {
    a + b
}

fn duals_add_bm(a: &Duals, b: &Duals) -> Duals {
    a + b
}

fn dual_sub_bm(a: &Dual, b: &Dual) -> Dual {
    a - b
}

fn dual_mul_bm(a: &Dual, b: &Dual) -> Dual {
    a * b
}

fn dual_div_bm(a: &Dual, b: &Dual) -> Dual {
    a / b
}

fn float_add_bm(a: &f64, b: &f64) -> f64 {
    a + b
}

fn criterion_benchmark(c: &mut Criterion) {
    let dual_ = Array::ones(1000);
    let vars = IndexSet::from_iter((0..1000).map(|x| format!("v{}", x).to_string()));
    let dual_2 = Array::ones(1000);
    let vars2 = IndexSet::from_iter((0..1000).map(|x| format!("u{}", x).to_string()));

    let a = Dual {
        real: 2.0,
        vars: Rc::new(vars),
        dual: dual_,
    };
    let b = Dual {
        real: 3.0,
        vars: Rc::new(vars2),
        dual: dual_2,
    };
    let x = 20.1;
    let y = 22.1;

    let duals_ = Array::ones(1000);
    let duals_2 = Array::ones(1000);
    let vars = IndexSet::from_iter((0..1000).map(|x| format!("v{}", x).to_string()));
    let vars2 = IndexSet::from_iter((0..1000).map(|x| format!("u{}", x).to_string()));
    let a2 = Duals::Dual(Dual {
        real: 2.0,
        vars: Rc::new(vars),
        dual: duals_,
    });
    let b2 = Duals::Dual(Dual {
        real: 3.0,
        vars: Rc::new(vars2),
        dual: duals_2,
    });
    let x2 = Duals::Float(20.1);
    let y2 = Duals::Float(22.1);

    c.bench_function("float add", |z| z.iter(|| float_add_bm(&x, &y)));
    c.bench_function("float add duals", |z| z.iter(|| duals_add_bm(&x2, &y2)));
    c.bench_function("dual add diff vars", |z| z.iter(|| dual_add_bm(&a, &b)));
    c.bench_function("duals add diff vars", |z| z.iter(|| duals_add_bm(&a2, &b2)));
    c.bench_function("dual sub diff vars", |z| z.iter(|| dual_sub_bm(&a, &b)));
    c.bench_function("dual mul diff vars", |z| z.iter(|| dual_mul_bm(&a, &b)));
    c.bench_function("dual div diff vars", |z| z.iter(|| dual_div_bm(&a, &b)));
    c.bench_function("dual add same ptr vars", |z| {
        z.iter(|| dual_add_bm(&a, &a.clone()))
    });
    c.bench_function("duals add same ptr vars", |z| {
        z.iter(|| duals_add_bm(&a2, &a2.clone()))
    });
    c.bench_function("dual sub same ptr vars", |z| {
        z.iter(|| dual_sub_bm(&a, &a.clone()))
    });
    c.bench_function("dual mul same ptr vars", |z| {
        z.iter(|| dual_mul_bm(&a, &a.clone()))
    });
    c.bench_function("dual div same ptr vars", |z| {
        z.iter(|| dual_div_bm(&a, &a.clone()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
