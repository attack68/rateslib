use crate::dual::{MathFuncs, Number};
use num_traits::Pow;
use pyo3::{pyfunction, PyErr};

#[pyfunction]
pub(crate) fn _sabr_x0(
    k: Number,
    f: Number,
    _t: Number,
    a: Number,
    b: f64,
    _p: Number,
    _v: Number,
    derivative: u8,
) -> Result<(Number, Option<Number>), PyErr> {
    // X0 = a / ((fk)^((1-b)/2) * (1 + (1-b)^2/24 ln^2(f/k) + (1-b)^4/1920 ln^4(f/k) )
    //If ``derivative`` is 1 also returns dX0/dk, calculated using sympy.
    //If ``derivative`` is 2 also returns dX0/df, calculated using sympy.
    let x0 = 1_f64 / &k;
    let x1 = 1_f64 / 24_f64 - b / 24_f64;
    let x2 = (&f * &x0).log();
    let x3 = (1_f64 - b).pow(4_f64);
    let x4 = x1 * (&x2).pow(2_f64) + (&x2).pow(4_f64) * x3 / 1920_f64 + 1_f64;
    let x5 = b / 2_f64 - 0.5_f64;
    let x6 = &a * (&f * &k).pow(x5);

    let x = &x6
        / ((&x2).pow(4_f64) * (1_f64 - b).pow(4_f64) / 1920_f64
            + (&x2).pow(2_f64) * (1_f64 / 24_f64 - b / 24_f64)
            + 1_f64);

    let dx: Option<Number> = match derivative {
        1 => Some(
            &x0 * x5 * &x6 / &x4
                + &x6 * (2_f64 * &x0 * x1 * &x2 + &x0 * (&x2).pow(3_f64) * x3 / 480_f64)
                    / x4.pow(2_f64),
        ),
        2 => {
            let y0 = b - 1_f64;
            let y2 = (&y0).pow(2_f64) * (&x2).pow(2_f64);
            let y3 = (&y0).pow(4_f64) * (&x2).pow(4_f64) + 80_f64 * &y2 + 1920_f64;
            Some(
                960_f64 * &a * y0 * (&f * &k).pow(x5) * (-8_f64 * y0 * x2 * (&y2 + 40_f64) + &y3)
                    / (f * y3.pow(2_f64)),
            )
        }
        _ => None,
    };

    Ok((x, dx))
}
