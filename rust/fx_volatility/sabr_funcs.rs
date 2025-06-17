use crate::dual::linalg::fouter11_;
use crate::dual::{Dual, Dual2, MathFuncs, Number, Vars};

use num_traits::{Pow, Signed};
use pyo3::{pyfunction, PyErr};
use std::sync::Arc;

#[pyfunction]
pub(crate) fn _sabr_x0(
    k: Number,
    f: Number,
    _t: Number,
    a: Number,
    b: Number,
    _p: Number,
    _v: Number,
    derivative: u8,
) -> Result<(Number, Option<Number>), PyErr> {
    // X0 = a / ((fk)^((1-b)/2) * (1 + (1-b)^2/24 ln^2(f/k) + (1-b)^4/1920 ln^4(f/k) )
    //If ``derivative`` is 1 also returns dX0/dk, calculated using sympy.
    //If ``derivative`` is 2 also returns dX0/df, calculated using sympy.
    let x0 = 1_f64 / &k;
    let x1 = 1_f64 / 24_f64 - &b / 24_f64;
    let x2 = (&f * &x0).log();
    let x3 = (1_f64 - &b).pow(4_f64);
    let x4 = &x1 * (&x2).pow(2_f64) + (&x2).pow(4_f64) * &x3 / 1920_f64 + 1_f64;
    let x5 = &b / 2_f64 - 0.5_f64;
    let x6 = &a * (&f * &k).pow(&x5);

    let x = &x6
        / ((&x2).pow(4_f64) * (1_f64 - &b).pow(4_f64) / 1920_f64
            + (&x2).pow(2_f64) * (1_f64 / 24_f64 - &b / 24_f64)
            + 1_f64);

    let dx: Option<Number> = match derivative {
        1 => Some(
            &x0 * x5 * &x6 / &x4
                + &x6 * (2_f64 * &x0 * x1 * &x2 + &x0 * (&x2).pow(3_f64) * x3 / 480_f64)
                    / x4.pow(2_f64),
        ),
        2 => {
            let y0 = &b - 1_f64;
            let y2 = (&y0).pow(2_f64) * (&x2).pow(2_f64);
            let y3 = (&y0).pow(4_f64) * (&x2).pow(4_f64) + 80_f64 * &y2 + 1920_f64;
            Some(
                960_f64 * &a * &y0 * (&f * &k).pow(x5) * (-8_f64 * y0 * x2 * (&y2 + 40_f64) + &y3)
                    / (f * y3.pow(2_f64)),
            )
        }
        _ => None,
    };

    Ok((x, dx))
}

#[pyfunction]
pub(crate) fn _sabr_x1(
    k: Number,
    f: Number,
    t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: u8,
) -> Result<(Number, Option<Number>), PyErr> {
    let x0 = 1_f64 / &k;
    let x1 = &b / 2_f64 - 0.5_f64;
    let x2 = &f * &k;
    let x3 = &b - 1_f64;
    let x = &t
        * ((&a).pow(2_f64) * (&x2).pow(&x3) * (&x3).pow(2_f64) / 24_f64
            + 0.25_f64 * &a * &b * &p * &v * (&x2).pow(&x1)
            + (&v).pow(2_f64) * (2_f64 - 3_f64 * (&p).pow(2_f64)) / 24_f64)
        + 1_f64;

    let dx: Option<Number> = match derivative {
        1 => Some(
            &t * ((&a).pow(2_f64) * &x0 * (&x2).pow(&x3) * (&x3).pow(3_f64) / 24_f64
                + 0.25 * &a * &b * p * &v * x0 * &x1 * x2.pow(x1)),
        ),
        2 => Some(
            &a * &t
                * &x3
                * (&a * (&x3).pow(2_f64) * (&x2).pow(x3) + 3_f64 * b * p * v * x2.pow(x1))
                / (24_f64 * f),
        ),
        _ => None,
    };

    Ok((x, dx))
}

#[pyfunction]
pub(crate) fn _sabr_x2(
    k: Number,
    f: Number,
    _t: Number,
    a: Number,
    b: Number,
    p: Number,
    v: Number,
    derivative: u8,
) -> Result<(Number, Option<Number>), PyErr> {
    let x0 = 1_f64 / &k;
    let x1 = (&f * &x0).log();
    let x2 = 1_f64 / &a;
    let x3 = &f * &k;
    let x4 = &b / 2_f64 - 0.5_f64;
    let x5 = (&x3).pow(-&x4);
    let x6 = &v * &x2 * &x5;

    let z = &x6 * &x1;
    let chi = (((1_f64 - 2_f64 * &p * &z + &z * &z).pow(0.5_f64) + &z - &p) / (1_f64 - &p)).log();

    let x: Number;
    if z.abs() > 1e-15_f64 {
        x = &z / &chi;
    } else {
        // handle the undefined quotient case when f=k by directly specifying dual numbers
        let p_f64 = f64::from(&p);

        x = match &z {
            Number::F64(_z) => Number::F64(1_f64),
            Number::Dual(z_) => Number::Dual(Dual {
                real: 1_f64,
                dual: &z_.dual * p_f64 * -0.5_f64,
                vars: Arc::clone(&z_.vars),
            }),
            Number::Dual2(z_) => {
                let (z_cast, p_cast): (Dual2, Dual2) = match &p {
                    Number::F64(p_) => {
                        let temp = Dual2::new_from(z_, *p_, vec![]);
                        z_.to_union_vars(&temp, None)
                    }
                    Number::Dual(_) => panic!("Unexpected Dual/Dual2 type crossing in _sabr_x2"),
                    Number::Dual2(p_) => z_.to_union_vars(p_, None),
                };
                let f_z = -0.5_f64 * p_f64;
                // f_p = 0.0
                let f_zz = (2_f64 - 3_f64 * p_f64 * p_f64) / 6_f64;
                let f_zp = -0.5_f64;
                // f_pp = 0.0

                let mut dual2 = f_z * &z_cast.dual2.clone();
                dual2 =
                    dual2 + 0.5_f64 * f_zz * fouter11_(&z_cast.dual.view(), &z_cast.dual.view());
                // dual2 += 0.5 * f_pp * np.outer(p_.dual, p_.dual)
                dual2 = dual2
                    + 0.5_f64
                        * f_zp
                        * (fouter11_(&z_cast.dual.view(), &p_cast.dual.view())
                            + fouter11_(&p_cast.dual.view(), &z_cast.dual.view()));
                Number::Dual2(Dual2 {
                    real: 1_f64,
                    vars: Arc::clone(&z_cast.vars),
                    dual: &z_cast.dual * p_f64 * -0.5_f64,
                    dual2,
                })
            }
        };
    }

    let dx: Option<Number>;
    match derivative {
        1 => {
            if z.abs() > 1e-15_f64 {
                let x7 = &x1 * &x6;
                let x8 = &p * &x7;
                let x9 = (&x1).pow(2_f64);
                let x10 = (&a).pow(-2_f64);
                let x11 = (&v).pow(2_f64);
                let x12 = &b - 1_f64;
                let x13 = (&x3).pow(-&x12);
                let x14 = &x10 * &x11 * &x13;
                let x15 = (&x14 * &x9 - 2_f64 * &x8 + 1_f64).pow(0.5_f64);
                let x16 = -&p + &x15 + &x7;
                let x17 = (&x16 / (1_f64 - &p)).log();
                let x18 = 1_f64 / &x17;
                let x19 = &x0 * &x6;
                let x20 = -&x4;
                let x21 = 1.0 * &x0;

                dx = Some(
                    &v * &x0 * &x1 * &x18 * &x2 * &x20 * &x5
                        - &x18 * &x19
                        - &x7
                            * (&x0 * &x20 * &x7 - x19
                                + (1.0_f64 * p * v * &x0 * x2 * x5
                                    - 0.5_f64 * x0 * x10 * x11 * x12 * x13 * x9
                                    - x1 * x14 * &x21
                                    - x20 * x21 * x8)
                                    / x15)
                            / (x16 * (&x17).pow(2_f64)),
                )
            } else {
                let dx_dz = _sabr_dx2_dz(&z, &p);

                let y0 = 1_f64 / &k;
                let y1 = &b / 2_f64 - 0.5_f64;
                let y2 = &v * &x0 / (&a * (&f * &k).pow(&y1));
                let dz = -y2 * (y1 * (&f * y0).log() + 1_f64);

                dx = Some(dx_dz * dz);
            }
        }
        2 => {
            if z.abs() > 1e-15_f64 {
                let y0 = (&a).pow(2_f64);
                let y1 = 1_f64 / &y0;
                let y3 = (&x3).pow(-x4);
                let y4 = &a * &p;
                let y6 = &v * &x1;
                let y7 = &y3 * &y6;
                let y8 = &b - 1_f64;
                let y9 = (&x3).pow(-&y8);
                let y10 =
                    (&y1 * (&v * &v * &x1 * &x1 * &y9 + &y0 - 2_f64 * &y4 * &y7)).pow(0.5_f64);
                let y11 = &a * (-&p + &y10) + &y7;
                let y12 = &a * &y10;
                let y13 = ((&a * &p - &y12 - &y7) / (&a * (&p - 1_f64))).log();
                let y14 = &x1 * &y8 - 2_f64;
                let y15 = -&y14;

                dx = Some(
                    &v * &y1
                        * &y3
                        * (&y11 * &y12 * &y13 * &y15
                            + &y6 * (&y12 * &y14 * &y3 + &y14 * &y6 * &y9 + &y15 * &y3 * &y4))
                        / (2_f64 * &f * &y10 * &y11 * (&y13).pow(2_f64)),
                )
            } else {
                let dx_dz = _sabr_dx2_dz(&z, &p);
                let dz = &v * &x5 * (-(&b - 1_f64) * &x1 + 2_f64) / (2_f64 * &a * &f);
                dx = Some(dx_dz * dz);
            }
        }
        _ => dx = None,
    };

    Ok((x, dx))
}

fn _sabr_dx2_dz(z: &Number, p: &Number) -> Number {
    let p_f64 = f64::from(p);
    match z {
        Number::F64(_) => Number::F64(-p_f64 / 2_f64),
        Number::Dual(z_) => {
            let (z_cast, p_cast): (Dual, Dual) = match &p {
                Number::F64(p_) => {
                    let temp = Dual::new_from(z_, *p_, vec![]);
                    z_.to_union_vars(&temp, None)
                }
                Number::Dual(p_) => z_.to_union_vars(p_, None),
                Number::Dual2(_) => panic!("Unexpected Dual/Dual2 type crossing in _sabr_x2"),
            };
            let mut dual = -0.5_f64 * &p_cast.dual;
            dual = dual + (2_f64 - 3_f64 * p_f64 * p_f64) / 6_f64 * &z_cast.dual;
            Number::Dual(Dual {
                real: -0.5_f64 * p_f64,
                vars: Arc::clone(&z_cast.vars),
                dual,
            })
        }
        Number::Dual2(z_) => {
            let (z_cast, p_cast): (Dual2, Dual2) = match &p {
                Number::F64(p_) => {
                    let temp = Dual2::new_from(z_, *p_, vec![]);
                    z_.to_union_vars(&temp, None)
                }
                Number::Dual(_) => panic!("Unexpected Dual/Dual2 type crossing in _sabr_x2"),
                Number::Dual2(p_) => z_.to_union_vars(p_, None),
            };
            let mut dual = -0.5_f64 * &p_cast.dual;
            dual = dual + (2_f64 - 3_f64 * p_f64 * p_f64) / 6_f64 * &z_cast.dual;
            let mut dual2 = (2_f64 - 3_f64 * p_f64 * p_f64) / 6_f64 * &z_cast.dual2;
            dual2 = dual2 - 0.5_f64 * &p_cast.dual2;
            dual2 = dual2
                + p_f64 * (5_f64 - 6_f64 * p_f64 * p_f64) / 8_f64
                    * fouter11_(&z_cast.dual.view(), &z_cast.dual.view());
            dual2 = dual2
                - 0.5_f64
                    * p_f64
                    * (fouter11_(&z_cast.dual.view(), &p_cast.dual.view())
                        + fouter11_(&p_cast.dual.view(), &z_cast.dual.view()));
            Number::Dual2(Dual2 {
                real: -0.5_f64 * p_f64,
                vars: Arc::clone(&z_cast.vars),
                dual: dual,
                dual2: dual2,
            })
        }
    }
}
