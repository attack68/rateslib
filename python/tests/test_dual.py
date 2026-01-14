# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

import math
from statistics import NormalDist

import numpy as np
import pytest
from packaging import version
from rateslib import IRS, Curve, FXRates, Solver, default_context, dt
from rateslib.dual import (
    Dual,
    Dual2,
    Variable,
    dual_exp,
    dual_inv_norm_cdf,
    dual_log,
    dual_norm_cdf,
    dual_norm_pdf,
    dual_solve,
    gradient,
    set_order,
)
from rateslib.dual.utils import _abs_float, _set_ad_order_objects

DUAL_CORE_PY = False


@pytest.fixture
def x_1():
    return Dual(1, vars=["v0", "v1"], dual=[1, 2])


@pytest.fixture
def x_2():
    return Dual(2, vars=["v0", "v2"], dual=[0, 3])


@pytest.fixture
def y_1():
    return Dual2(1, vars=["v0", "v1"], dual=[1, 2], dual2=[])


@pytest.fixture
def y_2():
    return Dual2(1, vars=["v0", "v1"], dual=[1, 2], dual2=[1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def y_3():
    return Dual2(2, vars=["v0", "v2"], dual=[0, 3], dual2=[1.0, 1.0, 1.0, 1.0])


@pytest.fixture
def A():
    return np.random.randn(25).reshape(5, 5)


@pytest.fixture
def A_sparse():
    return np.array(
        [
            [24, -36, 12, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.25, 0.583333333333, 0.1666666666, 0, 0, 0, 0, 0],
            [0, 0, 0.1666666666, 0.6666666666, 0.1666666666, 0, 0, 0, 0],
            [0, 0, 0, 0.1666666666, 0.6666666666, 0.1666666666, 0, 0, 0],
            [0, 0, 0, 0, 0.1666666666, 0.6666666666, 0.1666666666, 0, 0],
            [0, 0, 0, 0, 0, 0.1666666666, 0.583333333333, 0.25, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 12, -36, 24],
        ],
    )


@pytest.fixture
def b():
    return np.random.randn(5).reshape(5, 1)


def test_zero_init() -> None:
    x = Dual(1, ["x"], [])
    assert np.all(x.dual == np.ones(1))

    y = Dual2(1, ["x"], [], [])
    assert np.all(y.dual == np.ones(1))
    assert np.all(y.dual2 == np.zeros((1, 1)))


@pytest.mark.parametrize(
    "op",
    [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__eq__",
    ],
)
def test_no_type_crossing_on_ops(x_1, y_1, op) -> None:
    # getattr(x_1, op)(y_1)
    with pytest.raises(TypeError):
        getattr(x_1, op)(y_1)

    with pytest.raises(TypeError):
        getattr(y_1, op)(x_1)


def test_functions_of_two_duals_analytic_formula():
    # test the analytic formula for determining the resultant dual number of a function of
    # 2 dual numbers

    a = Dual2(2.0, ["a"], [], [])
    b = Dual2(3.0, ["b"], [], [])

    # z and p contain 2nd order manifolds
    z = a**2 * b  # = 12
    p = b**2 * a  # = 18
    p = Dual2.vars_from(z, p.real, p.vars, p.dual, np.ravel(p.dual2))

    # f is the actual expected result, calculated using dual number arithmetic
    expected = z**2 * p**3

    # result is pieced together using the analytic formula
    f_0 = 12**2 * 18**3
    f_z = 2 * 12 * 18**3
    f_p = 3 * 12**2 * 18**2
    f_zz = 2 * 18**3
    f_zp = 6 * 12 * 18**2
    f_pp = 6 * 12**2 * 18

    real = f_0
    dual = z.dual * f_z + p.dual * f_p
    dual2 = f_z * z.dual2 + f_p * p.dual2
    dual2 += 0.5 * f_zz * np.outer(z.dual, z.dual)
    dual2 += 0.5 * f_pp * np.outer(p.dual, p.dual)
    dual2 += 0.5 * f_zp * (np.outer(z.dual, p.dual) + np.outer(p.dual, z.dual))
    result = Dual2.vars_from(z, real, z.vars, dual, np.ravel(dual2))

    assert result == expected


def test_dual_repr(x_1, y_2) -> None:
    result = x_1.__repr__()
    assert result == "<Dual: 1.000000, (v0, v1), [1.0, 2.0]>"

    result = y_2.__repr__()
    assert result == "<Dual2: 1.000000, (v0, v1), [1.0, 2.0], [[...]]>"


@pytest.mark.skipif(not DUAL_CORE_PY, reason="Rust Dual does not format string in this way.")
def test_dual_str(x_1, y_2) -> None:
    result = x_1.__str__()
    assert result == " val = 1.00000000\n  dv0 = 1.000000\n  dv1 = 2.000000\n"

    result = y_2.__str__()
    assert (
        result == " val = 1.00000000\n"
        "  dv0 = 1.000000\n"
        "  dv1 = 2.000000\n"
        "dv0dv0 = 2.000000\n"
        "dv0dv1 = 2.000000\n"
        "dv1dv1 = 2.000000\n"
    )


@pytest.mark.parametrize(
    ("vars_", "expected"),
    [
        (["v0"], 1.00),
        (["v1", "v0"], np.array([2.0, 1.0])),
    ],
)
def test_gradient_method(vars_, expected, x_1, y_2) -> None:
    result = gradient(x_1, vars_)
    assert np.all(result == expected)

    result = gradient(y_2, vars_)
    assert np.all(result == expected)


def test_gradient_on_float():
    result = gradient(1.0, ["v0", "s"])
    assert np.all(result == np.array([0.0, 0.0]))

    result = gradient(1.0, ["s"], order=2)
    assert np.all(result == np.array([[0.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize(
    ("vars_", "expected"),
    [
        (["v0"], 2.00),
        (["v1", "v0"], np.array([[2.0, 2.0], [2.0, 2.0]])),
    ],
)
def test_gradient_method2(vars_, expected, y_2) -> None:
    result = gradient(y_2, vars_, 2)
    assert np.all(result == expected)


def test_rdiv_raises(x_1, y_1) -> None:
    with pytest.raises(TypeError):
        _ = "string" / x_1

    with pytest.raises(TypeError):
        _ = "string" / y_1


def test_neg(x_1, y_2) -> None:
    assert -x_1 == Dual(-1, ["v0", "v1"], [-1.0, -2.0])
    assert -y_2 == Dual2(-1, ["v0", "v1"], [-1.0, -2.0], [-1.0, -1.0, -1.0, -1.0])


def test_eq_ne(x_1, y_1, y_2) -> None:
    # non-matching types
    assert Dual(0, ["single_var"], []) != 0
    assert Dual2(0, ["single_var"], [], []) != 0
    # ints
    assert Dual(2, [], []) == 2
    assert Dual2(2, [], [], []) == 2
    # floats
    assert Dual(3.3, [], []) == 3.3
    assert Dual2(3.3, [], [], []) == 3.3
    # no type crossing
    with pytest.raises(TypeError):
        assert x_1 != y_1
    # equality
    assert x_1 == Dual(1, ["v0", "v1"], [1, 2])
    assert y_1 == Dual2(1, ["v0", "v1"], [1, 2], [])
    assert y_2 == Dual2(1, ["v0", "v1"], [1, 2], [1.0, 1.0, 1.0, 1.0])
    # non-matching elements
    assert x_1 != Dual(2, ["v0", "v1"], [1, 2])
    assert x_1 != Dual(1, ["v0", "v1"], [2, 2])
    assert x_1 != Dual(1, ["v2", "v1"], [1, 2])
    # non-matching elements
    assert y_1 != Dual2(2, ["v0", "v1"], [1, 2], [])
    assert y_1 != Dual2(1, ["v0", "v1"], [2, 2], [])
    assert y_1 != Dual2(1, ["v2", "v1"], [1, 2], [])
    # non-matching dual2
    assert y_2 != Dual2(1, ["v0", "v1"], [1, 2], [2.0, 2.0, 2.0, 2.0])


def test_lt() -> None:
    assert Dual(1, ["x"], []) < Dual(2, ["y"], [])
    assert Dual2(1, ["z"], [], []) < Dual2(2, ["x"], [], [])
    assert Dual(1, ["x"], []) < 10
    assert not Dual(1, ["x"], []) < 0


def test_lt_raises() -> None:
    with pytest.raises(TypeError, match="Cannot compare"):
        assert Dual(1, ["x"], []) < Dual2(2, ["y"], [], [])


def test_gt() -> None:
    assert Dual(2, ["x"], []) > Dual(1, ["y"], [])
    assert Dual2(2, ["z"], [], []) > Dual2(1, ["x"], [], [])
    assert Dual(1, ["x"], []) > 0
    assert not Dual(1, ["x"], []) > 10


def test_gt_raises() -> None:
    with pytest.raises(TypeError, match="Cannot compare"):
        assert Dual(2, ["x"], []) > Dual2(1, ["y"], [], [])


def test_dual2_abs_float(x_1, y_1, y_2) -> None:
    assert _abs_float(x_1) == 1
    assert _abs_float(y_1) == 1
    assert _abs_float(y_2) == 1
    assert float(x_1) == float(1)
    assert float(y_1) == float(1)
    assert float(y_2) == float(1)
    assert abs(-x_1) == x_1
    assert abs(-y_1) == y_1
    assert abs(-y_2) == y_2


@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__truediv__"])
def test_dual2_immutable(y_1, y_2, op) -> None:
    _ = getattr(y_1, op)(y_2)
    assert y_1 == Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=[])
    assert y_2 == Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=[1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__truediv__"])
def test_dual_immutable(x_1, op) -> None:
    _ = getattr(x_1, op)(Dual(2, vars=["new"], dual=np.array([4])))
    assert x_1 == Dual(1, vars=["v0", "v1"], dual=np.array([1, 2]))


def test_dual_raises(x_1) -> None:
    with pytest.raises(ValueError, match="`Dual` variable cannot possess `dual2`"):
        x_1.dual2


def test_dual_is_not_iterable(x_1, y_1):
    # do not want isinstance checks for Dual to identify them as a Sequence kind
    assert getattr(x_1, "__iter__", None) is None
    assert getattr(y_1, "__iter__", None) is None


def test_dual_has_no_len(x_1, y_1):
    # do not want isinstance checks for Dual to identify them as a Sequence kind
    assert getattr(x_1, "__len__", None) is None
    assert getattr(y_1, "__len__", None) is None


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("__add__", Dual(3, vars=["v0", "v1", "v2"], dual=np.array([1, 2, 3]))),
        ("__sub__", Dual(-1, vars=["v0", "v1", "v2"], dual=np.array([1, 2, -3]))),
        ("__mul__", Dual(2, vars=["v0", "v1", "v2"], dual=np.array([2, 4, 3]))),
        ("__truediv__", Dual(0.5, vars=["v0", "v1", "v2"], dual=np.array([0.5, 1, -0.75]))),
    ],
)
def test_ops(x_1, x_2, op, expected) -> None:
    result = getattr(x_1, op)(x_2)
    assert result == expected


def test_op_inversions(x_1, x_2) -> None:
    assert (x_1 + x_2) - (x_2 + x_1) == 0
    assert (x_1 / x_2) * (x_2 / x_1) == 1


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("__add__", Dual2(3, ["v0", "v1", "v2"], [1, 2, 3], [2, 1, 1, 1, 1, 0, 1, 0, 1])),
        ("__sub__", Dual2(-1, ["v0", "v1", "v2"], [1, 2, -3], [0, 1, -1, 1, 1, 0, -1, 0, -1])),
        ("__mul__", Dual2(2, ["v0", "v1", "v2"], [2, 4, 3], [3, 2, 2.5, 2, 2, 3, 2.5, 3, 1])),
        (
            "__truediv__",
            Dual2(
                0.5,
                ["v0", "v1", "v2"],
                [0.5, 1.0, -0.75],
                [0.25, 0.5, -0.625, 0.5, 0.5, -0.75, -0.625, -0.75, 0.875],
            ),
        ),
    ],
)
def test_ops2(y_2, y_3, op, expected) -> None:
    result = getattr(y_2, op)(y_3)
    assert result == expected


def test_op_inversions2(y_2, y_3) -> None:
    assert (y_2 + y_3) - (y_3 + y_2) == 0
    assert (y_2 / y_3) * (y_3 / y_2) == 1


def test_inverse(x_1, y_2) -> None:
    assert x_1 * x_1**-1 == 1
    assert y_2 * y_2**-1 == 1


def test_power_identity(x_1, y_2) -> None:
    result = x_1**1
    assert result == x_1

    result = y_2**1
    assert result == y_2


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("__add__", Dual(1 + 2.5, vars=["v0", "v1"], dual=np.array([1, 2]))),
        ("__sub__", Dual(1 - 2.5, vars=["v0", "v1"], dual=np.array([1, 2]))),
        ("__mul__", Dual(1 * 2.5, vars=["v0", "v1"], dual=np.array([1, 2]) * 2.5)),
        ("__truediv__", Dual(1 / 2.5, vars=["v0", "v1"], dual=np.array([1, 2]) / 2.5)),
    ],
)
def test_left_op_with_float(x_1, op, expected) -> None:
    result = getattr(x_1, op)(2.5)
    assert result == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("__add__", Dual2(1 + 2.5, ["v0", "v1"], [1, 2], [1.0, 1.0, 1.0, 1.0])),
        (
            "__sub__",
            Dual2(1 - 2.5, ["v0", "v1"], [1, 2], [1.0, 1.0, 1.0, 1.0]),
        ),
        ("__mul__", Dual2(1 * 2.5, ["v0", "v1"], [2.5, 5.0], [2.5, 2.5, 2.5, 2.5])),
        (
            "__truediv__",
            Dual2(1 / 2.5, ["v0", "v1"], [1 / 2.5, 2 / 2.5], [1 / 2.5, 1 / 2.5, 1 / 2.5, 1 / 2.5]),
        ),
    ],
)
def test_left_op_with_float2(y_2, op, expected) -> None:
    result = getattr(y_2, op)(2.5)
    assert result == expected


def test_right_op_with_float(x_1) -> None:
    assert 2.5 + x_1 == Dual(1 + 2.5, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert 2.5 - x_1 == Dual(2.5 - 1, vars=["v0", "v1"], dual=-np.array([1, 2]))
    assert 2.5 * x_1 == x_1 * 2.5
    assert 2.5 / x_1 == (x_1 / 2.5) ** -1


def test_right_op_with_float2(y_2) -> None:
    assert 2.5 + y_2 == Dual2(
        1 + 2.5,
        vars=["v0", "v1"],
        dual=[1.0, 2.0],
        dual2=[1.0, 1.0, 1.0, 1.0],
    )
    assert 2.5 - y_2 == Dual2(
        2.5 - 1,
        vars=["v0", "v1"],
        dual=[-1.0, -2.0],
        dual2=[-1.0, -1.0, -1.0, -1.0],
    )
    assert 2.5 * y_2 == y_2 * 2.5
    assert 2.5 / y_2 == (y_2 / 2.5) ** -1


def test_dual2_second_derivatives() -> None:
    "test power, multiplication, addition"

    def f(x, y, z):
        """
        f_x = 4x^3 y^2, f_y = 2y x^4 + z, f_z = 3z^2 +y
        f_xx = 12x^2 y^2, f_xy = 8 x^3 y, f_xz = 0,
        f_yx = 8x^3 y, f_yy = 2 x^4, f_yz = 1,
        f_zx = 0, f_zy = 1, f_zz = 6z
        """
        return x**4 * y**2 + z**3 + y * z

    x_, y_, z_ = 3, 2, 1

    x = Dual2(x_, vars=["x"], dual=[1], dual2=[])
    y = Dual2(y_, vars=["y"], dual=[1], dual2=[])
    z = Dual2(z_, vars=["z"], dual=[1], dual2=[])

    result = f(x, y, z)
    assert result.dual[0] == 4 * x_**3 * y_**2  # 432
    assert result.dual[1] == 2 * y_ * x_**4 + z_  # 325
    assert result.dual[2] == 3 * z_**2 + y_  # 5

    assert result.dual2[0, 0] * 2 == 12 * x_**2 * y_**2
    assert result.dual2[0, 1] * 2 == 8 * x_**3 * y_
    assert result.dual2[0, 2] * 2 == 0
    assert result.dual2[1, 0] * 2 == 8 * x_**3 * y_
    assert result.dual2[1, 1] * 2 == 2 * x_**4
    assert result.dual2[1, 2] * 2 == 1
    assert result.dual2[2, 0] * 2 == 0
    assert result.dual2[2, 1] * 2 == 1
    assert result.dual2[2, 2] * 2 == 6 * z_


def test_dual2_second_derivatives2() -> None:
    "test dual_exp, multiplication, division, dual_log"

    def f(x, y, z):
        return (x / z).__exp__() + (x * y).__log__()

    x_, y_, z_ = 3, 2, 1

    x = Dual2(x_, vars=["x"], dual=[1], dual2=[])
    y = Dual2(y_, vars=["y"], dual=[1], dual2=[])
    z = Dual2(z_, vars=["z"], dual=[1], dual2=[])

    result = f(x, y, z)
    xi = result.vars.index("x")
    yi = result.vars.index("y")
    zi = result.vars.index("z")
    assert result.dual[xi] == math.exp(x_ / z_) / z_ + 1 / x_
    assert result.dual[yi] == 1 / y_
    assert result.dual[zi] == -x_ * math.exp(x_ / z_) / z_**2

    assert result.dual2[xi, xi] * 2 == math.exp(x_ / z_) / z_**2 - 1 / x_**2
    assert result.dual2[xi, yi] * 2 == 0
    assert result.dual2[xi, zi] * 2 == math.exp(x_ / z_) * (-1 / z_**2 - x_ / z_**3)
    assert result.dual2[yi, xi] * 2 == 0
    assert result.dual2[yi, yi] * 2 == -1 / y_**2
    assert result.dual2[yi, zi] * 2 == 0
    assert result.dual2[zi, xi] * 2 == math.exp(x_ / z_) * (-1 / z_**2 - x_ / z_**3)
    assert result.dual2[zi, yi] * 2 == 0
    assert result.dual2[zi, zi] * 2 == math.exp(x_ / z_) * (x_**2 / z_**4 + 2 * x_ / z_**3)


def test_dual2_second_derivatives3() -> None:
    """
    h, f = dual_log(f), x^3y+y
    f_x = 1/f 3x^2y, f_y = 1/f (x^3+1),
    f_xx = -1/f^2 (3x^2y)^2 + 1/f 6xy, f_xy = -1/f^2 (3x^2y)(x^3+1),
    f_yy = -1/f^2 (x^3+1)^2 +1/f (0)
    """
    x_, y_ = 2, 1
    x = Dual2(x_, vars=["x"], dual=[1], dual2=[])
    y = Dual2(y_, vars=["y"], dual=[1], dual2=[])

    f = y * x**3 + y
    f_, fx_, fy_ = f.real, 3 * y_ * x_**2, x_**3 + 1
    fxx_, fxy_, fyy_ = 6 * x_ * y_, 3 * x_**2, 0

    xi = f.vars.index("x")
    yi = f.vars.index("y")

    assert f.dual[xi] == fx_
    assert f.dual[yi] == fy_
    assert f.dual2[xi, xi] * 2 == fxx_
    assert f.dual2[xi, yi] * 2 == fxy_
    assert f.dual2[yi, yi] * 2 == 0

    h = f.__log__()
    assert h.real == math.log(y_ * x_**3 + y_)
    assert h.dual[xi] == 1 / f_ * fx_
    assert h.dual[yi] == 1 / f_ * fy_
    assert h.dual2[xi, xi] * 2 == -1 / f_**2 * fx_**2 + 1 / f_ * fxx_
    assert h.dual2[xi, yi] * 2 == -1 / f_**2 * fx_ * fy_ + 1 / f_ * fxy_
    assert h.dual2[yi, xi] * 2 == -1 / f_**2 * fx_ * fy_ + 1 / f_ * fxy_
    assert h.dual2[yi, yi] * 2 == -1 / f_**2 * fy_**2 + 1 / f_ * fyy_


@pytest.mark.parametrize(
    ("power", "expected"),
    [
        (1, (2, 1, 0)),
        (2, (4, 4, 2)),
        (3, (8, 12, 12)),
        (4, (16, 32, 48)),
        (5, (32, 80, 160)),
        (6, (64, 192, 480)),
    ],
)
def test_dual_power_1d(power, expected) -> None:
    x = Dual(2, vars=["x"], dual=[1])
    y = Dual2(2, vars=["x"], dual=[1], dual2=[])
    f, g = x**power, y**power
    assert f.real == expected[0]
    assert f.dual[0] == expected[1]

    assert g.real == expected[0]
    assert g.dual[0] == expected[1]
    assert g.dual2[0, 0] * 2 == expected[2]


def test_dual2_power2_1d() -> None:
    x = Dual2(2, vars=["x"], dual=[1], dual2=[])
    assert (x**2) * (x ** (-2)) == 1
    assert (x**5) * (x ** (-5)) == 1
    z = (x**7.35) * (x ** (-7.35))
    assert abs(z - 1.0) < 1e-12


def test_dual2_power_2d() -> None:
    x = Dual2(2, vars=["x"], dual=[1], dual2=[])
    y = Dual2(3, vars=["y"], dual=[1], dual2=[])
    f = (x**4 * y**3) ** 2
    assert f.dual2[0, 1] * 2 == 1492992
    assert f.dual2[1, 0] * 2 == 1492992


def test_dual2_inv_specific() -> None:
    z = Dual2(2, vars=["x", "y"], dual=[2, 3], dual2=[])
    result = z**-1
    expected = Dual2(
        0.5,
        vars=["x", "y"],
        dual=[-0.5, -0.75],
        dual2=[0.5, 0.75, 0.75, 9 / 8],
    )
    assert result == expected


def test_dual_truediv(x_1) -> None:
    expected = Dual(1, [], [])
    result = x_1 / x_1
    assert result == expected


def test_dual2_exp_1d() -> None:
    x = Dual2(2, vars=["x"], dual=[1], dual2=[])
    f = x.__exp__()
    assert f.real == math.exp(2)
    assert f.dual[0] == math.exp(2)
    assert f.dual2[0, 0] * 2 == math.exp(2)


def test_dual2_log_1d() -> None:
    x = Dual2(2, vars=["x"], dual=[1], dual2=[])
    f = x.__log__()
    assert f.real == math.log(2)
    assert f.dual[0] == 0.5
    assert f.dual2[0] * 2 == -0.25


def test_dual2_log_exp() -> None:
    x = Dual2(2, vars=["x"], dual=[1], dual2=[])
    y = x.__log__()
    z = y.__exp__()
    assert x == z


def test_combined_vars_sorted(y_3) -> None:
    x = Dual2(2, vars=["a", "v0", "z"], dual=[1, 1, 1], dual2=[])
    result = x * y_3
    assert set(result.vars) == {"a", "v0", "v2", "z"}


@pytest.mark.parametrize(
    "x",
    [
        2,
        Dual(2, [], []),
        Dual2(2, [], [], []),
    ],
)
def test_log(x) -> None:
    result = dual_log(x)
    expected = math.log(2)
    assert result == expected


def test_dual_log_base() -> None:
    result = dual_log(16, 2)
    assert result == 4

    result = dual_log(Dual(16, [], []), 2)
    assert result == Dual(4, [], [])


@pytest.mark.parametrize(
    "x",
    [
        2,
        Dual(2, [], []),
        Dual2(2, [], [], []),
    ],
)
def test_exp(x) -> None:
    result = dual_exp(x)
    expected = math.exp(2)
    assert result == expected


@pytest.mark.parametrize(
    "x",
    [
        Dual(1.25, ["x"], []),
        Dual2(1.25, ["x"], [], []),
    ],
)
def test_norm_cdf(x) -> None:
    result = dual_norm_cdf(x)
    expected = NormalDist().cdf(1.250)
    assert abs(result - expected) < 1e-10

    approx_grad = (NormalDist().cdf(1.25001) - NormalDist().cdf(1.25)) * 100000
    assert abs(gradient(result, ["x"])[0] - approx_grad) < 1e-5

    if isinstance(x, Dual2):
        approx_grad2 = (NormalDist().cdf(1.25) - NormalDist().cdf(1.24999)) * 100000
        approx_grad2 = (approx_grad - approx_grad2) * 100000
        assert abs(gradient(result, ["x"], order=2)[0] - approx_grad2) < 1e-5


@pytest.mark.parametrize(
    "x",
    [
        Dual(0.75, ["x"], []),
        Dual2(0.75, ["x"], [], []),
    ],
)
def test_inv_norm_cdf(x) -> None:
    result = dual_inv_norm_cdf(x)
    expected = NormalDist().inv_cdf(0.75)
    assert abs(result - expected) < 1e-10

    approx_grad = (NormalDist().inv_cdf(0.75001) - NormalDist().inv_cdf(0.75)) * 100000
    assert abs(gradient(result, ["x"])[0] - approx_grad) < 1e-4

    if isinstance(x, Dual2):
        approx_grad2 = (NormalDist().inv_cdf(0.75) - NormalDist().inv_cdf(0.74999)) * 100000
        approx_grad2 = (approx_grad - approx_grad2) * 100000
        assert abs(gradient(result, ["x"], order=2)[0] - approx_grad2) < 1e-4


def test_norm_cdf_value() -> None:
    result = dual_norm_cdf(1.0)
    expected = 0.8413
    assert abs(result - expected) < 1e-4


def test_inv_norm_cdf_value() -> None:
    result = dual_inv_norm_cdf(0.50)
    expected = 0.0
    assert abs(result - expected) < 1e-4


@pytest.mark.skip(reason="downcast vars is not used within the library, kept only for compat.")
def test_downcast_vars() -> None:
    w = Dual(2, ["x", "y", "z"], [0, 1, 1])
    assert w.__downcast_vars__().vars == ("y", "z")

    x = Dual2(2, ["x", "y", "z"], [0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert x.__downcast_vars__().vars == ("y", "z")

    y = Dual2(2, ["x", "y", "z"], [0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert y.__downcast_vars__().vars == ("z",)

    z = Dual2(2, ["x", "y", "z"], [0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1, 1])
    assert z.__downcast_vars__().vars == ("y", "z")


def test_gradient_of_non_present_vars(x_1) -> None:
    result = gradient(x_1)
    assert np.all(np.isclose(result, np.array([1, 2])))


@pytest.mark.parametrize(("base", "exponent"), [(0, 1), (1, 0)])
def test_powers_bad_type(base, exponent, x_1, y_1) -> None:
    base = x_1 if base else y_1
    exponent = x_1 if exponent else y_1
    with pytest.raises(TypeError):
        base**exponent


def test_keep_manifold_gradient() -> None:
    du2 = Dual2(
        10,
        ["x", "y", "z"],
        dual=[1, 2, 3],
        dual2=[2, 3, 4, 3, 4, 5, 4, 5, 6],
    )
    result = gradient(du2, ["x", "z"], 1, keep_manifold=True)
    expected = np.array([Dual2(1, ["x", "z"], [4, 8], []), Dual2(3, ["x", "z"], [8, 12], [])])
    assertions = result == expected
    assert all(assertions)


def test_dual_set_order(x_1, y_1) -> None:
    assert set_order(x_1, 1) == x_1
    assert set_order(y_1, 2) == y_1
    assert set_order(1.0, 2) == 1.0
    assert set_order(x_1, 2) == y_1
    assert set_order(y_1, 1) == x_1
    assert set_order(x_1, 0) == 1.0


def test_variable_set_order() -> None:
    x = Variable(2.0, ["x"])
    x_dual = set_order(x, order=1)
    assert isinstance(x_dual, Dual)
    x_dual2 = set_order(x, order=2)
    assert isinstance(x_dual2, Dual2)


def test_perturbation_confusion() -> None:
    # https://www.bcl.hamilton.ie/~barak/papers/ifl2005.pdf

    # Utilised tagged variables
    x = Dual(1.0, ["x"], [])
    y = Dual(1.0, ["y"], [])
    z = gradient(x + y, ["y"])[0]
    result = gradient(x * z, ["x"])
    assert result == 1.0

    # Replicates untagged variables
    x = Dual(1.0, ["x"], [])
    y = Dual(1.0, ["x"], [])
    z = gradient(x + y, ["x"])[0]
    result = gradient(x * z, ["x"])
    assert result == 2.0


# Linalg dual_solve tests


def test_solve(A, b) -> None:
    x = dual_solve(A, b)
    x_np = np.linalg.solve(A, b)
    diff = x - x_np
    assertions = [abs(diff[i, 0]) < 1e-10 for i in range(A.shape[0])]
    assert all(assertions)


def test_solve_lsqrs() -> None:
    A = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    b = np.array([[-1, 0.2, 0.9, 2.1]]).T
    result = dual_solve(A, b, allow_lsq=True, types=(float, float))
    assert abs(result[0, 0] - 1.0) < 1e-9
    assert abs(result[1, 0] + 0.95) < 1e-9


def test_solve_dual() -> None:
    A = np.array([[1, 0], [0, 1]], dtype="object")
    b = np.array([Dual(2, ["x"], np.array([1])), Dual(5, ["x", "y"], np.array([1, 1]))])[
        :,
        np.newaxis,
    ]
    x = dual_solve(A, b, types=(float, Dual))
    assertions = abs(b - x) < 1e-10
    assert all(assertions)


def test_solve_dual2() -> None:
    A = np.array(
        [
            [Dual2(1, [], [], []), Dual2(0, [], [], [])],
            [Dual2(0, [], [], []), Dual2(1, [], [], [])],
        ],
        dtype="object",
    )
    b = np.array([Dual2(2, ["x"], [1], []), Dual2(5, ["x", "y"], [1, 1], [])])[:, np.newaxis]
    x = dual_solve(A, b, types=(Dual2, Dual2))
    assertions = abs(b - x) < 1e-10
    assert all(assertions)


def test_sparse_solve(A_sparse) -> None:
    b = np.array(
        [0, 0.90929743, 0.14112001, -0.7568025, -0.95892427, -0.2794155, 0.6569866, 0.98935825, 0],
    )
    b = b[:, np.newaxis]
    x = dual_solve(A_sparse, b)
    x_np = np.linalg.solve(A_sparse, b)
    diff = x - x_np
    assertions = [abs(diff[i, 0]) < 1e-10 for i in range(A_sparse.shape[0])]
    assert all(assertions)


@pytest.mark.skipif(not DUAL_CORE_PY, reason="Rust Dual has not implemented Multi-Dim Solve")
def test_multi_dim_solve() -> None:
    A = np.array([[Dual(0.5, [], []), Dual(2, ["y"], [])], [Dual(2.5, ["y"], []), Dual(4, [], [])]])
    b = np.array(
        [[Dual(6.5, [], []), Dual(9, ["z"], [])], [Dual(14.5, ["y"], []), Dual(21, ["z"], [])]],
    )

    x = dual_solve(A, b)
    result = np.matmul(A, x).flatten()
    expected = b.flatten()
    for i in range(4):
        assert abs(result[i] - expected[i]) < 1e-13
        assert all(np.isclose(gradient(result[i], ["y", "z"]), gradient(expected[i], ["y", "z"])))


# Test numpy compat


def test_numpy_isclose(y_2) -> None:
    # np.isclose not supported for non-numeric dtypes
    a = np.array([y_2, y_2])
    b = np.array([y_2, y_2])
    with pytest.raises(TypeError):
        assert np.isclose(a, b)


def test_numpy_equality(y_2) -> None:
    # instead of isclose use == (which uses math.isclose elementwise) and then np.all
    a = np.array([y_2, y_2])
    b = np.array([y_2, y_2])
    result = a == b
    assert np.all(result)


@pytest.mark.parametrize(
    "z",
    [
        Dual(2.0, ["y"], []),
        Dual2(3.0, ["x"], [1], [2]),
    ],
)
@pytest.mark.parametrize(
    "arg",
    [
        2.2,
        Dual(3, ["x"], []),
        Dual2(3, ["x"], [2], [3]),
    ],
)
@pytest.mark.parametrize(
    "op_str",
    [
        "add",
        "sub",
        "mul",
        "truediv",
    ],
)
def test_numpy_broadcast_ops_types(z, arg, op_str) -> None:
    op = "__" + op_str + "__"
    if type(z) in [Dual, Dual2] and type(arg) in [Dual, Dual2] and type(arg) is not type(z):
        pytest.skip("Cannot operate Dual and Dual2 together.")
    result = getattr(np.array([z, z]), op)(arg)
    expected = np.array([getattr(z, op)(arg), getattr(z, op)(arg)])
    assert np.all(result == expected)

    result = getattr(arg, op)(np.array([z, z]))
    if result is NotImplemented:
        opr = "__r" + op_str + "__"
        result = getattr(np.array([z, z]), opr)(arg)
        expected = np.array([getattr(z, opr)(arg), getattr(z, opr)(arg)])
    else:
        expected = np.array([getattr(arg, op)(z), getattr(arg, op)(z)])
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "z",
    [
        Dual(2.0, ["y"], []),
        Dual2(3.0, ["x"], [1], [2]),
    ],
)
def test_numpy_broadcast_pow_types(z) -> None:
    result = np.array([z, z]) ** 3
    expected = np.array([z**3, z**3])
    assert np.all(result == expected)

    result = z ** np.array([3, 4])
    expected = np.array([z**3, z**4])
    assert np.all(result == expected)


def test_numpy_matmul(y_2, y_1) -> None:
    a = np.array([y_2, y_1])
    result = np.matmul(a[:, np.newaxis], a[np.newaxis, :])
    expected = np.array([[y_2 * y_2, y_2 * y_1], [y_2 * y_1, y_1 * y_1]])
    assert np.all(result == expected)


@pytest.mark.skipif(
    version.parse(np.__version__) >= version.parse("1.25.0"),
    reason="Object dtypes accepted by NumPy in 1.25.0+",
)
def test_numpy_einsum(y_2, y_1) -> None:
    # einsum does not work with object dtypes
    a = np.array([y_2, y_1])
    with pytest.raises(TypeError):
        _ = np.einsum("i,j", a, a, optimize=True)


@pytest.mark.skipif(
    version.parse(np.__version__) < version.parse("1.25.0"),
    reason="Object dtypes not accepted by NumPy in <1.25.0",
)
def test_numpy_einsum_works(y_2, y_1) -> None:
    a = np.array([y_2, y_1])
    result = np.einsum("i,j", a, a, optimize=True)
    expected = np.array([[y_2 * y_2, y_2 * y_1], [y_2 * y_1, y_1 * y_1]])
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "z",
    [
        Dual(2.0, ["y"], []),
        Dual2(3.0, ["x"], [1], [2]),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    ],
)
def test_numpy_dtypes(z, dtype) -> None:
    np.array([1, 2], dtype=dtype) + z
    z + np.array([1, 2], dtype=dtype)

    z + dtype(2)
    dtype(2) + z


class TestVariable:
    @pytest.mark.parametrize(
        ("op", "exp"),
        [
            ("__add__", Variable(4.0, ["x"])),
            ("__radd__", Variable(4.0, ["x"])),
            ("__sub__", Variable(1.0, ["x"])),
            ("__rsub__", -Variable(1.0, ["x"])),
            ("__mul__", Variable(3.75, ["x"], [1.5])),
            ("__rmul__", Variable(3.75, ["x"], [1.5])),
            ("__truediv__", Variable(2.5 / 1.5, ["x"], [1.0 / 1.5])),
            ("__rtruediv__", Dual(1.5, [], []) / Dual(2.5, ["x"], [])),
        ],
    )
    def test_variable_f64(self, op, exp):
        with default_context("_global_ad_order", 1):
            f = 1.5
            v = Variable(2.5, ("x",))
            result = getattr(v, op)(f)
            assert result == exp

    def test_variable_f64_reverse(self):
        v = Variable(2.5, ("x",))
        assert (1.5 + v) == Variable(4.0, ["x"], [])
        assert (1.5 - v) == Variable(-1.0, ["x"], [-1.0])
        assert (1.5 * v) == Variable(1.5 * 2.5, ["x"], [1.5])
        assert (1.5 / v) == Dual(1.5, [], []) / Dual(2.5, ["x"], [])

    def test_rtruediv_global_ad(self):
        exp = Dual2(1.5, [], [], []) / Dual2(2.5, ["x"], [], [])
        with default_context("_global_ad_order", 2):
            f = 1.5
            v = Variable(2.5, ("x",))
            result = f / v
            assert result == exp

    @pytest.mark.parametrize(
        ("op", "exp"),
        [
            ("__add__", Dual(4.0, ["x"], [2])),
            ("__radd__", Dual(4.0, ["x"], [2])),
            ("__sub__", Dual(1.0, ["x"], [0])),
            ("__rsub__", Dual(-1.0, ["x"], [0])),
            ("__mul__", Dual(3.75, ["x"], [4.0])),
            ("__rmul__", Dual(3.75, ["x"], [4.0])),
            ("__truediv__", Dual(2.5, ["x"], []) / Dual(1.5, ["x"], [])),
            ("__rtruediv__", Dual(1.5, ["x"], []) / Dual(2.5, ["x"], [])),
        ],
    )
    def test_variable_dual(self, op, exp):
        f = Dual(1.5, ["x"], [])
        v = Variable(2.5, ("x",))
        result = getattr(v, op)(f)
        assert result == exp

    def test_variable_dual_reverse(self):
        f = Dual(1.5, ["x"], [])
        v = Variable(2.5, ("x",))
        assert f + v == Dual(4.0, ["x"], [2.0])
        assert f - v == Dual(-1.0, ["x"], [0.0])
        assert f * v == Dual(1.5 * 2.5, ["x"], [4.0])
        assert f / v == Dual(1.5, ["x"], [1.0]) / Dual(2.5, ["x"], [1.0])

    @pytest.mark.parametrize(
        ("op", "exp"),
        [
            ("__add__", Dual2(4.0, ["x"], [2], [])),
            ("__radd__", Dual2(4.0, ["x"], [2], [])),
            ("__sub__", Dual2(1.0, ["x"], [0], [])),
            ("__rsub__", Dual2(-1.0, ["x"], [0], [])),
            ("__mul__", Dual2(1.5, ["x"], [1.0], []) * Dual2(2.5, ["x"], [1.0], [])),
            ("__rmul__", Dual2(1.5, ["x"], [1.0], []) * Dual2(2.5, ["x"], [1.0], [])),
            ("__truediv__", Dual2(2.5, ["x"], [], []) / Dual2(1.5, ["x"], [], [])),
            ("__rtruediv__", Dual2(1.5, ["x"], [], []) / Dual2(2.5, ["x"], [], [])),
        ],
    )
    def test_variable_dual2(self, op, exp):
        f = Dual2(1.5, ["x"], [], [])
        v = Variable(2.5, ("x",))
        result = getattr(v, op)(f)
        assert result == exp

    def test_variable_dual2_reverse(self):
        f = Dual2(1.5, ["x"], [], [])
        v = Variable(2.5, ("x",))
        assert f + v == Dual2(4.0, ["x"], [2.0], [])
        assert f - v == Dual2(-1.0, ["x"], [0.0], [])
        assert f * v == Dual2(1.5, ["x"], [], []) * Dual2(2.5, ["x"], [], [])
        assert f / v == Dual2(1.5, ["x"], [], []) / Dual2(2.5, ["x"], [], [])

    @pytest.mark.parametrize(
        ("op", "exp"),
        [
            ("__add__", Dual(4.0, ["x"], [2])),
            ("__radd__", Dual(4.0, ["x"], [2])),
            ("__sub__", Dual(1.0, ["x"], [0])),
            ("__rsub__", Dual(-1.0, ["x"], [0])),
            ("__mul__", Dual(1.5, ["x"], [1.0]) * Dual(2.5, ["x"], [1.0])),
            ("__rmul__", Dual(1.5, ["x"], [1.0]) * Dual(2.5, ["x"], [1.0])),
            ("__truediv__", Dual(2.5, ["x"], []) / Dual(1.5, ["x"], [])),
        ],
    )
    def test_variable_variable_ad1(self, op, exp):
        f = Variable(1.5, ("x",))
        v = Variable(2.5, ("x",))
        with default_context("_global_ad_order", 1):
            result = getattr(v, op)(f)
            assert result == exp

    @pytest.mark.parametrize(
        ("op", "exp"),
        [
            ("__add__", Dual2(4.0, ["x"], [2], [])),
            ("__radd__", Dual2(4.0, ["x"], [2], [])),
            ("__sub__", Dual2(1.0, ["x"], [0], [])),
            ("__rsub__", Dual2(-1.0, ["x"], [0], [])),
            ("__mul__", Dual2(1.5, ["x"], [1.0], []) * Dual2(2.5, ["x"], [1.0], [])),
            ("__rmul__", Dual2(1.5, ["x"], [1.0], []) * Dual2(2.5, ["x"], [1.0], [])),
            ("__truediv__", Dual2(2.5, ["x"], [], []) / Dual2(1.5, ["x"], [], [])),
        ],
    )
    def test_variable_variable_ad2(self, op, exp):
        f = Variable(1.5, ("x",))
        v = Variable(2.5, ("x",))
        with default_context("_global_ad_order", 2):
            result = getattr(v, op)(f)
            assert result == exp

    @pytest.mark.parametrize(
        ("op", "ad", "exp"),
        [
            ("__exp__", 1, Dual(0.5, ["x"], []).__exp__()),
            ("__exp__", 2, Dual2(0.5, ["x"], [], []).__exp__()),
            ("__log__", 1, Dual(0.5, ["x"], []).__log__()),
            ("__log__", 2, Dual2(0.5, ["x"], [], []).__log__()),
            ("__norm_cdf__", 1, Dual(0.5, ["x"], []).__norm_cdf__()),
            ("__norm_cdf__", 2, Dual2(0.5, ["x"], [], []).__norm_cdf__()),
            ("__norm_inv_cdf__", 1, Dual(0.5, ["x"], []).__norm_inv_cdf__()),
            ("__norm_inv_cdf__", 2, Dual2(0.5, ["x"], [], []).__norm_inv_cdf__()),
        ],
    )
    def test_variable_funcs(self, op, ad, exp):
        with default_context("_global_ad_order", ad):
            var = Variable(0.5, ["x"])
            result = getattr(var, op)()
            assert result == exp

    @pytest.mark.parametrize(
        ("op", "ad", "exp"),
        [
            ("__pow__", 1, Dual(2.5, ["x"], []).__pow__(2)),
            ("__pow__", 2, Dual2(2.5, ["x"], [], []).__pow__(2)),
        ],
    )
    def test_variable_pow(self, op, ad, exp):
        with default_context("_global_ad_order", ad):
            var = Variable(2.5, ["x"])
            result = getattr(var, op)(2)
            assert result == exp

    @pytest.mark.parametrize(("order", "exp"), [(1, 2.0), (2, 0.0)])
    def test_gradient(self, order, exp):
        var = Variable(2.0, ["x"], [2.0])
        result = gradient(var, ["x"], order=order)[0]
        assert result == exp

    def test_eq(self):
        v1 = Variable(1.0, ["x", "y"])
        v2 = Variable(1.0, ["x", "y"])
        assert v1 == v2

    @pytest.mark.parametrize(
        ("func", "exp"),
        [
            (dual_exp, Dual(0.5, ["x"], []).__exp__()),
            (dual_log, Dual(0.5, ["x"], []).__log__()),
            (dual_norm_cdf, Dual(0.5, ["x"], []).__norm_cdf__()),
            (dual_inv_norm_cdf, Dual(0.5, ["x"], []).__norm_inv_cdf__()),
            (dual_norm_pdf, dual_norm_pdf(Dual(0.5, ["x"], []))),
        ],
    )
    def test_standalone_funcs(self, func, exp):
        var = Variable(0.5, ["x"])
        result = func(var)
        assert result == exp

    def test_z_exogenous_example(self):
        curve = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}, id="curve")
        solver = Solver(
            curves=[curve], instruments=[IRS(dt(2000, 1, 1), "6m", "S", curves=curve)], s=[2.50]
        )
        irs = IRS(
            effective=dt(2000, 1, 1),
            termination="6m",
            frequency="S",
            leg2_frequency="M",
            fixed_rate=Variable(3.0, ["R"]),
            notional=Variable(5e6, ["N"]),
            leg2_float_spread=Variable(0.0, ["z"]),
            curves="curve",
        )
        result = irs.exo_delta(vars=["N", "R", "z"], vars_scalar=[1.0, 0.01, 1.0], solver=solver)

        exp0 = irs.npv(solver=solver) / 5e6
        exp1 = irs.analytic_delta(curves=curve)
        exp2 = irs.analytic_delta(curves=curve, leg=2)

        assert abs(exp0 - result.iloc[0, 0]) < 1e-8
        assert abs(exp1 + result.iloc[1, 0]) < 1e-8
        assert abs(exp2 + result.iloc[2, 0]) < 1e-8


def test_set_multiple_objects_order():
    a = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}, id="a")
    b = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}, id="b")
    c = a

    result = _set_ad_order_objects([2, 2, 0], [a, b, c])
    assert a._ad == 2
    assert b._ad == 2
    assert c._ad == 2  # c is a!
    expected = {
        id(a): 0,
        id(b): 0,
    }
    assert result == expected

    _set_ad_order_objects(result, [a, b, c])
    assert a._ad == 0
    assert b._ad == 0
    assert c._ad == 0  # c is a!


def test_set_multiple_objects_order_raises():
    a = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}, id="a")
    with pytest.raises(ValueError):
        _set_ad_order_objects([0], [a, a])
