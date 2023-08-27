import pytest
import math
import numpy as np
from packaging import version

import context
from rateslib.dual import (
    Dual,
    Dual2,
    dual_exp,
    dual_log,
    dual_solve,
    set_order,
    _plu_decomp,
    _pivot_matrix,
)


@pytest.fixture()
def x_1():
    return Dual(1, vars=["v0", "v1"], dual=np.array([1, 2]))


@pytest.fixture()
def x_2():
    return Dual(2, vars=["v0", "v2"], dual=np.array([0, 3]))


@pytest.fixture()
def y_1():
    return Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]))


@pytest.fixture()
def y_2():
    return Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=np.ones((2, 2)))


@pytest.fixture()
def y_3():
    return Dual2(2, vars=["v0", "v2"], dual=np.array([0, 3]), dual2=np.ones((2, 2)))


@pytest.fixture()
def A():
    return np.random.randn(25).reshape(5, 5)


@pytest.fixture()
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
        ]
    )


@pytest.fixture()
def b():
    return np.random.randn(5).reshape(5, 1)


def test_zero_init():
    x = Dual(1, vars=["x"])
    assert np.all(x.dual == np.ones(1))

    y = Dual2(1, vars=["x"])
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
def test_no_type_crossing_on_ops(x_1, y_1, op):
    with pytest.raises(TypeError):
        getattr(x_1, op)(y_1)

    with pytest.raises(TypeError):
        getattr(y_1, op)(x_1)


def test_dual_repr(x_1, y_2):
    result = x_1.__repr__()
    assert result == "<Dual: 1.000000, ('v0', 'v1'), [1 2]>"

    result = y_2.__repr__()
    assert result == "<Dual2: 1.000000, ('v0', 'v1'), [1 2], [[...]]>"


def test_dual_str(x_1, y_2):
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
    "vars, expected",
    [
        ("v0", 1.00),
        (["v1", "v0"], np.array([2.0, 1.0])),
    ],
)
def test_gradient_method(vars, expected, x_1, y_2):
    result = x_1.gradient(vars)
    assert np.all(result == expected)

    result = y_2.gradient(vars)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "vars, expected",
    [
        ("v0", 2.00),
        (["v1", "v0"], np.array([[2.0, 2.0], [2.0, 2.0]])),
    ],
)
def test_gradient_method2(vars, expected, y_2):
    result = y_2.gradient(vars, 2)
    assert np.all(result == expected)


def test_rdiv_raises(x_1, y_1):
    with pytest.raises(TypeError):
        _ = "string" / x_1

    with pytest.raises(TypeError):
        _ = "string" / y_1


def test_neg(x_1, y_2):
    assert -x_1 == Dual(-1, vars=["v0", "v1"], dual=-np.array([1, 2]))
    assert -y_2 == Dual2(-1, vars=["v0", "v1"], dual=-np.array([1, 2]), dual2=-np.ones((2, 2)))


def test_eq_ne(x_1, y_1, y_2):
    # non-matching types
    assert 0 != Dual(0, "single_var")
    assert 0 != Dual2(0, "single_var")
    # ints
    assert 2 == Dual(2)
    assert 2 == Dual2(2)
    # floats
    assert 3.3 == Dual(3.3)
    assert 3.3 == Dual2(3.3)
    # no type crossing
    with pytest.raises(TypeError):
        assert x_1 != y_1
    # equality
    assert x_1 == Dual(1, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert y_1 == Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert y_2 == Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=np.ones((2, 2)))
    # non-matching elements
    assert x_1 != Dual(2, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert x_1 != Dual(1, vars=["v0", "v1"], dual=np.array([2, 2]))
    assert x_1 != Dual(1, vars=["v2", "v1"], dual=np.array([1, 2]))
    # non-matching elements
    assert y_1 != Dual2(2, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert y_1 != Dual2(1, vars=["v0", "v1"], dual=np.array([2, 2]))
    assert y_1 != Dual2(1, vars=["v2", "v1"], dual=np.array([1, 2]))
    # non-matching dual2
    assert y_2 != Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=np.ones((2, 2)) * 2)


def test_lt():
    assert Dual(1, "x") < Dual(2, "y")
    assert Dual2(1, "z") < Dual2(2, "x")
    assert Dual(1, "x") < 10
    assert not Dual(1, "x") < 0


def test_lt_raises():
    with pytest.raises(TypeError, match="Cannot compare"):
        assert Dual(1, "x") < Dual2(2, "y")


def test_gt():
    assert Dual(2, "x") > Dual(1, "y")
    assert Dual2(2, "z") > Dual2(1, "x")
    assert Dual(1, "x") > 0
    assert not Dual(1, "x") > 10


def test_gt_raises():
    with pytest.raises(TypeError, match="Cannot compare"):
        assert Dual(2, "x") > Dual2(1, "y")


def test_dual2_abs_float(x_1, y_1, y_2):
    assert abs(x_1) == 1
    assert abs(y_1) == 1
    assert abs(y_2) == 1
    assert float(x_1) == float(1)
    assert float(y_1) == float(1)
    assert float(y_2) == float(1)


@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__truediv__"])
def test_dual2_immutable(y_1, y_2, op):
    _ = getattr(y_1, op)(y_2)
    assert y_1 == Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert y_2 == Dual2(1, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=np.ones((2, 2)))


@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__", "__truediv__"])
def test_dual_immutable(x_1, op):
    _ = getattr(x_1, op)(Dual(2, vars=["new"], dual=np.array([4])))
    assert x_1 == Dual(1, vars=["v0", "v1"], dual=np.array([1, 2]))


def test_dual_raises(x_1):
    with pytest.raises(ValueError, match="`Dual` variable cannot possess `dual2`"):
        x_1.dual2


@pytest.mark.parametrize(
    "op, expected",
    [
        ("__add__", Dual(3, vars=["v0", "v1", "v2"], dual=np.array([1, 2, 3]))),
        ("__sub__", Dual(-1, vars=["v0", "v1", "v2"], dual=np.array([1, 2, -3]))),
        ("__mul__", Dual(2, vars=["v0", "v1", "v2"], dual=np.array([2, 4, 3]))),
        ("__truediv__", Dual(0.5, vars=["v0", "v1", "v2"], dual=np.array([0.5, 1, -0.75]))),
    ],
)
def test_ops(x_1, x_2, op, expected):
    result = getattr(x_1, op)(x_2)
    assert result == expected


def test_op_inversions(x_1, x_2):
    assert (x_1 + x_2) - (x_2 + x_1) == 0
    assert (x_1 / x_2) * (x_2 / x_1) == 1


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "__add__",
            Dual2(
                3,
                vars=["v0", "v1", "v2"],
                dual=np.array([1, 2, 3]),
                dual2=np.array([[2, 1, 1], [1, 1, 0], [1, 0, 1]]),
            ),
        ),
        (
            "__sub__",
            Dual2(
                -1,
                vars=["v0", "v1", "v2"],
                dual=np.array([1, 2, -3]),
                dual2=np.array([[0, 1, -1], [1, 1, 0], [-1, 0, -1]]),
            ),
        ),
        (
            "__mul__",
            Dual2(
                2,
                vars=["v0", "v1", "v2"],
                dual=np.array([2, 4, 3]),
                dual2=np.array([[3, 2, 2.5], [2, 2, 3], [2.5, 3, 1]]),
            ),
        ),
        (
            "__truediv__",
            Dual2(
                0.5,
                vars=["v0", "v1", "v2"],
                dual=np.array([0.5, 1.0, -0.75]),
                dual2=np.array([[0.25, 0.5, -0.625], [0.5, 0.5, -0.75], [-0.625, -0.75, 0.875]]),
            ),
        ),
    ],
)
def test_ops2(y_2, y_3, op, expected):
    result = getattr(y_2, op)(y_3)
    assert result == expected


def test_op_inversions2(y_2, y_3):
    assert (y_2 + y_3) - (y_3 + y_2) == 0
    assert (y_2 / y_3) * (y_3 / y_2) == 1


def test_inverse(x_1, y_2):
    assert x_1 * x_1**-1 == 1
    assert y_2 * y_2**-1 == 1


def test_power_identity(x_1, y_2):
    result = x_1**1
    assert result == x_1

    result = y_2**1
    assert result == y_2


@pytest.mark.parametrize(
    "op, expected",
    [
        ("__add__", Dual(1 + 2.5, vars=["v0", "v1"], dual=np.array([1, 2]))),
        ("__sub__", Dual(1 - 2.5, vars=["v0", "v1"], dual=np.array([1, 2]))),
        ("__mul__", Dual(1 * 2.5, vars=["v0", "v1"], dual=np.array([1, 2]) * 2.5)),
        ("__truediv__", Dual(1 / 2.5, vars=["v0", "v1"], dual=np.array([1, 2]) / 2.5)),
    ],
)
def test_left_op_with_float(x_1, op, expected):
    result = getattr(x_1, op)(2.5)
    assert result == expected


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "__add__",
            Dual2(1 + 2.5, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=np.ones((2, 2))),
        ),
        (
            "__sub__",
            Dual2(1 - 2.5, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=np.ones((2, 2))),
        ),
        (
            "__mul__",
            Dual2(
                1 * 2.5, vars=["v0", "v1"], dual=np.array([1, 2]) * 2.5, dual2=np.ones((2, 2)) * 2.5
            ),
        ),
        (
            "__truediv__",
            Dual2(
                1 / 2.5, vars=["v0", "v1"], dual=np.array([1, 2]) / 2.5, dual2=np.ones((2, 2)) / 2.5
            ),
        ),
    ],
)
def test_left_op_with_float2(y_2, op, expected):
    result = getattr(y_2, op)(2.5)
    assert result == expected


def test_right_op_with_float(x_1):
    assert 2.5 + x_1 == Dual(1 + 2.5, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert 2.5 - x_1 == Dual(2.5 - 1, vars=["v0", "v1"], dual=-np.array([1, 2]))
    assert 2.5 * x_1 == x_1 * 2.5
    assert 2.5 / x_1 == (x_1 / 2.5) ** -1


def test_right_op_with_float2(y_2):
    assert 2.5 + y_2 == Dual2(
        1 + 2.5, vars=["v0", "v1"], dual=np.array([1, 2]), dual2=np.ones((2, 2))
    )
    assert 2.5 - y_2 == Dual2(
        2.5 - 1, vars=["v0", "v1"], dual=-np.array([1, 2]), dual2=-np.ones((2, 2))
    )
    assert 2.5 * y_2 == y_2 * 2.5
    assert 2.5 / y_2 == (y_2 / 2.5) ** -1


def test_dual2_second_derivatives():
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

    x = Dual2(x_, vars=["x"], dual=np.array([1]))
    y = Dual2(y_, vars=["y"], dual=np.array([1]))
    z = Dual2(z_, vars=["z"], dual=np.array([1]))

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


def test_dual2_second_derivatives2():
    "test dual_exp, multiplication, division, dual_log"

    def f(x, y, z):
        return (x / z).__exp__() + (x * y).__log__()

    x_, y_, z_ = 3, 2, 1

    x = Dual2(x_, vars=["x"], dual=np.array([1]))
    y = Dual2(y_, vars=["y"], dual=np.array([1]))
    z = Dual2(z_, vars=["z"], dual=np.array([1]))

    result = f(x, y, z)
    assert result.dual[0] == math.exp(x_ / z_) / z_ + 1 / x_
    assert result.dual[1] == 1 / y_
    assert result.dual[2] == -x_ * math.exp(x_ / z_) / z_**2

    assert result.dual2[0, 0] * 2 == math.exp(x_ / z_) / z_**2 - 1 / x_**2
    assert result.dual2[0, 1] * 2 == 0
    assert result.dual2[0, 2] * 2 == math.exp(x_ / z_) * (-1 / z_**2 - x_ / z_**3)
    assert result.dual2[1, 0] * 2 == 0
    assert result.dual2[1, 1] * 2 == -1 / y_**2
    assert result.dual2[1, 2] * 2 == 0
    assert result.dual2[2, 0] * 2 == math.exp(x_ / z_) * (-1 / z_**2 - x_ / z_**3)
    assert result.dual2[2, 1] * 2 == 0
    assert result.dual2[2, 2] * 2 == math.exp(x_ / z_) * (x_**2 / z_**4 + 2 * x_ / z_**3)


def test_dual2_second_derivatives3():
    """
    h, f = dual_log(f), x^3y+y
    f_x = 1/f 3x^2y, f_y = 1/f (x^3+1),
    f_xx = -1/f^2 (3x^2y)^2 + 1/f 6xy, f_xy = -1/f^2 (3x^2y)(x^3+1),
    f_yy = -1/f^2 (x^3+1)^2 +1/f (0)
    """
    x_, y_ = 2, 1
    x = Dual2(x_, vars=["x"], dual=np.array([1]))
    y = Dual2(y_, vars=["y"], dual=np.array([1]))

    f = y * x**3 + y
    f_, fx_, fy_ = f.real, 3 * y_ * x_**2, x_**3 + 1
    fxx_, fxy_, fyy_ = 6 * x_ * y_, 3 * x_**2, 0
    assert f.dual[0] == fx_
    assert f.dual[1] == fy_
    assert f.dual2[0, 0] * 2 == fxx_
    assert f.dual2[0, 1] * 2 == fxy_
    assert f.dual2[1, 1] * 2 == 0

    h = f.__log__()
    assert h.real == math.log(y_ * x_**3 + y_)
    assert h.dual[0] == 1 / f_ * fx_
    assert h.dual[1] == 1 / f_ * fy_
    assert h.dual2[0, 0] * 2 == -1 / f_**2 * fx_**2 + 1 / f_ * fxx_
    assert h.dual2[0, 1] * 2 == -1 / f_**2 * fx_ * fy_ + 1 / f_ * fxy_
    assert h.dual2[1, 0] * 2 == -1 / f_**2 * fx_ * fy_ + 1 / f_ * fxy_
    assert h.dual2[1, 1] * 2 == -1 / f_**2 * fy_**2 + 1 / f_ * fyy_


@pytest.mark.parametrize(
    "power, expected",
    [
        (1, (2, 1, 0)),
        (2, (4, 4, 2)),
        (3, (8, 12, 12)),
        (4, (16, 32, 48)),
        (5, (32, 80, 160)),
        (6, (64, 192, 480)),
    ],
)
def test_dual_power_1d(power, expected):
    x = Dual(2, vars=["x"], dual=np.array([1]))
    y = Dual2(2, vars=["x"], dual=np.array([1]))
    f, g = x**power, y**power
    assert f.real == expected[0]
    assert f.dual[0] == expected[1]

    assert g.real == expected[0]
    assert g.dual[0] == expected[1]
    assert g.dual2[0, 0] * 2 == expected[2]


def test_dual2_power2_1d():
    x = Dual2(2, vars=["x"], dual=np.array([1]))
    assert (x**2) * (x ** (-2)) == 1
    assert (x**5) * (x ** (-5)) == 1
    assert (x**7.35) * (x ** (-7.35)) == 1


def test_dual2_power_2d():
    x = Dual2(2, vars=["x"], dual=np.array([1]))
    y = Dual2(3, vars=["y"], dual=np.array([1]))
    f = (x**4 * y**3) ** 2
    assert f.dual2[0, 1] * 2 == 1492992
    assert f.dual2[1, 0] * 2 == 1492992


def test_dual2_inv_specific():
    z = Dual2(2, vars=["x", "y"], dual=np.array([2, 3]))
    result = z**-1
    expected = Dual2(
        0.5,
        vars=["x", "y"],
        dual=np.array([-0.5, -0.75]),
        dual2=np.array([[0.5, 0.75], [0.75, 9 / 8]]),
    )
    assert result == expected


def test_dual_truediv(x_1):
    expected = Dual(1)
    result = x_1 / x_1
    assert result == expected


def test_dual2_exp_1d():
    x = Dual2(2, vars=["x"], dual=np.array([1]))
    f = x.__exp__()
    assert f.real == math.exp(2)
    assert f.dual[0] == math.exp(2)
    assert f.dual2[0, 0] * 2 == math.exp(2)


def test_dual2_log_1d():
    x = Dual2(2, vars=["x"], dual=np.array([1]))
    f = x.__log__()
    assert f.real == math.log(2)
    assert f.dual[0] == 0.5
    assert f.dual2[0] * 2 == -0.25


def test_dual2_log_exp():
    x = Dual2(2, vars=["x"], dual=np.array([1]))
    y = x.__log__()
    z = y.__exp__()
    assert x == z


def test_combined_vars_sorted(y_3):
    x = Dual2(2, vars=["a", "v0", "z"], dual=np.array([1, 1, 1]))
    result = x * y_3
    assert result.vars == ("a", "v0", "v2", "z")


@pytest.mark.parametrize(
    "x",
    [
        2,
        Dual(2),
        Dual2(2),
    ],
)
def test_log(x):
    result = dual_log(x)
    expected = math.log(2)
    assert result == expected


def test_dual_log_base():
    result = dual_log(16, 2)
    assert result == 4

    result = dual_log(Dual(16), 2)
    assert result == Dual(4)


@pytest.mark.parametrize(
    "x",
    [
        2,
        Dual(2),
        Dual2(2),
    ],
)
def test_exp(x):
    result = dual_exp(x)
    expected = math.exp(2)
    assert result == expected


def test_downcast_vars():
    w = Dual(
        2,
        vars=["x", "y", "z"],
        dual=np.array([0, 1, 1]),
    )
    assert w.__downcast_vars__().vars == ("y", "z")

    x = Dual2(
        2,
        vars=["x", "y", "z"],
        dual=np.array([0, 1, 1]),
        dual2=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    )
    assert x.__downcast_vars__().vars == ("y", "z")

    y = Dual2(
        2,
        vars=["x", "y", "z"],
        dual=np.array([0, 0, 1]),
        dual2=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    )
    assert y.__downcast_vars__().vars == ("z",)

    z = Dual2(
        2,
        vars=["x", "y", "z"],
        dual=np.array([0, 0, 1]),
        dual2=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]]),
    )
    assert z.__downcast_vars__().vars == ("y", "z")


def test_gradient_of_non_present_vars(x_1):
    result = x_1.gradient()
    assert np.all(np.isclose(result, np.array([1, 2])))


@pytest.mark.parametrize("base, exponent", [(1, 1), (1, 0), (0, 1), (0, 0)])
def test_powers_bad_type(base, exponent, x_1, y_1):
    base = x_1 if base else y_1
    exponent = x_1 if exponent else y_1
    with pytest.raises(TypeError):
        base**exponent


def test_keep_manifold_gradient():
    du2 = Dual2(
        10,
        ["x", "y", "z"],
        dual=np.array([1, 2, 3]),
        dual2=np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6]]),
    )
    result = du2.gradient(["x", "z"], 1, keep_manifold=True)
    expected = np.array(
        [Dual2(1, ["x", "z"], np.array([4, 8])), Dual2(3, ["x", "z"], np.array([8, 12]))]
    )
    assertions = result == expected
    assert all(assertions)


def test_dual_set_order(x_1, y_1):
    assert x_1._set_order(2) == y_1
    assert y_1._set_order(1) == x_1
    assert x_1._set_order(1) == x_1
    assert y_1._set_order(2) == y_1

    assert set_order(1.0, 2) == 1.0
    assert set_order(x_1, 2) == y_1
    assert set_order(y_1, 1) == x_1

    assert set_order(x_1, 0) == 1.0


# Linalg dual_solve tests


def test_pivoting():
    A = np.array(
        [[1, 0, 0, 0, 0], [-1, 1, 0, 0, 0], [0, -1, 0, 1, 0], [-1, 0, 0.5, 0, 0], [0, 0, -1, 0, 1]]
    )
    result, _ = _pivot_matrix(A, method=1)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    assert np.all(result == expected)
    result, _ = _pivot_matrix(A, method=2)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
    assert np.all(result == expected)


def test_sparse_plu():
    A = np.array(
        [
            [0, 1, 0],
            [0, 1, 1],
            [2, 3, 0],
        ]
    )
    P, L, U = _plu_decomp(A)

    diff = np.matmul(P, A) - np.matmul(L, U)
    assertions = [abs(diff[i, j]) < 1e-10 for i in range(3) for j in range(3)]
    assert all(assertions)


@pytest.mark.parametrize("sparse", [False, True])
def test_plu(A, A_sparse, sparse):
    if sparse:
        A = A_sparse
    n = A.shape[0]

    P, L, U = _plu_decomp(A)

    assertions = [abs(L[i, j]) < 1e-10 for i in range(n) for j in range(i + 1, n)]
    assert all(assertions)

    assertions = [abs(U[i, j]) < 1e-10 for j in range(n) for i in range(j + 1, n)]
    assert all(assertions)

    diff = np.matmul(P, A) - np.matmul(L, U)
    assertions = [abs(diff[i, j]) < 1e-10 for i in range(n) for j in range(n)]
    assert all(assertions)


def test_solve(A, b):
    x = dual_solve(A, b)
    x_np = np.linalg.solve(A, b)
    diff = x - x_np
    assertions = [abs(diff[i, 0]) < 1e-10 for i in range(A.shape[0])]
    assert all(assertions)


def test_solve_lsqrs():
    A = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    b = np.array([[-1, 0.2, 0.9, 2.1]]).T
    result = dual_solve(A, b, allow_lsq=True)
    assert abs(result[0, 0] - 1.0) < 1e-9
    assert abs(result[1, 0] + 0.95) < 1e-9


def test_solve_dual():
    A = np.array([[1, 0], [0, 1]], dtype="object")
    b = np.array([Dual(2, ["x"], np.array([1])), Dual(5, ["x", "y"], np.array([1, 1]))])[
        :, np.newaxis
    ]
    x = dual_solve(A, b)
    assertions = abs(b - x) < 1e-10
    assert all(assertions)


def test_solve_dual2():
    A = np.array([[Dual2(1), Dual2(0)], [Dual2(0), Dual2(1)]], dtype="object")
    b = np.array([Dual2(2, ["x"], np.array([1])), Dual2(5, ["x", "y"], np.array([1, 1]))])[
        :, np.newaxis
    ]
    x = dual_solve(A, b)
    assertions = abs(b - x) < 1e-10
    assert all(assertions)


def test_sparse_solve(A_sparse):
    b = np.array(
        [0, 0.90929743, 0.14112001, -0.7568025, -0.95892427, -0.2794155, 0.6569866, 0.98935825, 0]
    )
    b = b[:, np.newaxis]
    x = dual_solve(A_sparse, b)
    x_np = np.linalg.solve(A_sparse, b)
    diff = x - x_np
    assertions = [abs(diff[i, 0]) < 1e-10 for i in range(A_sparse.shape[0])]
    assert all(assertions)


def test_multi_dim_solve():
    A = np.array([[Dual(0.5), Dual(2, "y")], [Dual(2.5, "y"), Dual(4)]])
    b = np.array([[Dual(6.5), Dual(9, "z")], [Dual(14.5, "y"), Dual(21, "z")]])

    result = dual_solve(A, b)
    assert np.all(b == np.matmul(A, result))


# Test numpy compat


def test_numpy_isclose(y_2):
    # np.isclose not supported for non-numeric dtypes
    a = np.array([y_2, y_2])
    b = np.array([y_2, y_2])
    with pytest.raises(TypeError):
        assert np.isclose(a, b)


def test_numpy_equality(y_2):
    # instead of isclose use == (which uses math.isclose elementwise) and then np.all
    a = np.array([y_2, y_2])
    b = np.array([y_2, y_2])
    result = a == b
    assert np.all(result)


@pytest.mark.parametrize(
    "z",
    [
        Dual(2.0, "y"),
        Dual2(3.0, "x", np.array([1]), np.array([[2]])),
    ],
)
@pytest.mark.parametrize(
    "arg",
    [
        2.2,
        Dual(3, "x"),
        Dual2(3, "x", np.array([2]), np.array([[3]])),
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
def test_numpy_broadcast_ops_types(z, arg, op_str):
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
        Dual(2.0, "y"),
        Dual2(3.0, "x", np.array([1]), np.array([[2]])),
    ],
)
def test_numpy_broadcast_pow_types(z):
    result = np.array([z, z]) ** 3
    expected = np.array([z**3, z**3])
    assert np.all(result == expected)

    result = z ** np.array([3, 4])
    expected = np.array([z**3, z**4])
    assert np.all(result == expected)


def test_numpy_matmul(y_2, y_1):
    a = np.array([y_2, y_1])
    result = np.matmul(a[:, np.newaxis], a[np.newaxis, :])
    expected = np.array([[y_2 * y_2, y_2 * y_1], [y_2 * y_1, y_1 * y_1]])
    assert np.all(result == expected)


@pytest.mark.skipif(
    version.parse(np.__version__) >= version.parse("1.25.0"),
    reason="Object dtypes accepted by NumPy in 1.25.0+",
)
def test_numpy_einsum(y_2, y_1):
    # einsum does not work with object dtypes
    a = np.array([y_2, y_1])
    with pytest.raises(TypeError):
        _ = np.einsum("i,j", a, a, optimize=True)


@pytest.mark.skipif(
    version.parse(np.__version__) < version.parse("1.25.0"),
    reason="Object dtypes not accepted by NumPy in <1.25.0",
)
def test_numpy_einsum_works(y_2, y_1):
    a = np.array([y_2, y_1])
    result = np.einsum("i,j", a, a, optimize=True)
    expected = np.array([[y_2 * y_2, y_2 * y_1], [y_2 * y_1, y_1 * y_1]])
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "z",
    [
        Dual(2.0, "y"),
        Dual2(3.0, "x", np.array([1]), np.array([[2]])),
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
def test_numpy_dtypes(z, dtype):
    np.array([1], dtype=dtype) + z
    z + np.array([1], dtype=dtype)

    z + dtype(2)
    dtype(2) + z
