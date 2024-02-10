import pytest
import numpy as np

import context

from rateslibrs import Dual


@pytest.fixture()
def x_1():
    return Dual(1, vars=["v0", "v1"], dual=[1, 2])


@pytest.fixture()
def x_2():
    return Dual(2, vars=["v0", "v2"], dual=[0, 3])


def test_zero_init():
    x = Dual(1, vars=["x"], dual=[])
    assert np.all(x.dual == np.ones(1))


def test_dual_repr(x_1):
    result = x_1.__repr__()
    assert result == "<Dual: 1.000000, (v0, v1), [1, 2]>"


def test_dual_repr_4vars():
    x = Dual(1.23456789, ["a", "b", "c", "d"], [1.01, 2, 3.50001, 4])
    result = x.__repr__()
    assert result == "<Dual: 1.234568, (a, b, c, ...), [1.01, 2, 3.50001, ...]>"


def test_dual_str(x_1):
    result = x_1.__str__()
    assert result == " val = 1.00000000\n  dv0 = 1.000000\n  dv1 = 2.000000\n"


@pytest.mark.parametrize(
    "vars, expected",
    [
        (["v1"], 2.00),
        (["v1", "v0"], np.array([2.0, 1.0])),
    ],
)
def test_gradient_method(vars, expected, x_1):
    result = x_1.gradient(vars)
    assert np.all(result == expected)


def test_neg(x_1):
    result = -x_1
    expected = Dual(-1, vars=["v0", "v1"], dual=[-1, -2])
    assert result == expected


def test_eq_ne(x_1):
    # non-matching types
    assert 0 != Dual(0, ["single_var"], [])
    # floats
    assert 2.0 == Dual(2, [], [])
    assert Dual(2, [], []) == 2.0
    # equality
    assert x_1 == Dual(1, vars=["v0", "v1"], dual=np.array([1, 2]))
    # non-matching elements
    assert x_1 != Dual(2, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert x_1 != Dual(1, vars=["v0", "v1"], dual=np.array([2, 2]))
    assert x_1 != Dual(1, vars=["v2", "v1"], dual=np.array([1, 2]))


def test_lt():
    assert Dual(1, ["x"], []) < Dual(2, ["y"], [])
    assert Dual(1, ["x"], []) < 10
    assert 0.5 < Dual(1, ["x"], [])


def test_le():
    assert Dual(1.0, ["x"], []) <= Dual(1.0, ["y"], [])
    assert Dual(1, ["x"], []) <= 1.0
    assert 1.0 <= Dual(1.0, ["x"], [])


def test_gt():
    assert Dual(3, ["x"], []) > Dual(2, ["y"], [])
    assert Dual(1, ["x"], []) > 0.5
    assert 0.5 > Dual(0.3, ["x"], [])


def test_ge():
    assert Dual(1.0, ["x"], []) >= Dual(1.0, ["y"], [])
    assert Dual(1, ["x"], []) >= 1.0
    assert 1.0 >= Dual(1.0, ["x"], [])


@pytest.mark.parametrize(
    "op, expected",
    [
        ("__add__", Dual(3, vars=["v0", "v1", "v2"], dual=[1, 2, 3])),
        ("__sub__", Dual(-1, vars=["v0", "v1", "v2"], dual=[1, 2, -3])),
        ("__mul__", Dual(2, vars=["v0", "v1", "v2"], dual=[2, 4, 3])),
        ("__truediv__", Dual(0.5, vars=["v0", "v1", "v2"], dual=[0.5, 1, -0.75])),
    ],
)
def test_ops(x_1, x_2, op, expected):
    result = getattr(x_1, op)(x_2)
    assert result == expected


@pytest.mark.parametrize(
    "op, expected",
    [
        ("__add__", Dual(1 + 2.5, vars=["v0", "v1"], dual=[1, 2])),
        ("__sub__", Dual(1 - 2.5, vars=["v0", "v1"], dual=[1, 2])),
        ("__mul__", Dual(1 * 2.5, vars=["v0", "v1"], dual=[2.5, 5.0])),
        ("__truediv__", Dual(1 / 2.5, vars=["v0", "v1"], dual=[1/2.5, 2/2.5])),
    ],
)
def test_left_op_with_float(x_1, op, expected):
    result = getattr(x_1, op)(2.5)
    assert result == expected


def test_right_op_with_float(x_1):
    assert 2.5 + x_1 == Dual(1 + 2.5, vars=["v0", "v1"], dual=[1, 2])
    assert 2.5 - x_1 == Dual(2.5 - 1, vars=["v0", "v1"], dual=[-1, -2])
    assert 2.5 * x_1 == x_1 * 2.5
    assert 2.5 / x_1 == (x_1 / 2.5) ** -1.0


def test_op_inversions(x_1, x_2):
    assert (x_1 + x_2) - (x_2 + x_1) == 0
    assert (x_1 / x_2) * (x_2 / x_1) == 1


def test_inverse(x_1):
    assert x_1 * x_1**-1 == 1


def test_power_identity(x_1):
    result = x_1**1
    assert result == x_1


@pytest.mark.parametrize(
    "power, expected",
    [
        (1, (2, 1)),
        (2, (4, 4)),
        (3, (8, 12)),
        (4, (16, 32)),
        (5, (32, 80)),
        (6, (64, 192)),
    ],
)
def test_dual_power_1d(power, expected):
    x = Dual(2, vars=["x"], dual=[1])
    f = x**power
    assert f.real == expected[0]
    assert f.dual[0] == expected[1]


def test_dual_truediv(x_1):
    expected = Dual(1, [], [])
    result = x_1 / x_1
    assert result == expected


def test_combined_vars_sorted(x_1):
    x = Dual(2, vars=["a", "v0", "z"], dual=[])
    result = x_1 * x
    expected = ["v0", "v1", "a", "z"]
    assert result.vars == expected
    # x vars are stored first
    result = x * x_1
    expected = ["a", "v0", "z", "v1"]
    assert result.vars == expected

