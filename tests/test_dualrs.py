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