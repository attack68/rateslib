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
    assert result == "<Dual: 1.000000, ('v0', 'v1'), [1 2]>"


def test_dual_str(x_1):
    result = x_1.__str__()
    assert result == " val = 1.00000000\n  dv0 = 1.000000\n  dv1 = 2.000000\n"


@pytest.mark.parametrize(
    "vars, expected",
    [
        ("v0", 1.00),
        (["v1", "v0"], np.array([2.0, 1.0])),
    ],
)
def test_gradient_method(vars, expected, x_1):
    result = x_1.gradient(vars)
    assert np.all(result == expected)
