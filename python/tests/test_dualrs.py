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

import numpy as np
import pytest
from packaging import version
from rateslib.dual import Dual, Dual2, dual_exp, dual_log, dual_solve, gradient
from rateslib.rs import ADOrder

DUAL_CORE_PY = False


@pytest.fixture
def x_1():
    return Dual(1, vars=["v0", "v1"], dual=[1, 2])


@pytest.fixture
def x_2():
    return Dual(2, vars=["v0", "v2"], dual=[0, 3])


def test_zero_init() -> None:
    x = Dual(1, vars=["x"], dual=[])
    assert np.all(x.dual == np.ones(1))


def test_dual_repr(x_1) -> None:
    result = x_1.__repr__()
    assert result == "<Dual: 1.000000, (v0, v1), [1.0, 2.0]>"


def test_dual_repr_4vars() -> None:
    x = Dual(1.23456789, ["a", "b", "c", "d"], [1.01, 2, 3.50001, 4])
    result = x.__repr__()
    assert result == "<Dual: 1.234568, (a, b, c, ...), [1.0, 2.0, 3.5, ...]>"


def test_dual_str(x_1) -> None:
    result = x_1.__str__()
    assert result == "<Dual: 1.000000, (v0, v1), [1.0, 2.0]>"


@pytest.mark.skipif(DUAL_CORE_PY, reason="Gradient comparison cannot compare Py and Rs Duals.")
@pytest.mark.parametrize(
    ("vars_", "expected"),
    [
        (["v1"], 2.00),
        (["v1", "v0"], np.array([2.0, 1.0])),
    ],
)
def test_gradient_method(vars_, expected, x_1) -> None:
    result = gradient(x_1, vars_)
    assert np.all(result == expected)


def test_neg(x_1) -> None:
    result = -x_1
    expected = Dual(-1, vars=["v0", "v1"], dual=[-1, -2])
    assert result == expected


def test_eq_ne(x_1) -> None:
    # non-matching types
    assert Dual(0, ["single_var"], []) != 0
    # floats
    assert Dual(2, [], []) == 2.0
    assert Dual(2, [], []) == 2.0
    # equality
    assert x_1 == Dual(1, vars=["v0", "v1"], dual=np.array([1, 2]))
    # non-matching elements
    assert x_1 != Dual(2, vars=["v0", "v1"], dual=np.array([1, 2]))
    assert x_1 != Dual(1, vars=["v0", "v1"], dual=np.array([2, 2]))
    assert x_1 != Dual(1, vars=["v2", "v1"], dual=np.array([1, 2]))


def test_lt() -> None:
    assert Dual(1, ["x"], []) < Dual(2, ["y"], [])
    assert Dual(1, ["x"], []) < 10
    assert Dual(1, ["x"], []) > 0.5


def test_le() -> None:
    assert Dual(1.0, ["x"], []) <= Dual(1.0, ["y"], [])
    assert Dual(1, ["x"], []) <= 1.0
    assert Dual(1.0, ["x"], []) >= 1.0


def test_gt() -> None:
    assert Dual(3, ["x"], []) > Dual(2, ["y"], [])
    assert Dual(1, ["x"], []) > 0.5
    assert Dual(0.3, ["x"], []) < 0.5


def test_ge() -> None:
    assert Dual(1.0, ["x"], []) >= Dual(1.0, ["y"], [])
    assert Dual(1, ["x"], []) >= 1.0
    assert Dual(1.0, ["x"], []) <= 1.0


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("__add__", Dual(3, vars=["v0", "v1", "v2"], dual=[1, 2, 3])),
        ("__sub__", Dual(-1, vars=["v0", "v1", "v2"], dual=[1, 2, -3])),
        ("__mul__", Dual(2, vars=["v0", "v1", "v2"], dual=[2, 4, 3])),
        ("__truediv__", Dual(0.5, vars=["v0", "v1", "v2"], dual=[0.5, 1, -0.75])),
    ],
)
def test_ops(x_1, x_2, op, expected) -> None:
    result = getattr(x_1, op)(x_2)
    assert result == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("__add__", Dual(1 + 2.5, vars=["v0", "v1"], dual=[1, 2])),
        ("__sub__", Dual(1 - 2.5, vars=["v0", "v1"], dual=[1, 2])),
        ("__mul__", Dual(1 * 2.5, vars=["v0", "v1"], dual=[2.5, 5.0])),
        ("__truediv__", Dual(1 / 2.5, vars=["v0", "v1"], dual=[1 / 2.5, 2 / 2.5])),
    ],
)
def test_left_op_with_float(x_1, op, expected) -> None:
    result = getattr(x_1, op)(2.5)
    assert result == expected


def test_right_op_with_float(x_1) -> None:
    assert 2.5 + x_1 == Dual(1 + 2.5, vars=["v0", "v1"], dual=[1, 2])
    assert 2.5 - x_1 == Dual(2.5 - 1, vars=["v0", "v1"], dual=[-1, -2])
    assert 2.5 * x_1 == x_1 * 2.5
    assert 2.5 / x_1 == (x_1 / 2.5) ** -1.0


def test_op_inversions(x_1, x_2) -> None:
    assert (x_1 + x_2) - (x_2 + x_1) == 0
    assert (x_1 / x_2) * (x_2 / x_1) == 1


def test_inverse(x_1) -> None:
    assert x_1 * x_1**-1 == 1


def test_power_identity(x_1) -> None:
    result = x_1**1
    assert result == x_1


@pytest.mark.parametrize(
    ("power", "expected"),
    [
        (1, (2, 1)),
        (2, (4, 4)),
        (3, (8, 12)),
        (4, (16, 32)),
        (5, (32, 80)),
        (6, (64, 192)),
    ],
)
def test_dual_power_1d(power, expected) -> None:
    x = Dual(2, vars=["x"], dual=[1])
    f = x**power
    assert f.real == expected[0]
    assert f.dual[0] == expected[1]


def test_dual_truediv(x_1) -> None:
    expected = Dual(1, [], [])
    result = x_1 / x_1
    assert result == expected


def test_combined_vars_sorted(x_1) -> None:
    x = Dual(2, vars=["a", "v0", "z"], dual=[])
    result = x_1 * x
    expected = ["v0", "v1", "a", "z"]
    assert result.vars == expected
    # x vars are stored first
    result = x * x_1
    expected = ["a", "v0", "z", "v1"]
    assert result.vars == expected


def test_exp(x_1) -> None:
    result = x_1.__exp__()
    expected = Dual(math.e, ["v0", "v1"], [math.e, 2 * math.e])
    assert result == expected


def test_log(x_1) -> None:
    result = x_1.__log__()
    expected = Dual(0.0, ["v0", "v1"], [1.0, 2.0])
    assert result == expected


# Test NumPy compat


def test_numpy_isclose(x_1) -> None:
    # np.isclose not supported for non-numeric dtypes
    a = np.array([x_1, x_1])
    b = np.array([x_1, x_1])
    with pytest.raises(TypeError):
        assert np.isclose(a, b)


def test_numpy_equality(x_1) -> None:
    # instead of isclose use == (which uses math.isclose elementwise) and then np.all
    a = np.array([x_1, x_1])
    b = np.array([x_1, x_1])
    result = a == b
    assert np.all(result)


@pytest.mark.parametrize(
    "z",
    [
        Dual(2.0, ["y"], []),
        # Dual2(3.0, "x", np.array([1]), np.array([[2]])),
    ],
)
@pytest.mark.parametrize(
    "arg",
    [
        2.2,
        Dual(3, ["x"], []),
        # Dual2(3, "x", np.array([2]), np.array([[3]])),
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
    types = [Dual]  # ,Dual2]
    if type(z) in types and type(arg) in types and type(arg) is not type(z):
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
        # Dual2(3.0, "x", np.array([1]), np.array([[2]])),
    ],
)
def test_numpy_broadcast_pow_types(z) -> None:
    result = np.array([z, z]) ** 3
    expected = np.array([z**3, z**3])
    assert np.all(result == expected)

    result = z ** np.array([3, 4])
    expected = np.array([z**3, z**4])
    assert np.all(result == expected)


def test_numpy_matmul(x_1) -> None:
    x_2 = Dual(2.5, ["x", "y"], [3.0, -2.0])
    a = np.array([x_1, x_2])
    result = np.matmul(a[:, np.newaxis], a[np.newaxis, :])
    expected = np.array([[x_1 * x_1, x_1 * x_2], [x_2 * x_1, x_2 * x_2]])
    assert np.all(result == expected)


@pytest.mark.skipif(
    version.parse(np.__version__) < version.parse("1.25.0"),
    reason="Object dtypes not accepted by NumPy in <1.25.0",
)
def test_numpy_einsum_works(x_1) -> None:
    x_2 = Dual(2.5, ["x", "y"], [3.0, -2.0])
    a = np.array([x_1, x_2])
    result = np.einsum("i,j", a, a, optimize=True)
    expected = np.array([[x_1 * x_1, x_1 * x_2], [x_2 * x_1, x_2 * x_2]])
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "z",
    [
        Dual(2.0, ["y"], []),
        # Dual2(3.0, "x", np.array([1]), np.array([[2]])),
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


def test_dual_solve() -> None:
    a = np.array([[Dual(1.0, [], []), Dual(0.0, [], [])], [Dual(0.0, [], []), Dual(1.0, [], [])]])
    b = np.array([Dual(2.0, ["x"], [1.0]), Dual(5.0, ["x", "y"], [1.0, 1.0])])
    result = dual_solve(a, b[:, None], types=(Dual, Dual))[:, 0]
    expected = np.array([Dual(2.0, ["x", "y"], [1.0, 0.0]), Dual(5.0, ["x", "y"], [1.0, 1.0])])
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "obj",
    [
        Dual(1.0, ["x", "y"], [1.0, 2.0]),
        Dual2(2.0, ["x", "y"], [1.0, 2.0], [1.0, 2.0, 2.0, 3.0]),
    ],
)
def test_pickle(obj) -> None:
    import pickle

    pickled_obj = pickle.dumps(obj)
    reloaded = pickle.loads(pickled_obj)
    assert obj == reloaded


@pytest.mark.parametrize("z", [2.0, Dual(2.0, ["z"], [])])
@pytest.mark.parametrize("p", [2.0, Dual(2.0, ["p"], [])])
def test_dual_powers_finite_diff(z, p):
    if isinstance(z, float) and isinstance(p, float):
        return None  # float power not in scope

    result = z**p

    if isinstance(z, Dual):
        # Finite diff test
        z_diff = ((z + 0.00001) ** p - result) / 0.00001
        assert abs(gradient(result, ["z"])[0] - z_diff) < 1e-4

    if isinstance(p, Dual):
        # Finite diff test
        p_diff = (z ** (p + 0.00001) - result) / 0.00001
        assert abs(gradient(result, ["p"])[0] - p_diff) < 1e-4


def test_dual_powers_operators() -> None:
    z = Dual(2.3, ["x", "y", "z"], [1.0, 2.0, 3.0])
    p = Dual(4.4, ["x", "y", "p"], [2.0, 3.0, 4.0])
    result = z**p
    expected = dual_exp(p * dual_log(z))
    assert abs(result - expected) < 1e-12
    assert np.all(
        np.isclose(gradient(result, ["x", "y", "z", "p"]), gradient(expected, ["x", "y", "z", "p"]))
    )


@pytest.mark.parametrize("z", [2.0, Dual2(2.0, ["z"], [], [])])
@pytest.mark.parametrize("p", [2.0, Dual2(2.0, ["p"], [], [])])
def test_dual2_powers_finite_diff_first_order(z, p):
    if isinstance(z, float) and isinstance(p, float):
        return None  # float power not in scope

    result = z**p

    if isinstance(z, Dual2):
        # Finite diff test
        z_diff = ((z + 0.00001) ** p - result) / 0.00001
        assert abs(gradient(result, ["z"])[0] - z_diff) < 1e-4

    if isinstance(p, Dual2):
        # Finite diff test
        p_diff = (z ** (p + 0.00001) - result) / 0.00001
        assert abs(gradient(result, ["p"])[0] - p_diff) < 1e-4


@pytest.mark.parametrize("z", [2.0, Dual2(2.0, ["z"], [], [])])
@pytest.mark.parametrize("p", [2.0, Dual2(2.0, ["p"], [], [])])
def test_dual2_powers_finite_diff_second_order(z, p):
    if isinstance(z, float) and isinstance(p, float):
        return None  # float power not in scope

    result = z**p

    vars_ = (isinstance(z, Dual2), isinstance(p, Dual2))
    if vars_[0]:
        z_up = (z + 0.00001) ** p
        z_dw = (z - 0.00001) ** p
        diff = (z_up + z_dw - 2 * result) / 1e-10
        assert abs(gradient(result, ["z"], order=2)[0][0] - diff) < 1e-4

    if vars_[1]:
        p_up = z ** (p + 0.00001)
        p_dw = z ** (p - 0.00001)
        diff = (p_up + p_dw - 2 * result) / 1e-10
        assert abs(gradient(result, ["p"], order=2)[0][0] - diff) < 1e-4

    if vars_[1] and vars_[0]:
        upup = (z + 0.00001) ** (p + 0.00001)
        dwdw = (z - 0.00001) ** (p - 0.00001)
        updw = (z + 0.00001) ** (p - 0.00001)
        dwup = (z - 0.00001) ** (p + 0.00001)
        diff = (upup + dwdw - updw - dwup) / 4e-10
        assert abs(gradient(result, ["z", "p"], order=2)[0, 1] - diff) < 1e-4


def test_dual2_powers_operators() -> None:
    z = Dual2(2.3, ["x", "y", "z"], [1.0, 2.0, 3.0], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    p = Dual2(4.4, ["x", "y", "p"], [2.0, 3.0, 4.0], [2, 3, 4, 5, 2, 3, 4, 3, 4])
    result = z**p
    expected = dual_exp(p * dual_log(z))
    assert abs(result - expected) < 1e-12
    assert np.all(
        np.isclose(gradient(result, ["x", "y", "z", "p"]), gradient(expected, ["x", "y", "z", "p"]))
    )
    assert np.all(
        np.isclose(
            gradient(result, ["x", "y", "z", "p"], order=2),
            gradient(expected, ["x", "y", "z", "p"], order=2),
        )
    )
