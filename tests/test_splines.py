import pytest
import numpy as np
import copy

import context
from rateslib.splines import PPSplineF64


@pytest.fixture()
def t():
    return np.array([1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4])


@pytest.fixture()
def x():
    return np.linspace(1, 4, 7)


@pytest.mark.parametrize(
    "i, expected",
    [
        (0, np.array([1.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (1, np.array([0.0, 0.375, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (2, np.array([0.0, 0.375, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (3, np.array([0.0, 0.125, 1.0, 0.125, 0.0, 0.0, 0.0])),
        (4, np.array([0.0, 0.0, 0.0, 0.59375, 0.25, 0.03125, 0.0])),
        (5, np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.25, 0.0])),
        (6, np.array([0.0, 0.0, 0.0, 0.03125, 0.25, 0.59375, 0.0])),
        (7, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 1.0])),
    ],
)
def test_individual_bsplines(t, x, i, expected):
    bs = PPSplineF64(k=4, t=t)
    result = bs.bsplev(x, i=i)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "i, expected",
    [
        (0, np.array([-3.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (1, np.array([3.0, -0.75, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (2, np.array([0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (3, np.array([0.0, 0.75, -3.0, -0.75, 0.0, 0.0, 0.0])),
        (4, np.array([0.0, 0.0, 3.0, -0.1875, -0.75, -0.1875, 0.0])),
        (5, np.array([0.0, 0.0, 0.0, 0.75, 0.0, -0.75, 0.0])),
        (6, np.array([0.0, 0.0, 0.0, 0.1875, 0.75, 0.1875, -3.0])),
        (7, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 3.0])),
    ],
)
def test_first_derivative_endpoint_support(t, x, i, expected):
    bs = PPSplineF64(k=4, t=t)
    result = bs.bspldnev(x, i=i, m=1)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "i, expected",
    [
        (0, np.array([6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (1, np.array([-12.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (2, np.array([6.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (3, np.array([0.0, 3.0, 6.0, 3.0, 0.0, 0.0, 0.0])),
        (4, np.array([0.0, 0.0, -9.0, -3.75, 1.5, 0.75, 0.0])),
        (5, np.array([0.0, 0.0, 3.0, 0.0, -3.0, 0.0, 3.0])),
        (6, np.array([0.0, 0.0, 0.0, 0.75, 1.5, -3.75, -9.0])),
        (7, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0])),
    ],
)
def test_second_derivative_endpoint_support(t, x, i, expected):
    bs = PPSplineF64(k=4, t=t)
    result = bs.bspldnev(x, i=i, m=2)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "i, expected",
    [
        (0, np.array([-6.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (1, np.array([18.0, 18.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (2, np.array([-18.0, -18.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        (3, np.array([6.0, 6.0, -6.0, -6.0, 0.0, 0.0, 0.0])),
        (4, np.array([0.0, 0.0, 10.5, 10.5, -1.5, -1.5, -1.5])),
        (5, np.array([0.0, 0.0, -6.0, -6.0, 6.0, 6.0, 6.0])),
        (6, np.array([0.0, 0.0, 1.5, 1.5, -10.5, -10.5, -10.5])),
        (7, np.array([0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0])),
    ],
)
def test_third_derivative_endpoint_support(t, x, i, expected):
    bs = PPSplineF64(k=4, t=t)
    result = bs.bspldnev(x, i=i, m=3)
    assert (result == expected).all()


def test_fourth_derivative_endpoint_support(t, x):
    bs = PPSplineF64(k=4, t=t)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(8):
        test = bs.bspldnev(x, i=i, m=4) == expected
        assert test.all()


def test_ppdnev(t):
    bs = PPSplineF64(k=4, t=t, c=[1, 2, -1, 2, 1, 1, 2, 2.0])
    r1 = bs.ppdnev_single(1.1, 2)
    r2 = bs.ppdnev_single(1.8, 2)
    r3 = bs.ppdnev_single(2.8, 2)
    result = bs.ppdnev(np.array([1.1, 1.8, 2.8]), 2)
    assert (result == np.array([r1, r2, r3])).all()


def test_ppev(t):
    bs = PPSplineF64(k=4, t=t, c=[1, 2, -1, 2, 1, 1, 2, 2.0])
    r1 = bs.ppev_single(1.1)
    r2 = bs.ppev_single(1.8)
    r3 = bs.ppev_single(2.8)
    result = bs.ppev(np.array([1.1, 1.8, 2.8]))
    assert (result == np.array([r1, r2, r3])).all()


def test_csolve():
    t = [0, 0, 0, 0, 4, 4, 4, 4]
    tau = np.array([0, 1, 3, 4])
    val = np.array([0, 0, 2, 2])
    bs = PPSplineF64(k=4, t=t, c=None)
    bs.csolve(tau, val, 0, 0, False)  # values solve spline
    result = bs.c
    expected = np.array([0.0, -1.11111111111111, 3.11111111111, 2.0], dtype=object)
    for i, res in enumerate(result):
        assert abs(expected[i] - res) < 1e-7


def test_csolve_lsq():
    t = [0, 0, 0, 0, 4, 4, 4, 4]
    tau = np.array([0, 1, 2, 3, 4])
    val = np.array([0, 0, 1.5, 2, 2])
    bs = PPSplineF64(k=4, t=t)
    bs.csolve(tau, val, 0, 0, allow_lsq=True)  # values solve spline
    result = bs.c
    expected = np.array([-0.042857, -0.7730158, 3.44920634, 1.9571428], dtype=object)
    for i, res in enumerate(result):
        assert abs(expected[i] - res) < 1e-5


@pytest.mark.parametrize(
    "tau, val, allow",
    [
        ([0, 1, 2, 3], [0, 0, 2, 2, 5], False),
        ([0, 1, 2, 3, 5], [0, 0, 2, 2], False),
        ([0, 1, 2, 3], [0, 0, 2, 2, 5], True),
    ],
)
def test_csolve_raises(tau, val, allow):
    t = [0, 0, 0, 0, 4, 4, 4, 4]
    tau = np.array(tau)
    val = np.array(val)
    bs = PPSplineF64(k=4, t=t)
    with pytest.raises(ValueError):
        bs.csolve(tau, val, 0, 0, allow_lsq=allow)


def test_copy():
    bs = PPSplineF64(k=2, t=[1, 1, 2, 3, 3], c=[1, 2, 3])
    bsc = copy.copy(bs)
    assert id(bs) != id(bsc)


def test_spline_equality_type():
    spline = PPSplineF64(k=1, t=[1, 2])
    assert not "bad" == spline

    spline2 = PPSplineF64(k=1, t=[1, 2, 3])
    assert not spline == spline2

    spline3 = PPSplineF64(k=1, t=[1, 3, 5])
    assert not spline2 == spline3

    spline4 = PPSplineF64(k=2, t=[1, 3, 5])
    assert not spline3 == spline4

    spline5 = PPSplineF64(k=2, t=[1, 3, 5])
    assert not spline4 == spline5

    spline6 = PPSplineF64(k=2, t=[1, 1, 3, 5, 5], c=[1, 2, 3])
    spline7 = PPSplineF64(k=2, t=[1, 1, 3, 5, 5], c=[1, 2, 3])
    assert spline6 == spline7
