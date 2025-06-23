from __future__ import annotations

import math
from functools import partial
from statistics import NormalDist
from typing import Union

import numpy as np

from rateslib.rs import ADOrder, Dual, Dual2, _dsolve1, _dsolve2, _fdsolve1, _fdsolve2

Dual.__doc__ = "Dual number data type to perform first derivative automatic differentiation."
Dual2.__doc__ = "Dual number data type to perform second derivative automatic differentiation."

FLOATS = (float, np.float16, np.float32, np.float64, np.longdouble)
INTS = (int, np.int8, np.int16, np.int32, np.int32, np.int64)

DualTypes = Union[float, Dual, Dual2]

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def set_order(val, order):
    """
    Changes the order of a :class:`Dual` or :class:`Dual2` leaving floats and ints
    unchanged.

    Parameters
    ----------
    val : float, int, Dual or Dual2
        The value to convert the order of.
    order : int in [0, 1, 2]
        The AD order to convert to. If ``val`` is float or int 0 will be used.

    Returns
    -------
    float, int, Dual or Dual2
    """
    if order == 2 and isinstance(val, Dual):
        return val.to_dual2()
    elif order == 1 and isinstance(val, Dual2):
        return val.to_dual()
    elif order == 0:
        return float(val)
    # otherwise:
    #  - val is a Float or an Int
    #  - val is a Dual and order == 1 OR val is Dual2 and order == 2
    return val


def set_order_convert(val, order, tag, vars_from=None):
    """
    Convert a float, :class:`Dual` or :class:`Dual2` type to a specified alternate type.

    Parameters
    ----------
    val : float, Dual or Dual2
        The value to convert.
    order : int
        The AD order to convert the value to if necessary.
    tag : list of str, optional
        The variable name(s) if upcasting a float to a Dual or Dual2
    vars_from : optional, Dual or Dual2
        A pre-existing Dual of correct order from which the Vars are extracted. Improves efficiency
        when given.

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(val, (*FLOATS, *INTS)):
        _ = [] if tag is None else tag
        if order == 0:
            return float(val)
        elif order == 1:
            if vars_from is None:
                return Dual(val, _, [])
            else:
                return Dual.vars_from(vars_from, val, _, [])
        elif order == 2:
            if vars_from is None:
                return Dual2(val, _, [], [])
            else:
                return Dual2.vars_from(vars_from, val, _, [], [])
    # else val is Dual or Dual2 so convert directly
    return set_order(val, order)


def gradient(dual, vars: list[str] | None = None, order: int = 1, keep_manifold: bool = False):
    """
    Return derivatives of a dual number.

    Parameters
    ----------
    dual : Dual or Dual2
        The dual variable from which to derive derivatives.
    vars : str, tuple, list optional
        Name of the variables which to return gradients for. If not given
        defaults to all vars attributed to the instance.
    order : {1, 2}
        Whether to return the first or second derivative of the dual number.
        Second order will raise if applied to a ``Dual`` and not ``Dual2`` instance.
    keep_manifold : bool
        If ``order`` is 1 and the type is ``Dual2`` one can return a ``Dual2``
        where the ``dual2`` values are converted to ``dual`` values to represent
        a first order manifold of the first derivative (and the ``dual2`` values
        set to zero). Useful for propagation in iterations.

    Returns
    -------
    float, ndarray, Dual2
    """
    if not isinstance(dual, (Dual, Dual2)):
        raise TypeError("Can call `gradient` only on dual-type variables.")
    if order == 1:
        if vars is None and not keep_manifold:
            return dual.dual
        elif vars is not None and not keep_manifold:
            return dual.grad1(vars)

        _ = dual.grad1_manifold(vars)
        return np.asarray(_)

    elif order == 2:
        if vars is None:
            return 2.0 * dual.dual2
        else:
            return dual.grad2(vars)
    else:
        raise ValueError("`order` must be in {1, 2} for gradient calculation.")


def dual_exp(x):
    """
    Calculate the exponential value of a regular int or float or a dual number.

    Parameters
    ----------
    x : int, float, Dual, Dual2
        Value to calculate exponent of.

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, (Dual, Dual2)):
        return x.__exp__()
    return math.exp(x)


def dual_log(x, base=None):
    """
    Calculate the logarithm of a regular int or float or a dual number.

    Parameters
    ----------
    x : int, float, Dual, Dual2
        Value to calculate exponent of.
    base : int, float, optional
        Base of the logarithm. Defaults to e to compute natural logarithm

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, (Dual, Dual2)):
        val = x.__log__()
        if base is None:
            return val
        else:
            return val * (1 / math.log(base))
    elif base is None:
        return math.log(x)
    else:
        return math.log(x, base)


def dual_norm_pdf(x):
    """
    Return the standard normal probability density function.

    Parameters
    ----------
    x : float, Dual, Dual2

    Returns
    -------
    float, Dual, Dual2
    """
    return dual_exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def dual_norm_cdf(x):
    """
    Return the cumulative standard normal distribution for given value.

    Parameters
    ----------
    x : float, Dual, Dual2

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, (Dual, Dual2)):
        return x.__norm_cdf__()
    else:
        return NormalDist().cdf(x)


def dual_inv_norm_cdf(x):
    """
    Return the inverse cumulative standard normal distribution for given value.

    Parameters
    ----------
    x : float, Dual, Dual2

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, (Dual, Dual2)):
        return x.__norm_inv_cdf__()
    else:
        return NormalDist().inv_cdf(x)


def dual_solve(A, b, allow_lsq=False, types=(Dual, Dual)):
    """
    Solve a linear system of equations involving dual number data types.

    The `x` value is found for the equation :math:`Ax=b`.

    Parameters
    ----------
    A: 2-d array
        Left side matrix of values.
    b: 1-d array
        Right side vector of values.
    allow_lsq: bool
        Whether to allow solutions for non-square `A`, i.e. when `len(b) > len(x)`.
    types: tuple
        Defining the input data type elements of `A` and `b`, e.g. (float, float) or (Dual, Dual).

    Returns
    -------
    1-d array
    """
    if types == (float, float):
        # Use basic Numpy LinAlg
        if allow_lsq:
            return np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            return np.linalg.solve(A, b)

    # Move to Rust implementation
    if types in [(Dual, float), (Dual2, float)]:
        raise TypeError(
            "Not implemented for type crossing. Use (Dual, Dual) or (Dual2, Dual2). It is no less"
            "efficient to preconvert `b` to dual types and then solve."
        )

    map = {float: 0, Dual: 1, Dual2: 2}
    A_ = np.vectorize(partial(set_order_convert, tag=[], order=map[types[0]], vars_from=None))(A)
    b_ = np.vectorize(partial(set_order_convert, tag=[], order=map[types[1]], vars_from=None))(b)

    a = [item for sublist in A_.tolist() for item in sublist]  # 1D array of A_
    b = b_[:, 0].tolist()

    if types == (Dual, Dual):
        out = _dsolve1(a, b, allow_lsq)
    elif types == (Dual2, Dual2):
        out = _dsolve2(a, b, allow_lsq)
    elif types == (float, Dual):
        out = _fdsolve1(A_, b, allow_lsq)
    elif types == (float, Dual2):
        out = _fdsolve2(A_, b, allow_lsq)

    return np.array(out)[:, None]


def _get_adorder(order: int):
    if order == 1:
        return ADOrder.One
    elif order == 0:
        return ADOrder.Zero
    elif order == 2:
        return ADOrder.Two
    else:
        raise ValueError("Order for AD can only be in {0,1,2}")
