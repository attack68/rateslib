from typing import Optional

from rateslib.dual.dual import (
    # Dual,
    # Dual2,
    dual_solve,
    set_order,
    DualTypes,
    # private methods use
    _plu_decomp,
    _pivot_matrix,
    FLOATS,
    INTS,
)
from rateslib.dual.dualrs import (
    Dual,
    Dual2,
)
import math

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def set_order_convert(val, order, tag):
    """
    Convert a float, :class:`Dual` or :class:`Dual2` type to a specified alternate type.

    Parameters
    ----------
    val : float, Dual or Dual2
        The value to convert.
    order : int
        The AD order to convert the value to if necessary.
    tag : str
        The variable name if upcasting a float to a Dual or Dual2

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(val, (*FLOATS, *INTS)):
        if order == 0:
            return val
        elif order == 1:
            return Dual(val, [tag], [])
        elif order == 2:
            return Dual2(val, [tag], [], [])
    elif isinstance(val, (Dual, Dual2)):
        if order == 0:
            return float(val)
        elif (order == 1 and isinstance(val, Dual)) or (order == 2 and isinstance(val, Dual2)):
            return val
        else:
            return val._set_order(order)


def gradient(dual, vars: Optional[list[str]] = None, order: int = 1, keep_manifold: bool = False):
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
        return dual.grad1(vars)
    elif order == 2:
        dual.grad2(vars, keep_manifold)
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