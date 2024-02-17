from rateslib.dual.dual import (
    Dual,
    Dual2,
    dual_solve,
    set_order,
    DualTypes,
    # private methods use
    _plu_decomp,
    FLOATS,
    INTS,
)
# from rateslibrs import Dual
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
            return Dual2(val, tag)
    elif isinstance(val, (Dual, Dual2)):
        if order == 0:
            return float(val)
        elif (order == 1 and isinstance(val, Dual)) or (order == 2 and isinstance(val, Dual2)):
            return val
        else:
            return val._set_order(order)


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