from __future__ import annotations

import math
from functools import partial
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

from rateslib import defaults
from rateslib.dual.variable import FLOATS, INTS, Variable
from rateslib.rs import ADOrder, Dual, Dual2, _dsolve1, _dsolve2, _fdsolve1, _fdsolve2

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        Arr1dF64,
        Arr1dObj,
        Arr2dF64,
        Arr2dObj,
        DualTypes,
        Number,
        Sequence,
    )

Dual.__doc__ = "Dual number data type to perform first derivative automatic differentiation."
Dual2.__doc__ = "Dual number data type to perform second derivative automatic differentiation."

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def _dual_float(val: DualTypes) -> float:
    """Overload for the float() builtin to handle Pyo3 issues with Variable"""
    try:
        return float(val)  # type: ignore[arg-type]
    except TypeError as e:  # val is not Number but a Variable
        if isinstance(val, Variable):
            #  This does not work well with rust.
            #  See: https://github.com/PyO3/pyo3/issues/3672
            #  and https://github.com/PyO3/pyo3/discussions/3911
            return val.real
        raise e


def _abs_float(val: DualTypes) -> float:
    """Overload the abs() builtin to return the abs of the real component only"""
    if isinstance(val, Dual | Dual2 | Variable):
        return abs(val.real)
    else:
        return abs(val)


def _get_order_of(val: DualTypes) -> int:
    """Get the AD order of a DualType including checking the globals for the current order."""
    if isinstance(val, Dual):
        ad_order: int = 1
    elif isinstance(val, Dual2):
        ad_order = 2
    elif isinstance(val, Variable):
        ad_order = defaults._global_ad_order
    else:
        ad_order = 0
    return ad_order


def _to_number(val: DualTypes) -> Number:
    """Convert a DualType to a Number Type by casting a Variable to the required global AD order."""
    if isinstance(val, Variable):
        return set_order(val, defaults._global_ad_order)
    return val


def set_order(val: DualTypes, order: int) -> Number:
    """
    Changes the order of a :class:`Dual` or :class:`Dual2` and a sets a :class:`Variable`.

    Parameters
    ----------
    val : float, Dual, Dual2, Variable
        The value to convert the order of.
    order : int in [0, 1, 2]
        The AD order to convert to. If ``val`` is float or int 0 will be used.

    Returns
    -------
    float, Dual or Dual2

    Notes
    ------
    **floats** are not affected by this function. There is no benefit to converting
    one of these types to a dual number type with no tagged variable sensitivity.

    If ``order`` is **zero**, all objects are converted to float.

    If ``order`` is **one**, *Dual2* are converted to *Dual* by dropping second order gradients.

    If ``order`` is **two**, *Dual* are converted to *DUal2* by setting second order gradients to
    default zero values.
    """
    if order == 0:
        return _dual_float(val)
    elif order == 1:
        if isinstance(val, Dual):
            return val
        elif isinstance(val, Dual2 | Variable):
            return val.to_dual()
        return val  # as float
    else:  # order == 2
        if isinstance(val, Dual2):
            return val
        elif isinstance(val, Dual | Variable):
            return val.to_dual2()
        return val  # as float


def set_order_convert(
    val: DualTypes, order: int, tag: list[str] | None, vars_from: Dual | Dual2 | None = None
) -> Number:
    """
    Convert a float, :class:`Dual` or :class:`Dual2` type to a specified alternate type with
    tagged variables.

    Parameters
    ----------
    val : float, Dual, Dual2, Variable
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

    Notes
    -----
    This function is used for AD variable management.

    ``tag`` and ``vars_from`` are only used when floats are upcast and the variables need to be
    specifically define.

    """
    if isinstance(val, FLOATS | INTS):
        _ = [] if tag is None else tag
        if order == 0:
            return float(val)
        elif order == 1:
            if vars_from is None:
                return Dual(val, _, [])
            elif isinstance(vars_from, Dual):
                return Dual.vars_from(vars_from, val, _, [])
            else:
                raise TypeError("`vars_from` must be a Dual when converting to ADOrder:1.")
        elif order == 2:
            if vars_from is None:
                return Dual2(val, _, [], [])
            elif isinstance(vars_from, Dual2):
                return Dual2.vars_from(vars_from, val, _, [], [])
            else:
                raise TypeError("`vars_from` must be a Dual2 when converting to ADOrder:2.")
    # else val is Dual or Dual2 so convert directly
    return set_order(val, order)


def gradient(
    dual: Dual | Dual2 | Variable,
    vars: Sequence[str] | None = None,  # noqa: A002
    order: int = 1,
    keep_manifold: bool = False,
) -> Arr1dF64 | Arr2dF64:
    """
    Return derivatives of a dual number.

    Parameters
    ----------
    dual : Dual, Dual2, Variable
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
    if not isinstance(dual, Dual | Dual2 | Variable):
        raise TypeError("Can call `gradient` only on dual-type variables.")
    if order == 1:
        if isinstance(dual, Variable):
            dual = Dual(dual.real, vars=dual.vars, dual=dual.dual)
        if vars is None and not keep_manifold:
            return dual.dual
        elif vars is not None and not keep_manifold:
            return dual.grad1(vars)
        elif isinstance(dual, Dual):  # and keep_manifold:
            raise TypeError("Dual type cannot perform `keep_manifold`.")
        _ = dual.grad1_manifold(dual.vars if vars is None else vars)
        return np.asarray(_)  # type: ignore[return-value]

    elif order == 2:
        if isinstance(dual, Variable):
            dual = Dual2(dual.real, vars=dual.vars, dual=dual.dual, dual2=[])
        elif isinstance(dual, Dual):
            raise TypeError("Dual type cannot derive second order automatic derivatives.")

        if vars is None:
            return 2.0 * dual.dual2  #  type: ignore[return-value]
        else:
            return dual.grad2(vars)
    else:
        raise ValueError("`order` must be in {1, 2} for gradient calculation.")


def dual_exp(x: DualTypes) -> Number:
    """
    Calculate the exponential value of a regular int or float or a dual number.

    Parameters
    ----------
    x : int, float, Dual, Dual2, Variable
        Value to calculate exponent of.

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, Dual | Dual2 | Variable):
        return x.__exp__()
    return math.exp(x)


def dual_log(x: DualTypes, base: int | None = None) -> Number:
    """
    Calculate the logarithm of a regular int or float or a dual number.

    Parameters
    ----------
    x : int, float, Dual, Dual2, Variable
        Value to calculate exponent of.
    base : int, float, optional
        Base of the logarithm. Defaults to e to compute natural logarithm

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, Dual | Dual2 | Variable):
        val = x.__log__()
        if base is None:
            return val
        else:
            return val * (1 / math.log(base))
    elif base is None:
        return math.log(x)
    else:
        return math.log(x, base)


def dual_norm_pdf(x: DualTypes) -> Number:
    """
    Return the standard normal probability density function.

    Parameters
    ----------
    x : float, Dual, Dual2, Variable

    Returns
    -------
    float, Dual, Dual2
    """
    return dual_exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def dual_norm_cdf(x: DualTypes) -> Number:
    """
    Return the cumulative standard normal distribution for given value.

    Parameters
    ----------
    x : float, Dual, Dual2, Variable

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, Dual | Dual2 | Variable):
        return x.__norm_cdf__()
    else:
        return NormalDist().cdf(x)


def dual_inv_norm_cdf(x: DualTypes) -> Number:
    """
    Return the inverse cumulative standard normal distribution for given value.

    Parameters
    ----------
    x : float, Dual, Dual2, Variable

    Returns
    -------
    float, Dual, Dual2
    """
    if isinstance(x, Dual | Dual2 | Variable):
        return x.__norm_inv_cdf__()
    else:
        return NormalDist().inv_cdf(x)


def dual_solve(
    A: Arr2dObj | Arr2dF64,
    b: Arr1dObj | Arr1dF64,
    allow_lsq: bool = False,
    types: tuple[type[float] | type[Dual] | type[Dual2], type[float] | type[Dual] | type[Dual2]] = (
        Dual,
        Dual,
    ),
) -> Arr1dObj | Arr1dF64:
    """
    Solve a linear system of equations involving dual number data types.

    The `x` value is found for the equation :math:`Ax=b`.

    .. warning::

       This method has not yet implemented :class:`~rateslib.dual.Variable` types.

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
            return np.linalg.lstsq(A, b, rcond=None)[0]  # type: ignore[arg-type,return-value]
        else:
            return np.linalg.solve(A, b)  # type: ignore[arg-type,return-value]

    # Move to Rust implementation
    if types in [(Dual, float), (Dual2, float)]:
        raise TypeError(
            "Not implemented for type crossing. Use (Dual, Dual) or (Dual2, Dual2). It is no less"
            "efficient to preconvert `b` to dual types and then solve.",
        )

    map_ = {float: 0, Dual: 1, Dual2: 2}
    A_ = np.vectorize(partial(set_order_convert, tag=[], order=map_[types[0]], vars_from=None))(A)
    b_ = np.vectorize(partial(set_order_convert, tag=[], order=map_[types[1]], vars_from=None))(b)

    a_ = [item for sublist in A_.tolist() for item in sublist]  # 1D array of A_
    b_ = b_[:, 0].tolist()

    if types == (Dual, Dual):
        return np.array(_dsolve1(a_, b_, allow_lsq))[:, None]  # type: ignore[return-value]
    elif types == (Dual2, Dual2):
        return np.array(_dsolve2(a_, b_, allow_lsq))[:, None]  # type: ignore[return-value]
    elif types == (float, Dual):
        return np.array(_fdsolve1(A_, b_, allow_lsq))[:, None]  # type: ignore[return-value]
    elif types == (float, Dual2):
        return np.array(_fdsolve2(A_, b_, allow_lsq))[:, None]  # type: ignore[return-value]
    else:
        raise TypeError(
            "Provided `types` argument are not permitted. Must be a 2-tuple with "
            "elements from {float, Dual, Dual2}"
        )


def _get_adorder(order: int) -> ADOrder:
    """Convert int AD order to an ADOrder enum type."""
    if order == 1:
        return ADOrder.One
    elif order == 0:
        return ADOrder.Zero
    elif order == 2:
        return ADOrder.Two
    else:
        raise ValueError("Order for AD can only be in {0,1,2}")


def _set_ad_order_objects(order: list[int] | dict[int, int], objs: list[Any]) -> dict[int, int]:
    """
    Set the order on multiple Objects, returning their previous order indexed my memory id.

    Parameters
    ----------
    order: list[int] or dict[int,int]
        A list of orders to set the objects to. If a dict indexed my memory id.
    objs: list[Any]
        A list of objects to convert the AD orders of.

    Returns
    -------
    dict[int]

    Notes
    -----
    If an Object does not have a `_set_ad_order` method then
    it will simply be passed and return 0 for its associated
    previous AD order.
    """
    # this function catches duplicate objects that are identical by memory id
    if isinstance(order, list) and len(order) != len(objs):
        raise ValueError("`order` and `objs` must have the same length")

    original_order: dict[int, int] = {}
    for i, obj in enumerate(objs):
        if id(obj) in original_order:
            continue  # object has already been parsed

        _ad = getattr(obj, "_ad", None)
        if _ad is None:
            # object cannot be set_ad_order
            continue

        if isinstance(order, dict):
            obj._set_ad_order(order[id(obj)])
            original_order[id(obj)] = _ad
        else:  # isinstance(order, list)
            obj._set_ad_order(order[i])
            original_order[id(obj)] = _ad

    return original_order
