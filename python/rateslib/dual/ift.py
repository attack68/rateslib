from __future__ import annotations

from collections.abc import Callable
from time import time
from typing import TYPE_CHECKING, Any, ParamSpec
import numpy as np

from rateslib.dual.utils import _dual_float, _get_order_of, gradient
from rateslib.rs import Dual, Dual2
from rateslib.dual.newton import _dual_float_or_unchanged, _solver_result

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, Number

P = ParamSpec("P")
Q = ParamSpec("Q")


def ift_1dim(
    s: Callable[P, DualTypes],
    s_tgt: DualTypes,
    h: Callable[P, Q],
    ini_h_args: tuple[Any, ...] = (),
    max_iter: int = 50,
    func_tol: float = 1e-14,
    conv_tol: float = 1e-9,
    raise_on_fail: bool = True,
):
    r"""
    Use the inverse function theorem to implement AD safe version of a 1-dimensional root solution.

    Parameters
    ----------
    s: Callable[DualTypes, DualTypes]
        The known inverse function of *g* such that *g(s(x))=x*. Of the signature: `s(x)`.
    s_tgt: DualTypes
        The value of *s* for which *g* is to be found.
    h: Callable, string
        The iterative function to use to determine the solution g. See notes.
    ini_h_args:
        Initial arguments passed to the iterative function, ``h``.
    max_iter: int > 1
        Number of maximum iterations to perform.
    func_tol: float, optional
        The absolute function tolerance to reach before exiting.
    conv_tol: float, optional
        The convergence tolerance for subsequent iterations of *g*, passed to ``h`` to implement.
    raise_on_fail: bool, optional
        If *False* will return a solver result dict with state and message indicating failure.

    Notes
    ------
    **Mathematical background**

    This method is used to find the value of *g* from *s* in the one-dimensional equation:

    .. math::

       g(s) \qquad \text{where,} \qquad s(g) \; \text{is a known analytical inverse of} \; g.

    :math:`g(s)` is not analytical and hence requires iterations to determine.

    **What is ``h``**

    *h()* is a function that is used to perform iterations to determine *g* from *s*.

    *h* can use the iterative methods already implemented:

    - *'bisection'*: The Bisection method which requires ``ini_h_args`` to be a tuple of two floats.
      The first float is the lower bound of g. The second float is the
      upper bound of g. Bounds must provide function values of different signs.

    Or, it can be a custom function. The signature of *h* is important and must conform to:

    `h(s, s_target, conv_tol, *h_args) -> (g_i, f_i, state, *h_args_i)`

    The input parameters provide:

    - *s*: The inverse function of *g* such that *g(s(x))=x*.
    - *s_target*: The target value of *s* for which *g* is to be found.
    - *conv_tol*: The convergence tolerance which is measured internally by *h*.
    - *h_args*: Additional arguments passed to *h* which facilitate its internal operation.

    The output parameters provide:

    - *g_i*: The value of *g* at the current iteration, representative of :math:`g(s_i)`.
    - *f_i*: A measure of error in the iteration
    - *state*: A state flag return from the iteration as indicator to the controlling process.
    - *h_args_i*: Arguments passed to the next iteration of *h*.

    ``state`` flag returns are:

    - -2: The algorithm failed for an internal reason.
    - 1: `conv_tol` has been satisfied and the solution is considered to have converged.
    - None: The algorithm has not yet converged and will continue.

    **AD Implementation**

    The AD order of the solution is determined by the AD order of the ``s_tgt`` input.

    Examples
    --------
    The most prevalent use of this technique in *rateslib* is to solve bond yield-to-maturity from
    a given price. Suppose we develop a formula, *s(g)* which determines the price (*s*) of a
    2y bond with 3% annual coupon given its ytm (*g*):

    .. math::

       s(g) = \frac{3}{1+g/100} + \frac{103}{(1+g/100)^2}

    Then we use the *bisection* method to discover the ytm given a price of 101:

    .. ipython:: python

       from rateslib.dual import ift_1dim, Dual

       def s(g):
            return 3 / (1 + g / 100) + 103 / (1 + g / 100) ** 2

       # solve for a bond price of 101 with lower and upper ytm bounds of 2.0 and 3.0.
       result = ift_1dim(s, Dual(101.0, ["price"], []), "bisection", (2.0, 3.0))
       print(result)

    """
    if isinstance(h, str):
        if h == "bisection":
            h = _bisection
        else:
            raise ValueError(f"Unknown iterative function: {h}")

    t0 = time()
    i = 1

    float_ini_hargs = tuple(_dual_float_or_unchanged(_) for _ in ini_h_args)
    s0_: float = _dual_float(s_tgt)

    g0, f0, state, *hargs = h(s, s0_, conv_tol, *float_ini_hargs)
    while i < max_iter:
        if state == 1:
            break
        elif state == -2:
            if raise_on_fail:
                raise ValueError(
                    "The internal iterative function `h` has reported a iteration failure."
                )
            else:
                return _solver_result(-2, i, g0, time() - t0, log=True, algo="ift_1dim")
        if abs(f0) < func_tol:
            state = 2
            g1 = g0
            break
        g1, f1, state, *hargs = h(s, s0_, conv_tol, *hargs)
        i += 1
        g0 = g1
        f0 = f1

    if i == max_iter:
        if raise_on_fail:
            raise ValueError(
                f"`max_iter`: {max_iter} exceeded in 'ift_1dim' algorithm'.\n"
                f"Last iteration values:\nf0: {f0}\nf1: {f1}\ng0: {g0}"
            )
        else:
            return _solver_result(-1, i, g1, time() - t0, log=True, algo="ift_1dim")

    # # IFT to preserve AD
    ad_order = _get_order_of(s_tgt)
    if ad_order == 0:
        # return g1 as is.
        ret: Number = g1
    elif ad_order == 1:
        s_ = s(Dual(g1, ["x"], []))
        ds_dx = gradient(s_, vars=["x"])[0]
        ret = Dual.vars_from(s_tgt, g1, s_tgt.vars, 1.0 / ds_dx * s_tgt.dual)
    else:  # ad_order == 2
        s_ = s(Dual2(g1, ["x"], [], []))
        ds_dx = gradient(s_, vars=["x"])[0]
        d2s_dx2 = gradient(s_, vars=["x"], order=2)[0][0]
        ret = Dual2.vars_from(
            s_tgt,
            g1,
            s_tgt.vars,
            1.0 / ds_dx * s_tgt.dual,
            np.ravel(
                1.0 / ds_dx * s_tgt.dual2
                - 0.5 * d2s_dx2 * ds_dx**-3 * np.outer(s_tgt.dual, s_tgt.dual)
            ),
        )

    return _solver_result(state, i, ret, time() - t0, log=False, algo="ift_1dim")


def _bisection(
    s: Callable[P, DualTypes],
    s_target: float,
    conv_tol: float,
    g_lower: float,
    g_upper: float,
    s_lower: float | None = None,
    s_upper: float | None = None,
) -> tuple[float, float, int | None, float, float, float]:
    """
    Perform an iteration by bisection.

    The bounds `g` must yield values of `s` that are either side of the target value.

    The interval will be bisected and the side kept that envelopes the target value.

    All calculated values are returned to prevent re-calculation in the next iteration.

    The `ini_hargs` needed for this method are only (g_lower, g_upper).
    """
    if s_lower is None:
        s_lower = s(g_lower)
    if s_upper is None:
        s_upper = s(g_upper)

    f_lower = s_lower - s_target
    f_upper = s_upper - s_target

    if f_lower > 0 and f_upper > 0:
        return 0, 0, -2, 0, 0, 0  # return failed state
    elif f_lower < 0 and f_upper < 0:
        return 0, 0, -2, 0, 0, 0  # return failed state

    g_mid = (g_lower + g_upper) / 2.0
    s_mid = s(g_mid)
    f_mid = s_mid - s_target

    if (g_mid - g_lower) < conv_tol:
        state: int | None = 1
    else:
        state = None

    if abs(f_lower) < abs(f_mid):
        # g_lower is closer to the target value than g_mid
        return g_lower, f_lower, state, g_lower, g_mid, s_lower, s_mid
    elif abs(f_upper) < abs(f_mid):
        # g_upper is closer to the target value than g_mid
        return g_upper, f_upper, state, g_mid, g_upper, s_mid, s_upper
    elif abs(f_lower) < abs(f_upper):
        # g_mid is closest to the target value with g_lower being the better side
        return g_mid, f_mid, state, g_lower, g_mid, s_lower, s_mid
    else:
        return g_mid, f_mid, state, g_mid, g_upper, s_mid, s_upper
