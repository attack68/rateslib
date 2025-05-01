from __future__ import annotations

from collections.abc import Callable
from time import time
from typing import TYPE_CHECKING, Any, ParamSpec
import numpy as np

from rateslib.dual.utils import _dual_float, _get_order_of
from rateslib.dual import gradient
from rateslib.rs import Dual, Dual2
from rateslib.dual.newton import _dual_float_or_unchanged, _solver_result

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, Number

P = ParamSpec("P")
Q = ParamSpec("Q")

def ift_1dim(
    s: Callable[P, DualTypes],
    s0: DualTypes,
    h: Callable[P, Q],
    ini_hargs: tuple[Any, ...] = (),
    max_iter: int = 50,
    func_tol: float = 1e-14,
    conv_tol: float = 1e-9,
    raise_on_fail: bool = True,
):
    """
    Use the inverse function theorem to implement AD safe version of a 1-dimensional root solution.
    
    Parameters
    ----------
    s: Callable[DualTypes, DualTypes]
        The known inverse function of *g* such that *g(s(x))=x*. Of the signature: `s(x)`.
    s_target: DualTypes
        The target value of *s* for which *g* is to be found.
    h: Callable, string
        The iterative function to use to determine the solution g.
        Of the signature: `h(s, s_target, *hargs) -> (g_i, f_i, *hargs_i)`.
        Must return the value of *g* at the current iteration, the current difference between
        s and s_target and ``hargs`` to be used in the next iteration.
    ini_hargs:
        Initial arguments passed to the iterative function, ``h``.
    max_iter: int > 1
        Number of maiximum iterations to perform.

    Notes
    ------
    **Mathematical background**

    This method is used to find the value of *g* from *s* in the one-dimensional equation:

    .. math::

       g(s) \qquad \text{where,} \qquad s(g) \; \text{is a known analytical function of} \; g.

    :math:`g(s)` is not analytical and hence requires iterations to determine.

    **What is ``h``**

    *h()* is a function that is used to perform iterations to determine *g* from *s*.

    *h* can use the iterative methods already implemented, such as "bisection" or "brent", or
    it can be a custom function. The signature of *h* is important, and must conform to:
    `h(s, s_target, conv_tol, *h_args) -> (g_i, f_i, tol, *h_args_i)`

    The input parameters provide:

    - *s*: The inverse function of *g* such that *g(s(x))=x*.
    - *s_target*: The target value of *s* for which *g* is to be found.
    - *conv_tol*: The convergence tolerance which is measured internally by *h*.
    - *h_args*: Additional arguments passed to *h* which facilitate its internal operation.

    The output parameters provide:

    - *g_i*: The value of *g* at the current iteration, representative of :math:`g(s_i)`.
    - *f_i*: A measure of error in the iteration
    - *tol*: A bool describing whether ``conv_tol`` has been reached. *True* breaks the iteration.
    - *h_args_i*: Arguments passed to the next iteration of *h*.

    **AD Implementation**

    The AD order of the solution is determined by the AD order of the ``s_target``.
    """
    if isinstance(h, str):
        if h == "bisection":
            h = _bisection
        else:
            raise ValueError(f"Unknown iterative function: {h}")

    t0 = time()
    i = 1

    # First attempt solution using faster float calculations
    float_ini_hargs = tuple(_dual_float_or_unchanged(_) for _ in ini_hargs)
    s0_: float = _dual_float(s0)
    state = -1

    g0, f0, c_tol, *hargs = h(s, s0_, conv_tol, *float_ini_hargs)
    while i < max_iter:
        if abs(f0) < func_tol:
            state = 2
            g1 = g0
            break
        g1, f1, c_tol, *hargs = h(s, s0_, conv_tol, *hargs)
        i += 1
        if c_tol:
            state = 1
            break
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
    ad_order = _get_order_of(s0)
    if ad_order == 0:
        # return g1 as is.
        ret: Number = g1
    elif ad_order == 1:
        s_ = s(Dual(g1, ["x"], []))
        ds_dx = gradient(s_, vars=["x"])[0]
        ret = Dual.vars_from(s0, g1, s0.vars, 1.0 / ds_dx * s0.dual)
    else: # ad_order == 2
        s_ = s(Dual2(g1, ["x"], [], []))
        ds_dx = gradient(s_, vars=["x"])[0]
        d2s_dx2 = gradient(s_, vars=["x"], order=2)[0][0]
        ret = Dual.vars_from(
            s0,
            g1,
            s0.vars,
            1.0 / ds_dx * s0.dual,
            np.ravel(1.0 / ds_dx * s0.dual2 - 0.5 * d2s_dx2 * ds_dx ** -3 * np.outer(s0.dual, s0.dual))
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
) -> tuple[float, float, float, float, float, float]:
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
        raise ValueError("Provided bounds, `g_lower` and `g_upper`, do not provide a valid root.")
    elif f_lower < 0 and f_upper < 0:
        raise ValueError("Provided bounds, `g_lower` and `g_upper`, do not provide a valid root.")

    g_mid = (g_lower + g_upper) / 2.0
    s_mid = s(g_mid)
    f_mid = s_mid - s_target

    tol = (g_mid - g_lower) < conv_tol

    if abs(f_lower) < abs(f_mid):
        # g_lower is closer to the target value than g_mid
        return g_lower, f_lower, tol, g_lower, g_mid, s_lower, s_mid
    elif abs(f_upper) < abs(f_mid):
        # g_upper is closer to the target value than g_mid
        return g_upper, f_upper, tol, g_mid, g_upper, s_mid, s_upper
    elif abs(f_lower) < abs(f_upper):
        # g_mid is closest to the target value with g_lower being the better side
        return g_mid, f_mid, tol, g_lower, g_mid, s_lower, s_mid
    else:
        return g_mid, f_mid, tol, g_mid, g_upper, s_mid, s_upper
