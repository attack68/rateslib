from __future__ import annotations

from collections.abc import Callable
from time import time
from typing import TYPE_CHECKING, Any, ParamSpec

import numpy as np

from rateslib.dual.newton import _dual_float_or_unchanged, _solver_result
from rateslib.dual.utils import _dual_float, _get_order_of, gradient
from rateslib.rs import Dual, Dual2

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, Number

P = ParamSpec("P")


def ift_1dim(
    s: Callable[[DualTypes], DualTypes],
    s_tgt: DualTypes,
    h: Callable[P, tuple[float, float, int, tuple[Any, ...]]] | str,
    ini_h_args: tuple[Any, ...] = (),
    max_iter: int = 50,
    func_tol: float = 1e-14,
    conv_tol: float = 1e-9,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    r"""
    Use the inverse function theorem to determine a function value of **one** variable.

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
    **Available iterative methods**

    - **'bisection'**: requires ``ini_h_args`` to be a tuple of two floats defining the interval.
      The interval will be halved in each iteration and the relevant interval side kept.
    - **'modified_dekker'**: Requires ``ini_h_args`` to be a tuple of two floats defining the
      interval. For info see
      :download:`Halving Interval for Dekker<_static/modified-dekker.pdf>`.
    - **'modified_brent'**: Requires ``ini_h_args`` to be a tuple of two floats defining the
      interval. For info see
      :download:`Halving Interval for Brent<_static/modified-dekker.pdf>`.
    - **'ytm_quadratic'**: Requires ``ini_h_args`` to be a tuple of three floats defining the
      interval and interior point. This algorithm utilises sequential quadratic approximation
      and is specifically tuned for solving bond yield-to-maturity.

    **Mathematical background**

    This method is used to find the value of *g* from *s* in the one-dimensional equation:

    .. math::

       g(s) \qquad \text{where,} \qquad s(g) \; \text{is a known analytical inverse of} \; g.

    :math:`g(s)` is not analytical and hence requires iterations to determine.

    **What is ``h``**

    *h()* is a function that is used to perform iterations to determine *g* from *s*. If
    a custom function is provided, it must conform to the following signature:

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
        h_: Callable[P, tuple[float, float, int, tuple[Any, ...]]] = ift_map[h]
    else:
        h_ = h

    t0 = time()
    i = 1

    float_ini_hargs = tuple(_dual_float_or_unchanged(_) for _ in ini_h_args)
    s0_: float = _dual_float(s_tgt)

    g0, f0, state, *hargs = h_(s, s0_, conv_tol, *float_ini_hargs)  # type: ignore[call-arg, arg-type]
    while i < max_iter:
        if state == 1:
            g1 = g0
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
        g1, f1, state, *hargs = h_(s, s0_, conv_tol, *hargs)  # type: ignore[call-arg, arg-type]
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
        s_: Dual | Dual2 = s(Dual(g1, ["x"], []))  # type: ignore[assignment]
        ds_dx = gradient(s_, vars=["x"])[0]
        ret = Dual.vars_from(s_tgt, g1, s_tgt.vars, 1.0 / ds_dx * s_tgt.dual)  # type: ignore[union-attr, arg-type]
    else:  # ad_order == 2
        s_ = s(Dual2(g1, ["x"], [], []))  # type: ignore[assignment]
        ds_dx = gradient(s_, vars=["x"])[0]
        d2s_dx2 = gradient(s_, vars=["x"], order=2)[0][0]
        ret = Dual2.vars_from(
            s_tgt,  # type: ignore[arg-type]
            g1,
            s_tgt.vars,  # type: ignore[union-attr, arg-type]
            1.0 / ds_dx * s_tgt.dual,  # type: ignore[union-attr]
            np.ravel(
                1.0 / ds_dx * s_tgt.dual2  # type: ignore[union-attr]
                - 0.5 * d2s_dx2 * ds_dx**-3 * np.outer(s_tgt.dual, s_tgt.dual)  # type: ignore[union-attr]
            ),
        )

    return _solver_result(state, i, ret, time() - t0, log=False, algo="ift_1dim")


def _bisection(
    s: Callable[[DualTypes], DualTypes],
    s_tgt: float,
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
        s_lower = s(g_lower)  # type: ignore[assignment]
    if s_upper is None:
        s_upper = s(g_upper)  # type: ignore[assignment]

    f_lower = s_lower - s_tgt  # type: ignore[operator]
    f_upper = s_upper - s_tgt  # type: ignore[operator]

    if float(f_lower * f_upper) > 0:
        return 0, 0, -2, 0, 0, 0  # return failed state

    g_mid = (g_lower + g_upper) / 2.0
    s_mid = s(g_mid)
    f_mid = s_mid - s_tgt

    if (g_mid - g_lower) < conv_tol:
        state: int | None = 1
    else:
        state = None

    if float(f_lower * f_mid) > 0:  # type: ignore[arg-type]
        # then lower and mid have same sign so must return upper interval
        if abs(f_mid) < abs(f_upper):
            return g_mid, f_mid, state, g_mid, g_upper, s_mid, s_upper  # type: ignore[return-value]
        else:
            return g_upper, f_upper, state, g_mid, g_upper, s_mid, s_upper  # type: ignore[return-value]
    else:
        # then lower and mid have opposite sign so return the lower interval
        if abs(f_mid) < abs(f_lower):
            # g_mid is closest to the target value with g_lower being the better side
            return g_mid, f_mid, state, g_lower, g_mid, s_lower, s_mid  # type: ignore[return-value]
        else:
            return g_lower, f_lower, state, g_lower, g_mid, s_lower, s_mid  # type: ignore[return-value]


def _root_f(x: float, s: Callable[[DualTypes], DualTypes], s_tgt: float) -> float:
    """Root reformulation for Dekker's algorithm"""
    return s(x) - s_tgt  # type: ignore[return-value]


def _dekker(
    s: Callable[[DualTypes], DualTypes],
    s_tgt: float,
    conv_tol: float,
    a_k: float,
    b_k: float,
    b_k_: float | None = None,
    cached_f_a_k: float | None = None,
    cached_f_b_k: float | None = None,
    cached_f_b_k_: float | None = None,
) -> tuple[float, float, int | None, float, float, float, float, float, float]:
    """
    Alternative root solver.
    See docs/source/_static/modified-dekker.pdf for details.

    Cached values allow value transmission from one function to the next with many efficiencies.
    """

    # Load cached values
    if cached_f_a_k is None:
        f_a_k = _root_f(a_k, s, s_tgt)
    else:
        f_a_k = cached_f_a_k

    if cached_f_b_k is None:
        f_b_k = _root_f(b_k, s, s_tgt)
    else:
        f_b_k = cached_f_b_k

    if abs(a_k - b_k) < conv_tol:
        return b_k, f_b_k, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # for the first iteration set b_k_m1 equal to a_k, else it is returned from previous
    if b_k_ is None:
        # then the first iteration is active (b_k_ is only None once), also check for side
        if abs(f_a_k) < abs(f_b_k):
            # switch to make b_k the 'best' solution
            f_a_k, f_b_k = f_b_k, f_a_k
            a_k, b_k = b_k, a_k

        b_k_m1: float = a_k
        f_b_k_m1 = f_a_k
    else:
        b_k_m1 = b_k_
        if cached_f_b_k_ is None:
            f_b_k_m1 = _root_f(b_k_m1, s, s_tgt)
        else:
            f_b_k_m1 = cached_f_b_k_

    # provisional values for the next iteration
    m = (a_k + b_k) / 2.0  # midpoint
    q = b_k - f_b_k * (b_k - b_k_m1) / (f_b_k - f_b_k_m1)  # secant

    if q >= min(b_k, m) and q <= max(b_k, m):
        b_k_p1 = q
    else:
        b_k_p1 = m

    f_b_k_p1 = _root_f(b_k_p1, s, s_tgt)

    # determine a_k_p1
    a_k_p1 = a_k
    f_a_k_p1 = f_a_k
    if float(f_a_k * f_b_k_p1) > 0:
        a_k_p1 = b_k
        f_a_k_p1 = f_b_k
    elif q >= min(b_k, m) and q <= max(b_k, m):
        f_m = _root_f(m, s, s_tgt)
        if float(f_m * f_b_k_p1) < 0:
            a_k_p1 = m
            f_a_k_p1 = f_m

    if abs(f_a_k_p1) < abs(f_b_k_p1):
        # switch to make b_k the 'best' solution
        f_a_k_p1, f_b_k_p1 = f_b_k_p1, f_a_k_p1
        a_k_p1, b_k_p1 = b_k_p1, a_k_p1
        # also switch the existing values
        f_a_k, f_b_k = f_b_k, f_a_k
        a_k, b_k = b_k, a_k

    return b_k_p1, f_b_k_p1, None, a_k_p1, b_k_p1, b_k, f_a_k_p1, f_b_k_p1, f_b_k


def _brent(
    s: Callable[[DualTypes], DualTypes],
    s_tgt: float,
    conv_tol: float,
    a_k: float,
    b_k: float,
    b_k_: float | None = None,
    cached_f_a_k: float | None = None,
    cached_f_b_k: float | None = None,
    cached_f_b_k_: float | None = None,
) -> tuple[float, float, int | None, float, float, float, float, float, float]:
    """
    Alternative root solver.
    See docs/source/_static/modified-dekker.pdf for details.

    Cached values allow value transmission from one function to the next with many efficiencies.
    """

    # Load cached values
    if cached_f_a_k is None:
        f_a_k = _root_f(a_k, s, s_tgt)
    else:
        f_a_k = cached_f_a_k

    if cached_f_b_k is None:
        f_b_k = _root_f(b_k, s, s_tgt)
    else:
        f_b_k = cached_f_b_k

    if abs(a_k - b_k) < conv_tol:
        return b_k, f_b_k, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # for the first iteration set b_k_m1 equal to a_k, else it is returned from previous
    if b_k_ is None:
        # then the first iteration is active (b_k_ is only None once) so also check the side
        if abs(f_a_k) < abs(f_b_k):
            # switch to make b_k the 'best' solution
            f_a_k, f_b_k = f_b_k, f_a_k
            a_k, b_k = b_k, a_k

        b_k_m1: float = a_k
        f_b_k_m1 = f_a_k
    else:
        b_k_m1 = b_k_
        if cached_f_b_k_ is None:
            f_b_k_m1 = _root_f(b_k_m1, s, s_tgt)
        else:
            f_b_k_m1 = cached_f_b_k_

    # provisional values for the next iteration
    if b_k == b_k_m1 or a_k == b_k_m1:
        denom = f_b_k - f_b_k_m1
        q = b_k - f_b_k * (b_k - b_k_m1) / denom  # secant
    else:
        fba = f_b_k / f_a_k
        fbbm = f_b_k / f_b_k_m1
        fabm = f_a_k / f_b_k_m1
        denom = (fbbm - 1.0) * (fba - 1.0) * (fabm - 1.0)
        q = b_k + fba * ((1.0 - fbbm) * (a_k - b_k) + fabm * (fbbm - fabm) * (b_k_m1 - b_k)) / denom

    if q <= min(b_k, (3.0 * a_k + b_k) / 4.0) or q >= max(b_k, (3.0 * a_k + b_k) / 4.0):
        q = (a_k + b_k) / 2.0

    b_k_p1 = q
    f_b_k_p1 = _root_f(b_k_p1, s, s_tgt)

    a_k_p1 = a_k
    f_a_k_p1 = f_a_k
    if float(f_a_k * f_b_k_p1) > 0:
        a_k_p1 = b_k
        f_a_k_p1 = f_b_k
    else:
        m = (a_k + b_k) / 2.0
        f_m = _root_f(m, s, s_tgt)
        if float(f_m * f_b_k_p1) < 0:
            a_k_p1 = m
            f_a_k_p1 = f_m

    if abs(f_a_k_p1) < abs(f_b_k_p1):
        # switch to make b_k the 'best' solution
        f_a_k_p1, f_b_k_p1 = f_b_k_p1, f_a_k_p1
        a_k_p1, b_k_p1 = b_k_p1, a_k_p1
        # also switch the existing values
        f_a_k, f_b_k = f_b_k, f_a_k
        a_k, b_k = b_k, a_k

    return b_k_p1, f_b_k_p1, None, a_k_p1, b_k_p1, b_k, f_a_k_p1, f_b_k_p1, f_b_k


def _ytm_quadratic(
    s: Callable[[DualTypes], DualTypes],
    s_tgt: float,
    conv_tol: float,
    g0: float,
    g1: float,
    g2: float,
    cached_f0: float | None = None,
    cached_f1: float | None = None,
    cached_f2: float | None = None,
) -> tuple[float, float, int | None, float, float, float, float | None, float | None, float | None]:
    """
    Alternative root solver.
    See docs/source/_static/modified-dekker.pdf for details.

    Cached values allow value transmission from one function to the next with many efficiencies.
    """

    # Load cached values
    f0: float = cached_f0 if cached_f0 is not None else _root_f(g0, s, s_tgt)
    f1: float = cached_f1 if cached_f1 is not None else _root_f(g1, s, s_tgt)
    f2: float = cached_f2 if cached_f2 is not None else _root_f(g2, s, s_tgt)

    # Test interval
    if f0 < 0 and f1 < 0 and f2 < 0:
        # then g(s*) must be
        g0_ = g0 - (g2 - g0)
        g1_ = g1 - (g2 - g1)
        g2_ = g0
        return g1_, 1e9, None, g0_, g1_, g2_, None, None, None
    elif f0 > 0 and f1 > 0 and f2 > 0:
        g0_ = g2
        g1_ = g1 + (g2 - g0)
        g2_ = g2 + 2 * (g2 - g0)
        return g1_, 1e9, None, g0_, g1_, g2_, None, None, None

    # Solve g_new via quadratic approximation

    # # Linear algebra solution
    # _b = np.array([g0, g1, g2])[:, None]
    # _A = np.array([[f0**2, f0, 1], [f1**2, f1, 1], [f2**2, f2, 1]])
    # x = np.linalg.solve(_A, _b)
    # g_new = x[2, 0]

    # Analytical solution
    f012, f022, f01, f02, g01, g02 = (
        f0**2 - f1**2,
        f0**2 - f2**2,
        f0 - f1,
        f0 - f2,
        g0 - g1,
        g0 - g2,
    )
    x0 = (g01 * f02 - g02 * f01) / (f012 * f02 - f022 * f01)
    x1 = (g01 - x0 * f012) / f01
    x2 = g0 - x1 * f0 - x0 * f0**2
    g_new = x2

    if g_new < g0 or g_new > g2:
        # if the quadratic approximation is outside the interval then use a bisection method
        if f0 * f1 < 0:
            # bisect in the left hand side
            g_new = g0 + (g1 - g0) * f0 / (f0 - f1)
        else:
            # bisect in the right hand side
            g_new = g1 - (g2 - g1) * f1 / (f2 - f1)

    f_new = _root_f(g_new, s, s_tgt)
    for g_ in [g0, g1, g2]:
        if abs(g_ - g_new) < conv_tol:
            return g_new, f_new, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if g0 < g_new and g_new < g1:
        return g_new, f_new, None, g0, g_new, g1, f0, f_new, f1
    else:  # g1 < g_new and g_new < g2:
        return g_new, f_new, None, g1, g_new, g2, f1, f_new, f2
    # else:
    #     raise RuntimeError("Unexpected interval: this line should never be reached.")


ift_map: dict[str, Callable[P, tuple[float, float, int, tuple[Any, ...]]]] = {
    "bisection": _bisection,  # type: ignore[dict-item]
    "modified_dekker": _dekker,  # type: ignore[dict-item]
    "modified_brent": _brent,  # type: ignore[dict-item]
    "ytm_quadratic": _ytm_quadratic,  # type: ignore[dict-item]
}
