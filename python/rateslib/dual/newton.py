from __future__ import annotations

from collections.abc import Callable, Sequence
from time import time
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import numpy as np

from rateslib.dual.utils import _dual_float, dual_solve
from rateslib.dual.variable import Variable
from rateslib.rs import Dual, Dual2

if TYPE_CHECKING:
    from rateslib.typing import DualTypes
P = ParamSpec("P")

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.

STATE_MAP = {
    1: ["SUCCESS", "`conv_tol` reached"],
    2: ["SUCCESS", "`func_tol` reached"],
    3: ["SUCCESS", "closed form valid"],
    -1: ["FAILURE", "`max_iter` breached"],
    -2: ["FAILURE", "internal iteration function failure"],
}


def _solver_result(
    state: int, i: int, func_val: DualTypes, time: float, log: bool, algo: str
) -> dict[str, Any]:
    if log:
        print(
            f"{STATE_MAP[state][0]}: {STATE_MAP[state][1]} after {i} iterations "
            f"({algo}), `f_val`: {func_val}, "
            f"`time`: {time:.4f}s",
        )
    return {
        "status": STATE_MAP[state][0],
        "state": state,
        "g": func_val,
        "iterations": i,
        "time": time,
    }


T = TypeVar("T")


def _dual_float_or_unchanged(x: T | DualTypes) -> T | float:
    """If x is a DualType convert it to float otherwise leave it as is"""
    if isinstance(x, float | Dual | Dual2 | Variable):
        return _dual_float(x)
    return x


def newton_1dim(
    f: Callable[P, tuple[DualTypes, DualTypes]],
    g0: DualTypes,
    max_iter: int = 50,
    func_tol: float = 1e-14,
    conv_tol: float = 1e-9,
    args: tuple[Any, ...] = (),
    pre_args: tuple[Any, ...] = (),
    final_args: tuple[Any, ...] = (),
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """
    Use the Newton-Raphson algorithm to determine the root of a function searching **one** variable.

    Parameters
    ----------
    f: callable
        The function, *f*, to find the root of. Of the signature: `f(g, *args)`.
        Must return a tuple where the second value is the derivative of *f* with respect to *g*.
    g0: DualTypes
        Initial guess of the root. Should be reasonable to avoid failure.
    max_iter: int
        The maximum number of iterations to try before exiting.
    func_tol: float, optional
        The absolute function tolerance to reach before exiting.
    conv_tol: float, optional
        The convergence tolerance for subsequent iterations of *g*.
    args: tuple of float, Dual, Dual2 or str
        Additional arguments passed to ``f``.
    pre_args: tuple of float, Dual, Dual2 or str
        Additional arguments passed to ``f`` used only in the float solve section of
        the algorithm.
        Functions are called with the signature `f(g, *(*args[as float], *pre_args))`.
    final_args: tuple of float, Dual, Dual2 or str
        Additional arguments passed to ``f`` in the final iteration of the algorithm
        to capture AD sensitivities.
        Functions are called with the signature `f(g, *(*args, *final_args))`.
    raise_on_fail: bool, optional
        If *False* will return a solver result dict with state and message indicating failure.

    Returns
    -------
    dict

    Notes
    ------
    Solves the root equation :math:`f(g; s_i)=0` for *g*. This method is AD-safe, meaning the
    iteratively determined solution will preserve AD sensitivities, if the functions are suitable.
    Functions which are not AD suitable, such as discontinuous functions or functions with
    no derivative at given points, may yield spurious derivative results.

    This method works by first solving in the domain of floats (which is typically faster
    for most complex functions), and then performing final iterations in higher AD modes to
    capture derivative sensitivities.

    For special cases arguments can be passed separately to each of these modes using the
    ``pre_args`` and ``final_args`` arguments, rather than generically supplying it to ``args``.

    Examples
    --------
    Iteratively solve the equation: :math:`f(g, s) = g^2 - s = 0`. This has solution
    :math:`g=\\pm \\sqrt{s}` and :math:`\\frac{dg}{ds} = \\frac{1}{2 \\sqrt{s}}`.
    Thus for :math:`s=2` we expect the solution :code:`g=Dual(1.41.., ["s"], [0.35..])`.

    .. ipython:: python

       from rateslib.dual import newton_1dim

       def f(g, s):
           f0 = g**2 - s   # Function value
           f1 = 2*g        # Analytical derivative is required
           return f0, f1

       s = Dual(2.0, ["s"], [])
       newton_1dim(f, g0=1.0, args=(s,))
    """
    t0 = time()
    i = 0

    # First attempt solution using faster float calculations
    float_args = tuple(_dual_float_or_unchanged(_) for _ in args)
    g0 = _dual_float(g0)
    state = -1

    while i < max_iter:
        f0, f1 = f(*(g0, *float_args, *pre_args))  # type: ignore[call-arg]
        i += 1
        g1 = g0 - f0 / f1
        if abs(f0) < func_tol:
            state = 2
            break
        elif abs(g1 - g0) < conv_tol:
            state = 1
            break
        g0 = g1

    if i == max_iter:
        if raise_on_fail:
            raise ValueError(
                f"`max_iter`: {max_iter} exceeded in 'newton_1dim' algorithm'.\n"
                f"Last iteration values:\nf0: {f0}\nf1: {f1}\ng0: {g0}"
            )
        else:
            return _solver_result(-1, i, g1, time() - t0, log=True, algo="newton_1dim")

    # # Final iteration method to preserve AD
    f0, f1 = f(*(g1, *args, *final_args))  # type: ignore[call-arg]
    if isinstance(f0, Dual | Dual2) or isinstance(f1, Dual | Dual2):
        i += 1
        g1 = g1 - f0 / f1
    if isinstance(f0, Dual2) or isinstance(f1, Dual2):
        f0, f1 = f(*(g1, *args, *final_args))  # type: ignore[call-arg]
        i += 1
        g1 = g1 - f0 / f1

    # # Analytical approach to capture AD sensitivities
    # f0, f1 = f(g1, *(*args, *final_args))
    # if isinstance(f0, Dual):
    #     g1 = Dual.vars_from(f0, float(g1), f0.vars, float(f1) ** -1 * -gradient(f0))
    # if isinstance(f0, Dual2):
    #     g1 = Dual2.vars_from(f0, float(g1), f0.vars, float(f1) ** -1 * -gradient(f0), [])
    #     f02, f1 = f(g1, *(*args, *final_args))
    #
    #     #f0_beta = gradient(f0, order=1, vars=f0.vars, keep_manifold=True)
    #
    #     f0_gamma = gradient(f02, order=2)
    #     f0_beta = gradient(f0, order=1)
    #     # f1 = set_order_convert(g1, tag=[], order=2)
    #     f1_gamma = gradient(f1, f0.vars, order=2)
    #     f1_beta = gradient(f1, f0.vars, order=1)
    #
    #     g1_beta = -float(f1) ** -1 * f0_beta
    #     g1_gamma = (
    #         -float(f1)**-1 * f0_gamma +
    #         float(f1)**-2 * (
    #                 np.matmul(f0_beta[:, None], f1_beta[None, :]) +
    #                 np.matmul(f1_beta[:, None], f0_beta[None, :]) +
    #                 float(f0) * f1_gamma
    #         ) -
    #         2 * float(f1)**-3 * float(f0) * np.matmul(f1_beta[:, None], f1_beta[None, :])
    #     )
    #     g1 = Dual2.vars_from(f0, float(g1), f0.vars, g1_beta, g1_gamma.flatten())

    return _solver_result(state, i, g1, time() - t0, log=False, algo="newton_1dim")


def newton_ndim(
    f: Callable[P, tuple[Any, Any]],
    g0: Sequence[DualTypes],
    max_iter: int = 50,
    func_tol: float = 1e-14,
    conv_tol: float = 1e-9,
    args: tuple[Any, ...] = (),
    pre_args: tuple[Any, ...] = (),
    final_args: tuple[Any, ...] = (),
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    r"""
    Use the Newton-Raphson algorithm to determine a function root searching **many** variables.

    Solves the *n* root equations :math:`f_i(g_1, \hdots, g_n; s_k)=0` for each :math:`g_j`.

    Parameters
    ----------
    f: callable
        The function, *f*, to find the root of. Of the signature: `f([g_1, .., g_n], *args)`.
        Must return a tuple where the second value is the Jacobian of *f* with respect to *g*.
    g0: Sequence of DualTypes
        Initial guess of the root values. Should be reasonable to avoid failure.
    max_iter: int
        The maximum number of iterations to try before exiting.
    func_tol: float, optional
        The absolute function tolerance to reach before exiting.
    conv_tol: float, optional
        The convergence tolerance for subsequent iterations of *g*.
    args: tuple of float, Dual or Dual2
        Additional arguments passed to ``f``.
    pre_args: tuple
        Additional arguments passed to ``f`` only in the float solve section
        of the algorithm.
        Functions are called with the signature `f(g, *(*args[as float], *pre_args))`.
    final_args: tuple of float, Dual, Dual2
        Additional arguments passed to ``f`` in the final iteration of the algorithm
        to capture AD sensitivities.
        Functions are called with the signature `f(g, *(*args, *final_args))`.
    raise_on_fail: bool, optional
        If *False* will return a solver result dict with state and message indicating failure.

    Returns
    -------
    dict

    Examples
    --------
    Iteratively solve the equation system:

    - :math:`f_0(\mathbf{g}, s) = g_1^2 + g_2^2 + s = 0`.
    - :math:`f_1(\mathbf{g}, s) = g_1^2 - 2g_2^2 + s = 0`.

    .. ipython:: python

       from rateslib.dual import newton_ndim

       def f(g, s):
           # Function value
           f0 = g[0] ** 2 + g[1] ** 2 + s
           f1 = g[0] ** 2 - 2 * g[1]**2 - s
           # Analytical derivative as Jacobian matrix is required
           f00 = 2 * g[0]
           f01 = 2 * g[1]
           f10 = 2 * g[0]
           f11 = -4 * g[1]
           return [f0, f1], [[f00, f01], [f10, f11]]

       s = Dual(-2.0, ["s"], [])
       newton_ndim(f, g0=[1.0, 1.0], args=(s,))
    """
    t0 = time()
    i = 0
    n = len(g0)

    # First attempt solution using faster float calculations
    float_args = tuple(_dual_float_or_unchanged(_) for _ in args)
    g0_ = np.array([_dual_float(_) for _ in g0])
    state = -1

    while i < max_iter:
        f0, f1 = f(*(g0_, *float_args, *pre_args))  # type: ignore[call-arg]
        f0 = np.array(f0)[:, np.newaxis]
        f1 = np.array(f1)

        i += 1
        g1 = g0_ - np.matmul(np.linalg.inv(f1), f0)[:, 0]
        if all(abs(_) < func_tol for _ in f0[:, 0]):
            state = 2
            break
        elif all(abs(g1[_] - g0_[_]) < conv_tol for _ in range(n)):
            state = 1
            break
        g0_ = g1

    if i == max_iter:
        if raise_on_fail:
            raise ValueError(f"`max_iter`: {max_iter} exceeded in 'newton_ndim' algorithm'.")
        else:
            return _solver_result(-1, i, g1, time() - t0, log=True, algo="newton_ndim")

    # Final iteration method to preserve AD
    f0, f1 = f(*(g1, *args, *final_args))  # type: ignore[call-arg]
    f1, f0 = np.array(f1), np.array(f0)

    # get AD type
    ad: int = 0
    if _is_any_dual(f0) or _is_any_dual(f1):
        ad = 1
        DualType: type[Dual] | type[Dual2] = Dual
    elif _is_any_dual2(f0) or _is_any_dual2(f1):
        ad = 2
        DualType = Dual2

    if ad > 0:
        i += 1
        g1 = g0_ - dual_solve(f1, f0[:, None], allow_lsq=False, types=(DualType, DualType))[:, 0]  # type: ignore[arg-type]
    if ad == 2:
        f0, f1 = f(*(g1, *args, *final_args))  # type: ignore[call-arg]
        f1, f0 = np.array(f1), np.array(f0)
        i += 1
        g1 = g1 - dual_solve(f1, f0[:, None], allow_lsq=False, types=(DualType, DualType))[:, 0]  # type: ignore[arg-type]

    return _solver_result(state, i, g1, time() - t0, log=False, algo="newton_ndim")


def _is_any_dual(arr: np.ndarray[tuple[int, ...], np.dtype[np.object_]]) -> bool:
    return any(isinstance(_, Dual) for _ in arr.flatten())


def _is_any_dual2(arr: np.ndarray[tuple[int, ...], np.dtype[np.object_]]) -> bool:
    return any(isinstance(_, Dual2) for _ in arr.flatten())
