from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rateslib.dual.newton import _solver_result

if TYPE_CHECKING:
    from rateslib.typing import DualTypes

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


def quadratic_eqn(
    a: DualTypes, b: DualTypes, c: DualTypes, x0: DualTypes, raise_on_fail: bool = True
) -> dict[str, Any]:
    """
    Solve the quadratic equation, :math:`ax^2 + bx +c = 0`, with error reporting.

    Parameters
    ----------
    a: float, Dual Dual2
        The *a* coefficient value.
    b: float, Dual Dual2
        The *b* coefficient value.
    c: float, Dual Dual2
        The *c* coefficient value.
    x0: float, Dual, Dual2
        The expected solution to discriminate between two possible solutions.
    raise_on_fail: bool, optional
        Whether to raise if unsolved or return a solver result in failed state.

    Returns
    -------
    dict

    Notes
    -----
    If ``a`` is evaluated to be less that 1e-15 in absolute terms then it is treated as zero and the
    equation is solved as a linear equation in ``b`` and ``c`` only.

    Examples
    --------
    .. ipython:: python

       from rateslib.dual import quadratic_eqn

       quadratic_eqn(a=1.0, b=1.0, c=Dual(-6.0, ["c"], []), x0=-2.9)

    """
    discriminant = b**2 - 4 * a * c
    if discriminant < 0.0:
        if raise_on_fail:
            raise ValueError("`quadratic_eqn` has failed to solve: discriminant is less than zero.")
        else:
            return _solver_result(
                state=-1,
                i=0,
                func_val=1e308,
                time=0.0,
                log=True,
                algo="quadratic_eqn",
            )

    if abs(a) > 1e-15:  # machine tolerance on normal float64 is 2.22e-16
        sqrt_d = discriminant**0.5
        _1 = (-b + sqrt_d) / (2 * a)
        _2 = (-b - sqrt_d) / (2 * a)
        if abs(x0 - _1) < abs(x0 - _2):
            return _solver_result(
                state=3,
                i=1,
                func_val=_1,
                time=0.0,
                log=False,
                algo="quadratic_eqn",
            )
        else:
            return _solver_result(
                state=3,
                i=1,
                func_val=_2,
                time=0.0,
                log=False,
                algo="quadratic_eqn",
            )
    else:
        # 'a' is considered too close to zero for the quadratic eqn, solve the linear eqn
        # to avoid division by zero errors
        return _solver_result(
            state=3,
            i=1,
            func_val=-c / b,
            time=0.0,
            log=False,
            algo="quadratic_eqn->linear_eqn",
        )
