from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from rateslib.curves._parsers import _map_curve_from_solver, _validate_no_str_in_curve_input
from rateslib.enums.generics import NoInput, _drb

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        CurveOption_,
        FXForwards_,
        Solver_,
        _BaseCurve,
        _BaseCurve_,
        _Curves,
    )


def _get_curve_maybe_from_solver(
    curves_meta: _Curves,
    curves: _Curves,
    name: str,
    solver: Solver_,
) -> CurveOption_:
    curve = _drb(getattr(curves_meta, name), getattr(curves, name))
    if isinstance(solver, NoInput):
        return _validate_no_str_in_curve_input(curve)
    else:
        try:
            mapped_curve = _map_curve_from_solver(curve, solver)
            return mapped_curve
        except KeyError as e:
            raise ValueError(
                "`curves` must contain str curve `id` s existing in `solver` "
                "(or its associated `pre_solvers`).\n"
                f"The sought id was: '{e.args[0]}'.\n"
                f"The available ids are {list(solver.pre_curves.keys())}.",
            )


def _get_fx_maybe_from_solver(
    fx: FX_,
    solver: Solver_,
) -> FX_:
    # Get the `fx` from Solver only if not directly provided and Solver exists.
    fx_: FXForwards_
    if isinstance(fx, NoInput):
        if not isinstance(solver, NoInput):
            fx_ = solver.fx
        else:
            fx_ = NoInput(0)
    else:
        fx_ = fx
    return fx_


def _validate_curve_is_not_dict(curve: CurveOption_) -> _BaseCurve_:
    if isinstance(curve, dict):
        raise ValueError("`disc_curve` cannot be supplied as, or inferred from, a dict of Curves.")
    return curve


def _validate_curve_not_no_input(curve: _BaseCurve_) -> _BaseCurve:
    if isinstance(curve, NoInput):
        raise ValueError("`curve` must be supplied. Got NoInput or None.")
    return curve


T = TypeVar("T")


def _validate_obj_not_no_input(obj: T | NoInput, name: str) -> T:
    if isinstance(obj, NoInput):
        raise ValueError(f"`{name}` must be supplied. Got NoInput or None.")
    return obj


def _maybe_set_ad_order(
    curve: CurveOption_, order: int | dict[str, int | None] | None
) -> int | dict[str, int | None] | None:
    """method is used internally to set AD order and then later revert the curve to its original"""
    if isinstance(curve, NoInput) or order is None:
        return None  # do nothing
    else:
        if isinstance(curve, dict):
            # method will return a dict of orders if a dict of curves is provided as input
            if isinstance(order, dict):
                return {
                    k: _maybe_set_ad_order(v, order[k])  # type: ignore[misc]
                    for k, v in curve.items()
                }
            else:
                return {
                    k: _maybe_set_ad_order(v, order)  # type: ignore[misc]
                    for k, v in curve.items()
                }
        else:
            try:
                original_order = curve.ad
                curve._set_ad_order(order)  # type: ignore[arg-type]
            except AttributeError:
                # Curve has no method (possibly a custom curve and not a subclass of _BaseCurve)
                return None
            return original_order
