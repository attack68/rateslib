from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import rateslib.errors as err
from rateslib.curves._parsers import _map_curve_from_solver, _validate_no_str_in_curve_input
from rateslib.enums.generics import Err, NoInput, Ok, Result, _drb

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        CurveOption,
        CurveOption_,
        Curves_,
        Curves_DiscTuple,
        FXForwards_,
        FXVolOption_,
        InstrumentCurves,
        Sequence,
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


def _get_curves_fx_vol_maybe_from_solver(
    curves_meta: Curves_,
    curves: Curves_,
    fx_vol_meta: FXVolOption_,
    fx_vol: FXVolOption_,
    fx: FX_,
    solver: Solver_,
) -> tuple[dict[str, CurveOption_], FXVolOption_, FXForwards_]:
    """
    Attempt to resolve pricing objects from given inputs or attached to a *Solver*

    Parameters
    ----------
    curves_attr : Curves
        This is an external set of Curves which is used as a substitute for pricing. These might
        be taken from an Instrument at initialisation, for example.
    solver: Solver
        Solver containing the Curves mapping
    curves: Curves
        A possible override option to allow curves to be specified directly, even if they exist
        as an attribute on the Instrument.

    Returns
    -------
    curves: 6-Tuple of Curve, dict[str, Curve], NoInput,
    fx_vol: FXVol, NoInput
    fx: FXForwards, NoInput
    """
    is_solver = not isinstance(solver, NoInput)

    # Get the `fx` from Solver only if not directly provided and Solver exists.
    fx_: FXForwards_
    if isinstance(fx, NoInput):
        if is_solver:
            fx_ = solver.fx
        else:
            fx_ = NoInput(0)
    else:
        fx_ = fx

    # Get the `curves` from a combination
    curves_: InstrumentCurves
    if isinstance(curves, NoInput) and isinstance(curves_meta, NoInput):
        # no data is available to derive curves
        curves_ = (NoInput(0), NoInput(0), NoInput(0), NoInput(0), NoInput(0), NoInput(0))
    elif isinstance(curves, NoInput):
        # set the `curves` input as that which is set as attribute at instrument init.
        curves = curves_meta

    # refactor curves into a list
    if isinstance(curves, str) or not isinstance(curves, Sequence):  # Sequence can be str!
        # convert isolated value input to list
        curves_as_list: list[
            _BaseCurve
            | dict[str, str | _BaseCurve]
            | dict[str, str]
            | dict[str, _BaseCurve]
            | NoInput
            | str
        ] = [curves]
    else:
        curves_as_list = list(curves)

    # parse curves_as_list
    if isinstance(solver, NoInput):
        curves_parsed: tuple[CurveOption_, ...] = tuple(
            _validate_no_str_in_curve_input(curve) for curve in curves_as_list
        )
    else:
        try:
            curves_parsed = tuple(_map_curve_from_solver(curve, solver) for curve in curves_as_list)
        except KeyError as e:
            raise ValueError(
                "`curves` must contain str curve `id` s existing in `solver` "
                "(or its associated `pre_solvers`).\n"
                f"The sought id was: '{e.args[0]}'.\n"
                f"The available ids are {list(solver.pre_curves.keys())}.",
            )

    curves_tuple = _make_4_tuple_of_curve(curves_parsed)
    return _validate_disc_curves_are_not_dict(curves_tuple)


def _make_4_tuple_of_curve(curves: tuple[CurveOption_, ...]) -> Curves_Tuple:
    """Convert user sequence input to a 4-Tuple."""
    n = len(curves)
    if n == 1:
        curves *= 4
    elif n == 2:
        curves *= 2
    elif n == 3:
        curves += (curves[1],)
    elif n > 4:
        raise ValueError("Can only supply a maximum of 4 `curves`.")
    return curves  # type: ignore[return-value]


def _validate_curve_is_not_dict(curve: CurveOption_) -> _BaseCurve_:
    if isinstance(curve, dict):
        raise ValueError("`disc_curve` cannot be supplied as, or inferred from, a dict of Curves.")
    return curve


def _validate_disc_curves_are_not_dict(curves_tuple: Curves_Tuple) -> Curves_DiscTuple:
    return (
        curves_tuple[0],
        _validate_curve_is_not_dict(curves_tuple[1]),
        curves_tuple[2],
        _validate_curve_is_not_dict(curves_tuple[3]),
    )


def _validate_curve_not_no_input(curve: _BaseCurve_) -> _BaseCurve:
    if isinstance(curve, NoInput):
        raise ValueError("`curve` must be supplied. Got NoInput or None.")
    return curve


T = TypeVar("T")


def _validate_obj_not_no_input(obj: T | NoInput, name: str) -> T:
    if isinstance(obj, NoInput):
        raise ValueError(f"`{name}` must be supplied. Got NoInput or None.")
    return obj


def _disc_maybe_from_curve(curve: CurveOption_, disc_curve: _BaseCurve_) -> _BaseCurve_:
    """Return a discount curve, pointed as the `curve` if not provided and if suitable Type."""
    if isinstance(disc_curve, NoInput):
        if isinstance(curve, dict):
            raise ValueError("`disc_curve` cannot be inferred from a dictionary of curves.")
        elif isinstance(curve, NoInput):
            return NoInput(0)
        elif curve._base_type == _CurveType.values:
            raise ValueError("`disc_curve` cannot be inferred from a non-DF based curve.")
        _: _BaseCurve | NoInput = curve
    else:
        _ = disc_curve
    return _


def _disc_required_maybe_from_curve(curve: CurveOption_, disc_curve: CurveOption_) -> _BaseCurve:
    """Return a discount curve, pointed as the `curve` if not provided and if suitable Type."""
    if isinstance(disc_curve, dict):
        raise NotImplementedError("`disc_curve` cannot currently be inferred from a dict.")
    _: _BaseCurve_ = _disc_maybe_from_curve(curve, disc_curve)
    if isinstance(_, NoInput):
        raise TypeError(
            "`curves` have not been supplied correctly. "
            "A `disc_curve` is required to perform function."
        )
    return _


def _try_disc_required_maybe_from_curve(
    curve: CurveOption_, disc_curve: CurveOption_
) -> Result[_BaseCurve]:
    """Return a discount curve, pointed as the `curve` if not provided and if suitable Type."""
    if isinstance(disc_curve, dict):
        return Err(NotImplementedError(err.NI_NO_DISC_FROM_DICT))
    if isinstance(disc_curve, NoInput):
        if isinstance(curve, dict):
            return Err(NotImplementedError(err.NI_NO_DISC_FROM_DICT))
        elif isinstance(curve, NoInput):
            return Err(ValueError(err.VE_NEEDS_DISC_CURVE))
        elif curve._base_type == _CurveType.values:
            return Err(ValueError(err.VE_NO_DISC_FROM_VALUES))
        return Ok(curve)
    if disc_curve._base_type == _CurveType.values:
        return Err(ValueError(err.VE_NO_DISC_FROM_VALUES))
    return Ok(disc_curve)


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


def _to_six_curve_dict(
    curves: CurveOption | list[CurveOption] | dict[str, CurveOption],
) -> dict[str, CurveOption_]:
    if isinstance(curves, list | tuple):
        if len(curves) == 1:
            return dict(
                rate=curves[0],
                disc=curves[0],
                index=NoInput(0),
                rate2=curves[0],
                disc2=curves[0],
                index2=NoInput(0),
            )
        if len(curves) == 2:
            return dict(
                rate=curves[0],
                disc=curves[1],
                index=NoInput(0),
                rate2=curves[0],
                disc2=curves[1],
                index2=NoInput(0),
            )
        if len(curves) == 3:
            return dict(
                rate=curves[0],
                disc=curves[1],
                index=curves[2],
                rate2=curves[0],
                disc2=curves[1],
                index2=curves[2],
            )
        if len(curves) == 4:
            return dict(
                rate=curves[0],
                disc=curves[1],
                index=NoInput(0),
                rate2=curves[2],
                disc2=curves[3],
                index2=NoInput(0),
            )
        if len(curves) == 5:
            return dict(
                rate=curves[0],
                disc=curves[1],
                index=curves[2],
                rate2=curves[3],
                disc2=curves[4],
                index2=curves[2],
            )
        if len(curves) == 6:
            return dict(
                rate=curves[0],
                disc=curves[1],
                index=curves[2],
                rate2=curves[3],
                disc2=curves[4],
                index2=curves[5],
            )
        else:
            raise ValueError(
                f"`curves` as sequence must not be greater than 6 in length, got: {len(curves)}."
            )
    elif isinstance(curves, dict):
        return dict(
            rate=curves.get("rate", None) or NoInput(0),
            disc=curves.get("disc", None) or curves.get("rate", None) or NoInput(0),
            index=curves.get("index", None) or NoInput(0),
            rate2=curves.get("rate2", None) or curves.get("rate", None) or NoInput(0),
            disc2=curves.get("disc2", None) or curves.get("disc", None) or NoInput(0),
            index2=curves.get("index2", None) or curves.get("index", None) or NoInput(0),
        )
    else:
        return dict(
            rate=curves,
            disc=curves,
            index=NoInput(0),
            rate2=curves,
            disc2=curves,
            index2=NoInput(0),
        )
