from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from rateslib.enums.generics import NoInput, _drb

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
        FXVolOption_,
        Solver_,
        _BaseCurve,
        _Vol,
    )


def _get_fx_vol_maybe_from_solver(
    vol_meta: _Vol, vol: _Vol, name: str, solver: Solver_
) -> FXVolOption_:
    vol_ = _drb(getattr(vol_meta, name), getattr(vol, name))
    if isinstance(solver, NoInput):
        if isinstance(vol_, str):
            raise ValueError("`vol` must contain FXVolObj, not str, if `solver` not given.")
        return vol_
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

    if vol is None:  # capture blank user input and reset
        vol = NoInput(0)

    if isinstance(vol, NoInput):
        if isinstance(vol_attr, NoInput):
            return NoInput(0)
        else:
            vol_: FXVol = vol_attr
    else:
        vol_ = vol

    if isinstance(solver, NoInput):
        if isinstance(vol_, str):
            raise ValueError(
                "String `vol` ids require a `solver` to be mapped. No `solver` provided.",
            )
        return vol_
    elif isinstance(vol_, float | Dual | Dual2 | Variable):
        return vol_
    elif isinstance(vol_, str):
        return solver._get_pre_fxvol(vol_)
    else:  # vol is a Smile or Surface - check that it is in the Solver
        try:
            # it is a safeguard to load curves from solvers when a solver is
            # provided and multiple curves might have the same id
            _: FXDeltaVolSmile | FXDeltaVolSurface | FXSabrSmile | FXSabrSurface = (
                solver._get_pre_fxvol(vol_.id)
            )
            if id(_) != id(vol_):
                raise ValueError(  # ignore: type[union-attr]
                    "A ``vol`` object has been supplied which has the same "
                    f"`id` ('{vol_.id}'),\nas one of those available as part of the "
                    "Solver's collection but is not the same object.\n"
                    "This is ambiguous and may lead to erroneous prices.\n",
                )
            return _
        except AttributeError:
            raise AttributeError(
                "`vol` has no attribute `id`, likely it not a valid object, got: "
                f"{vol_}.\nSince a solver is provided have you missed labelling the `vol` "
                f"of the instrument or supplying `vol` directly?",
            )
        except KeyError:
            if defaults.curve_not_in_solver == "ignore":
                return vol_
            elif defaults.curve_not_in_solver == "warn":
                warnings.warn("`vol` not found in `solver`.", UserWarning)
                return vol_
            else:
                raise ValueError("`vol` must be in `solver`.")


# def _get_fx_maybe_from_solver(
#     fx: FX_,
#     solver: Solver_,
# ) -> FX_:
#     # Get the `fx` from Solver only if not directly provided and Solver exists.
#     fx_: FXForwards_
#     if isinstance(fx, NoInput):
#         if not isinstance(solver, NoInput):
#             fx_ = solver.fx
#         else:
#             fx_ = NoInput(0)
#     else:
#         fx_ = fx
#     return fx_


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


def _map_fx_vol_or_id_from_solver_(curve: CurveOrId, solver: Solver) -> _BaseCurve:
    """
    Maps a "FXVol | str" to a "Curve" via a Solver mapping.

    If a Curve, runs a check against whether that Curve is associated with the given Solver,
    and perform an action based on `defaults.curve_not_in_solver`
    """
    if isinstance(curve, str):
        return solver._get_pre_curve(curve)
    elif type(curve) is ProxyCurve or type(curve) is MultiCsaCurve:
        # TODO: (mid) consider also adding CompositeCurves as exceptions under the same rule
        # Proxy curves and MultiCsaCurves can exist outside of Solvers but be constructed
        # directly from an FXForwards object tied to a Solver using only a Solver's
        # dependent curves and AD variables.
        return curve
    else:
        try:
            # it is a safeguard to load curves from solvers when a solver is
            # provided and multiple curves might have the same id
            __: _BaseCurve = solver._get_pre_curve(curve.id)
            if id(__) != id(curve):  # Python id() is a memory id, not a string label id.
                raise ValueError(
                    "A curve has been supplied, as part of ``curves``, which has the same "
                    f"`id` ('{curve.id}'),\nas one of the curves available as part of the "
                    "Solver's collection but is not the same object.\n"
                    "This is ambiguous and cannot price.\n"
                    "Either refactor the arguments as follows:\n"
                    "1) remove the conflicting curve: [curves=[..], solver=<Solver>] -> "
                    "[curves=None, solver=<Solver>]\n"
                    "2) change the `id` of the supplied curve and ensure the rateslib.defaults "
                    "option 'curve_not_in_solver' is set to 'ignore'.\n"
                    "   This will remove the ability to accurately price risk metrics.",
                )
            return __
        except AttributeError:
            raise AttributeError(
                "`curve` has no attribute `id`, likely it not a valid object, got: "
                f"{curve}.\nSince a solver is provided have you missed labelling the `curves` "
                f"of the instrument or supplying `curves` directly?",
            )
        except KeyError:
            if defaults.curve_not_in_solver == "ignore":
                return curve
            elif defaults.curve_not_in_solver == "warn":
                warnings.warn("`curve` not found in `solver`.", UserWarning)
                return curve
            else:
                raise ValueError("`curve` must be in `solver`.")
