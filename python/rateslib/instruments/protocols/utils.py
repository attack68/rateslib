# SPDX-License-Identifier: LicenseRef-Rateslib-Dual
#
# Copyright (c) 2026 Siffrorna Technology Limited
#
# Dual-licensed: Free Educational Licence or Paid Commercial Licence (commercial/professional use)
# Source-available, not open source.
#
# See LICENSE and https://rateslib.com/py/en/latest/i_licence.html for details,
# and/or contact info (at) rateslib (dot) com
####################################################################################################

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        CurveOption_,
    )


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


# def _map_fx_vol_or_id_from_solver_(curve: CurveOrId, solver: Solver) -> _BaseCurve:
#     """
#     Maps a "FXVol | str" to a "Curve" via a Solver mapping.
#
#     If a Curve, runs a check against whether that Curve is associated with the given Solver,
#     and perform an action based on `defaults.curve_not_in_solver`
#     """
#     if isinstance(curve, str):
#         return solver._get_pre_curve(curve)
#     elif type(curve) is ProxyCurve or type(curve) is MultiCsaCurve:
#         # TODO: (mid) consider also adding CompositeCurves as exceptions under the same rule
#         # Proxy curves and MultiCsaCurves can exist outside of Solvers but be constructed
#         # directly from an FXForwards object tied to a Solver using only a Solver's
#         # dependent curves and AD variables.
#         return curve
#     else:
#         try:
#             # it is a safeguard to load curves from solvers when a solver is
#             # provided and multiple curves might have the same id
#             __: _BaseCurve = solver._get_pre_curve(curve.id)
#             if id(__) != id(curve):  # Python id() is a memory id, not a string label id.
#                 raise ValueError(
#                     "A curve has been supplied, as part of ``curves``, which has the same "
#                     f"`id` ('{curve.id}'),\nas one of the curves available as part of the "
#                     "Solver's collection but is not the same object.\n"
#                     "This is ambiguous and cannot price.\n"
#                     "Either refactor the arguments as follows:\n"
#                     "1) remove the conflicting curve: [curves=[..], solver=<Solver>] -> "
#                     "[curves=None, solver=<Solver>]\n"
#                     "2) change the `id` of the supplied curve and ensure the rateslib.defaults "
#                     "option 'curve_not_in_solver' is set to 'ignore'.\n"
#                     "   This will remove the ability to accurately price risk metrics.",
#                 )
#             return __
#         except AttributeError:
#             raise AttributeError(
#                 "`curve` has no attribute `id`, likely it not a valid object, got: "
#                 f"{curve}.\nSince a solver is provided have you missed labelling the `curves` "
#                 f"of the instrument or supplying `curves` directly?",
#             )
#         except KeyError:
#             if defaults.curve_not_in_solver == "ignore":
#                 return curve
#             elif defaults.curve_not_in_solver == "warn":
#                 warnings.warn("`curve` not found in `solver`.", UserWarning)
#                 return curve
#             else:
#                 raise ValueError("`curve` must be in `solver`.")
