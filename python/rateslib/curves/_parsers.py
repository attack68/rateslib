from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves import MultiCsaCurve, ProxyCurve
from rateslib.default import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        Curve,
        CurveInput,
        CurveInput_,
        CurveOption,
        CurveOption_,
        CurveOrId_,
        Solver,
    )


def _map_curve_or_id_from_solver_(curve: CurveOrId_, solver: Solver) -> Curve:
    """
    Maps a "Curve | str" to a "Curve" via a Solver mapping.

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
            __: Curve = solver._get_pre_curve(curve.id)
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


def _map_curve_from_solver_(curve: CurveInput_, solver: Solver) -> CurveOption_:
    """
    Maps a "Curve | str | dict[str, Curve | str]" to a "Curve | dict[str, Curve]" via a Solver.

    If curve input involves strings get objects directly from solver curves mapping.

    This is the explicit variety which does not handle NoInput.
    """
    if isinstance(curve, dict):
        mapped_dict: dict[str, Curve] = {
            k: _map_curve_or_id_from_solver_(v, solver) for k, v in curve.items()
        }
        return mapped_dict
    else:
        return _map_curve_or_id_from_solver_(curve, solver)


def _map_curve_from_solver(curve: CurveInput, solver: Solver) -> CurveOption:
    """
    Maps a "Curve | str | dict[str, Curve | str] | NoInput" to a
    "Curve | dict[str, Curve] | NoInput" via a Solver.

    This is the inexplicit variety which handles NoInput.
    """
    if isinstance(curve, NoInput) or curve is None:
        return NoInput(0)
    else:
        return _map_curve_from_solver_(curve, solver)


def _validate_curve_not_str(curve: CurveOrId_) -> Curve:
    if isinstance(curve, str):
        raise ValueError("`curves` must contain Curve, not str, if `solver` not given.")
    return curve


def _validate_no_str_in_curve_input(curve: CurveInput) -> CurveOption:
    """
    If a Solver is not available then raise an Exception if a CurveInput contains string Id.
    """
    if isinstance(curve, dict):
        return {k: _validate_curve_not_str(v) for k, v in curve.items()}
    elif isinstance(curve, NoInput) or curve is None:
        return NoInput(0)
    else:
        return _validate_curve_not_str(curve)
