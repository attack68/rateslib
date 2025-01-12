
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.curves import ProxyCurve, MultiCsaCurve

if TYPE_CHECKING:
    from rateslib.typing import CurveInput_, CurveInput, Solver, CurveOption_, CurveOption, Curve


def _map_curve_from_solver_(curve: CurveInput_, solver: Solver) -> CurveOption_:
    """If curve input involves strings get objects directly from solver curves mapping.

    This is the explicit variety which does not handle NoInput.
    """
    if isinstance(curve, str):
        return solver._get_pre_curve(curve)
    elif isinstance(curve, dict):
        return {k: _map_curve_from_solver_(v, solver) for k, v in curve.items()}
    else:
        # look to return the curve directly but perform a validation against the
        # collection of the solver's curve dict
        if type(curve) is ProxyCurve or type(curve) is MultiCsaCurve:
            # TODO: (mid) consider also adding CompositeCurves as exceptions under the same rule
            # Proxy curves and MultiCsaCurves can exist outside of Solvers but be constructed
            # directly from an FXForwards object tied to a Solver using only a Solver's
            # dependent curves and AD variables.
            return curve

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


def _map_curve_from_solver(curve: CurveInput, solver: Solver) -> CurveOption:
    """If curve input involves strings get objects directly from solver curves mapping.

    This is the inexplicit variety which handles NoInput.
    """
    if isinstance(curve, NoInput):
        return NoInput(0)
    else:
        return _map_curve_from_solver_(curve, solver)
