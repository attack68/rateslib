from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from rateslib import defaults
from rateslib.curves import MultiCsaCurve, ProxyCurve
from rateslib.curves.utils import _CurveType
from rateslib.default import NoInput

if TYPE_CHECKING:
    from rateslib.typing import (
        Curve,
        Curve_,
        CurveInput,
        CurveInput_,
        CurveOption,
        CurveOption_,
        CurveOrId,
        Curves_,
        Curves_DiscTuple,
        Curves_Tuple,
        Solver,
    )


def _map_curve_or_id_from_solver_(curve: CurveOrId, solver: Solver) -> Curve:
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


def _map_curve_from_solver_(curve: CurveInput, solver: Solver) -> CurveOption:
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


def _map_curve_from_solver(curve: CurveInput_, solver: Solver) -> CurveOption_:
    """
    Maps a "Curve | str | dict[str, Curve | str] | NoInput" to a
    "Curve | dict[str, Curve] | NoInput" via a Solver.

    This is the inexplicit variety which handles NoInput.
    """
    if isinstance(curve, NoInput) or curve is None:
        return NoInput(0)
    else:
        return _map_curve_from_solver_(curve, solver)


def _validate_curve_not_str(curve: CurveOrId) -> Curve:
    if isinstance(curve, str):
        raise ValueError("`curves` must contain Curve, not str, if `solver` not given.")
    return curve


def _validate_no_str_in_curve_input(curve: CurveInput_) -> CurveOption_:
    """
    If a Solver is not available then raise an Exception if a CurveInput contains string Id.
    """
    if isinstance(curve, dict):
        return {k: _validate_curve_not_str(v) for k, v in curve.items()}
    elif isinstance(curve, NoInput) or curve is None:
        return NoInput(0)
    else:
        return _validate_curve_not_str(curve)


def _get_curves_maybe_from_solver(
    curves_attr: Curves_,
    solver: Solver | NoInput,
    curves: Curves_,
) -> Curves_DiscTuple:
    """
    Attempt to resolve curves as a variety of input types to a 4-tuple consisting of:
    (leg1 forecasting, leg1 discounting, leg2 forecasting, leg2 discounting)

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
    4-Tuple of Curve, dict[str, Curve], NoInput
    """
    if isinstance(curves, NoInput) and isinstance(curves_attr, NoInput):
        # no data is available so consistently return a 4-tuple of no data
        return (NoInput(0), NoInput(0), NoInput(0), NoInput(0))
    elif isinstance(curves, NoInput):
        # set the `curves` input as that which is set as attribute at instrument init.
        curves = curves_attr

    # refactor curves into a list
    if isinstance(curves, str) or not isinstance(curves, Sequence):  # Sequence can be str!
        # convert isolated value input to list
        curves_as_list: list[
            Curve | dict[str, str | Curve] | dict[str, str] | dict[str, Curve] | NoInput | str
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


def _validate_curve_is_not_dict(curve: CurveOption_) -> Curve_:
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


def _validate_curve_not_no_input(curve: Curve_) -> Curve:
    if isinstance(curve, NoInput):
        raise ValueError("`curve` must be supplied. Got NoInput or None.")
    return curve


T = TypeVar("T")


def _validate_obj_not_no_input(obj: T | NoInput, name: str) -> T:
    if isinstance(obj, NoInput):
        raise ValueError(f"`{name}` must be supplied. Got NoInput or None.")
    return obj


def _disc_maybe_from_curve(curve: CurveOption_, disc_curve: Curve_) -> Curve_:
    """Return a discount curve, pointed as the `curve` if not provided and if suitable Type."""
    if isinstance(disc_curve, NoInput):
        if isinstance(curve, dict):
            raise ValueError("`disc_curve` cannot be inferred from a dictionary of curves.")
        elif isinstance(curve, NoInput):
            return NoInput(0)
        elif curve._base_type == _CurveType.values:
            raise ValueError("`disc_curve` cannot be inferred from a non-DF based curve.")
        _: Curve | NoInput = curve
    else:
        _ = disc_curve
    return _


def _disc_required_maybe_from_curve(curve: CurveOption_, disc_curve: CurveOption_) -> Curve:
    """Return a discount curve, pointed as the `curve` if not provided and if suitable Type."""
    if isinstance(disc_curve, dict):
        raise NotImplementedError("`disc_curve` cannot currently be inferred from a dict.")
    _: Curve_ = _disc_maybe_from_curve(curve, disc_curve)
    if isinstance(_, NoInput):
        raise TypeError(
            "`curves` have not been supplied correctly. "
            "A `disc_curve` is required to perform function."
        )
    return _
