from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol

from rateslib import defaults
from rateslib.curves import MultiCsaCurve, ProxyCurve
from rateslib.enums.generics import NoInput, _drb

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        Curves_,
        FXVolOption_,
        Solver,
        Solver_,
        Vol_,
        _BaseCurve,
        _BaseCurveOrDict,
        _BaseCurveOrDict_,
        _BaseCurveOrId,
        _BaseCurveOrIdOrIdDict,
        _BaseCurveOrIdOrIdDict_,
    )


class _WithPricingObjs(Protocol):
    """
    Protocol to determine individual *curves* and *vol* inputs for each *Instrument*,
    possibly deriving those from a :class:`~rateslib.solver.Solver` mapping.
    """

    def _parse_curves(self, curves: Curves_) -> _Curves:
        """Method is needed to map the `curves` argument input for any individual *Instrument* into
        the more defined :class:`~rateslib.curves._parsers._Curves` structure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_parse_curves` of class `_WithPricingObjs`."
        )

    def _parse_vol(self, vol: Vol_) -> _Vol:
        """Method is needed to map the `vol` argument input for any individual *Instrument* into
        the more defined :class:`~rateslib.curves._parsers._Vol` structure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_parse_vol` of class `_WithPricingObjs`."
        )


class _Curves:
    """
    Container for a pricing object providing a mapping for curves.
    """

    def __init__(
        self,
        *,
        rate_curve: _BaseCurveOrIdOrIdDict_ = NoInput(0),
        disc_curve: _BaseCurveOrIdOrIdDict_ = NoInput(0),
        index_curve: _BaseCurveOrIdOrIdDict_ = NoInput(0),
        leg2_rate_curve: _BaseCurveOrIdOrIdDict_ = NoInput(0),
        leg2_disc_curve: _BaseCurveOrIdOrIdDict_ = NoInput(0),
        leg2_index_curve: _BaseCurveOrIdOrIdDict_ = NoInput(0),
    ):
        self._rate_curve = rate_curve
        self._disc_curve = disc_curve
        self._index_curve = index_curve
        self._leg2_rate_curve = leg2_rate_curve
        self._leg2_disc_curve = leg2_disc_curve
        self._leg2_index_curve = leg2_index_curve

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _Curves):
            return False
        else:
            bools = [
                self.disc_curve == other.disc_curve,
                self.index_curve == other.index_curve,
                self.rate_curve == other.rate_curve,
                self.leg2_rate_curve == other.leg2_rate_curve,
                self.leg2_disc_curve == other.leg2_disc_curve,
                self.leg2_index_curve == other.leg2_index_curve,
            ]
            return all(bools)

    @property
    def rate_curve(self) -> _BaseCurveOrIdOrIdDict_:
        """The curve used for floating rate or hazard rate forecasting on leg1."""
        return self._rate_curve

    @property
    def disc_curve(self) -> _BaseCurveOrIdOrIdDict_:
        """The curve used for discounting on leg1."""
        return self._disc_curve

    @property
    def index_curve(self) -> _BaseCurveOrIdOrIdDict_:
        """The index curve used for forecasting index values on leg1."""
        return self._index_curve

    @property
    def leg2_rate_curve(self) -> _BaseCurveOrIdOrIdDict_:
        """The curve used for floating rate or hazard rate forecasting on leg2."""
        return self._leg2_rate_curve

    @property
    def leg2_disc_curve(self) -> _BaseCurveOrIdOrIdDict_:
        """The curve used for discounting on leg2."""
        return self._leg2_disc_curve

    @property
    def leg2_index_curve(self) -> _BaseCurveOrIdOrIdDict_:
        """The index curve used for forecasting index values on leg2."""
        return self._leg2_index_curve


class _Vol:
    """
    Container for a pricing object providing a mapping for volatility.
    """

    def __init__(
        self,
        *,
        fx_vol: FXVolOption_ = NoInput(0),
    ):
        self._fx_vol = fx_vol

    @property
    def fx_vol(self) -> FXVolOption_:
        """The FX vol object used for modelling FX volatility."""
        return self._fx_vol


def _get_maybe_curve_maybe_from_solver(
    curves_meta: _Curves,
    curves: _Curves,
    name: str,
    solver: Solver_,
) -> _BaseCurveOrDict_:
    curve: _BaseCurveOrIdOrIdDict_ = _drb(getattr(curves_meta, name), getattr(curves, name))
    if isinstance(curve, NoInput):
        return curve
    elif isinstance(solver, NoInput):
        return _validate_curve_is_not_id(curve=curve)
    else:
        return _get_curve_from_solver(
            curve=curve,
            solver=solver,
        )
        # try:
        #     parsed_curve = _parse_curve_or_id_from_solver_(curve=curve, solver=solver)
        #     return parsed_curve
        # except KeyError as e:
        #     raise ValueError(
        #         "`curves` must contain str curve `id` s existing in `solver` "
        #         "(or its associated `pre_solvers`).\n"
        #         f"The sought id was: '{e.args[0]}'.\n"
        #         f"The available ids are {list(solver.pre_curves.keys())}.",
        #     )


# def _get_curve_maybe_from_solver(
#     curves_meta: _Curves,
#     curves: _Curves,
#     name: str,
#     solver: Solver_,
# ) -> _BaseCurveOrDict:
#     curve: _BaseCurveOrIdOrIdDict = _drb(getattr(curves_meta, name), getattr(curves, name))
#     if isinstance(solver, NoInput):
#         return _validate_curve_is_not_id(curve=curve)
#     else:
#         try:
#             parsed_curve = _parse_curve_or_id_from_solver_(curve=curve, solver=solver)
#             return parsed_curve
#         except KeyError as e:
#             raise ValueError(
#                 "`curves` must contain str curve `id` s existing in `solver` "
#                 "(or its associated `pre_solvers`).\n"
#                 f"The sought id was: '{e.args[0]}'.\n"
#                 f"The available ids are {list(solver.pre_curves.keys())}.",
#             )


def _get_maybe_curve_from_solver(
    curve: _BaseCurveOrIdOrIdDict_, solver: Solver
) -> _BaseCurveOrDict_:
    """
    Maps a "Curve | str | dict[str, Curve | str] | NoInput" to a
    "Curve | dict[str, Curve] | NoInput" via a Solver.

    This is the inexplicit variety which handles NoInput.
    """
    if isinstance(curve, NoInput) or curve is None:
        return NoInput(0)
    else:
        return _get_curve_from_solver_(curve=curve, solver=solver)


def _get_curve_from_solver(curve: _BaseCurveOrIdOrIdDict, solver: Solver) -> _BaseCurveOrDict:
    """
    Maps a "Curve | str | dict[str, Curve | str]" to a "Curve | dict[str, Curve]" via a Solver.

    If curve input involves strings get objects directly from solver curves mapping.

    This is the explicit variety which does not handle NoInput.
    """
    if isinstance(curve, dict):
        parsed_dict: dict[str, _BaseCurve] = {
            k: _parse_curve_or_id_from_solver_(curve=v, solver=solver) for k, v in curve.items()
        }
        return parsed_dict
    else:
        return _parse_curve_or_id_from_solver_(curve, solver)


def _parse_curve_or_id_from_solver_(curve: _BaseCurveOrId, solver: Solver) -> _BaseCurve:
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


def _validate_curve_is_not_id(curve: _BaseCurveOrIdOrIdDict) -> _BaseCurveOrDict_:
    """
    Validate that a curve input is an object and not a string id.
    """
    if isinstance(curve, dict):
        return {k: _validate_base_curve_is_not_id(v) for k, v in curve.items()}
    elif isinstance(curve, NoInput) or curve is None:
        return NoInput(0)
    else:
        return _validate_base_curve_is_not_id(curve)


def _validate_base_curve_is_not_id(curve: _BaseCurveOrId) -> _BaseCurve:
    if isinstance(curve, str):  # curve is a str ID
        raise ValueError(
            f"`curves` must contain _BaseCurve, not str, if `solver` not given. Got id: '{curve}'"
        )
    return curve


def _get_fx_maybe_from_solver(solver: Solver_, fx: FX_) -> FX_:
    if isinstance(fx, NoInput):
        if isinstance(solver, NoInput):
            fx_: FX_ = NoInput(0)
        else:
            if isinstance(solver.fx, NoInput):
                fx_ = NoInput(0)
            else:
                fx_ = solver._get_fx()
    else:
        fx_ = fx
        if (
            not isinstance(solver, NoInput)
            and not isinstance(solver.fx, NoInput)
            and id(fx) != id(solver.fx)
        ):
            warnings.warn(
                "Solver contains an `fx` attribute but an `fx` argument has been "
                "supplied which will be used but is not the same. This can lead "
                "to calculation inconsistencies, mathematically.",
                UserWarning,
            )

    return fx_
