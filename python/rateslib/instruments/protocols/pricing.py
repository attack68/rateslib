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

import warnings
from typing import TYPE_CHECKING, Literal, Protocol, overload

from rateslib import defaults
from rateslib.curves import MultiCsaCurve, ProxyCurve
from rateslib.dual import Dual, Dual2, Variable
from rateslib.enums.generics import NoInput, _drb
from rateslib.volatility import _BaseIRCube, _BaseIRSmile

if TYPE_CHECKING:
    from rateslib.local_types import (  # pragma: no cover
        FX_,
        Any,
        CurvesT_,
        DualTypes,
        FXForwards_,
        FXVol_,
        IRVol_,
        NoInput,
        Solver,
        Solver_,
        VolStrat_,
        VolT_,
        _BaseCurve,
        _BaseCurve_,
        _BaseCurveOrDict,
        _BaseCurveOrDict_,
        _BaseCurveOrId,
        _BaseCurveOrIdOrIdDict,
        _BaseCurveOrIdOrIdDict_,
        _BaseInstrument,
        _FXVolObj,
        _IRVolObj,
        _IRVolOption_,
    )


class _WithPricingObjs(Protocol):
    """
    Protocol to determine individual *curves* and *vol* inputs for each *Instrument*.

    This protocol contains two internal methods for parsing ``curves`` and ``vol`` inputs
    according to individual *Instruments* for pricing methods, such as
    :meth:`~rateslib.instruments.protocols._WithNpv.npv` and
    :meth:`~rateslib.instruments.protocols._WithRate.rate`.
    """

    def _parse_curves(self, curves: CurvesT_) -> _Curves:
        """Method is needed to map the `curves` argument input for any individual *Instrument* into
        the more defined :class:`~rateslib.curves._parsers._Curves` structure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_parse_curves` of class `_WithPricingObjs`."
        )

    def _parse_vol(self, vol: VolT_) -> _Vol:
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
        fx_vol: FXVol_ = NoInput(0),
        ir_vol: IRVol_ = NoInput(0),
    ):
        self._fx_vol = fx_vol
        self._ir_vol = ir_vol

    @property
    def fx_vol(self) -> FXVol_:
        """The FX vol object used for modelling FX volatility."""
        return self._fx_vol

    @property
    def ir_vol(self) -> IRVol_:
        """The IR vol object used for modelling IR volatility."""
        return self._ir_vol

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _Vol):
            return False
        else:
            return self.fx_vol == other.fx_vol and self.ir_vol == other.ir_vol


def _parse_curves(
    obj: _BaseInstrument, curves: CurvesT_, solver: Solver_
) -> tuple[_Curves, _Curves, Solver_]:
    return (obj._parse_curves(curves), obj.kwargs.meta["curves"], solver)


@overload
def _parse_vol(
    obj: _BaseInstrument,
    vol: VolT_,
    solver: Solver_,
    sequence: Literal[False],
) -> tuple[_Vol, _Vol, Solver_]: ...


@overload
def _parse_vol(
    obj: _BaseInstrument,
    vol: VolStrat_,
    solver: Solver_,
    sequence: Literal[True],
) -> tuple[VolStrat_, VolStrat_, Solver_]: ...


def _parse_vol(
    obj: _BaseInstrument,
    vol: VolT_ | VolStrat_,
    solver: Solver_,
    sequence: bool,
) -> tuple[_Vol | VolStrat_, _Vol | VolStrat_, Solver_]:
    return obj._parse_vol(vol), obj.kwargs.meta["vol"], solver  # type: ignore[arg-type]


# Solver and Curve mapping


@overload
def _get_curve(
    name: str,
    allow_dict: Literal[False],
    allow_no_input: Literal[True],
    curves: _Curves,
    curves_meta: _Curves,
    solver: Solver_,
) -> _BaseCurve_: ...


@overload
def _get_curve(
    name: str,
    allow_dict: Literal[False],
    allow_no_input: Literal[False],
    curves: _Curves,
    curves_meta: _Curves,
    solver: Solver_,
) -> _BaseCurve: ...


@overload
def _get_curve(
    name: str,
    allow_dict: Literal[True],
    allow_no_input: Literal[True],
    curves: _Curves,
    curves_meta: _Curves,
    solver: Solver_,
) -> _BaseCurveOrDict_: ...


@overload
def _get_curve(
    name: str,
    allow_dict: Literal[True],
    allow_no_input: Literal[False],
    curves: _Curves,
    curves_meta: _Curves,
    solver: Solver_,
) -> _BaseCurveOrDict: ...


def _get_curve(
    name: str,
    allow_dict: bool,
    allow_no_input: bool,
    curves: _Curves,
    curves_meta: _Curves,
    solver: Solver_,
) -> _BaseCurveOrDict_:
    curve: _BaseCurveOrIdOrIdDict_ = _drb(getattr(curves_meta, name), getattr(curves, name))
    if isinstance(curve, NoInput) or curve is None:
        if allow_no_input:
            return NoInput(0)
        else:
            raise ValueError(f"`{name}` must be provided. Got NoInput.")
    elif isinstance(solver, NoInput):
        return _validate_base_curve_or_dict(  # type: ignore[no-any-return, call-overload]
            curve=curve, allow_dict=allow_dict, allow_no_input=allow_no_input
        )
    else:
        return _get_curve_from_solver(  # type: ignore[no-any-return, call-overload]
            curve=curve,
            solver=solver,
            allow_dict=allow_dict,
        )


@overload
def _validate_base_curve_or_dict(
    curve: _BaseCurveOrIdOrIdDict,
    allow_dict: Literal[True],
    allow_no_input: Literal[True],
) -> _BaseCurveOrDict_: ...


@overload
def _validate_base_curve_or_dict(
    curve: _BaseCurveOrIdOrIdDict,
    allow_dict: Literal[True],
    allow_no_input: Literal[False],
) -> _BaseCurveOrDict: ...


@overload
def _validate_base_curve_or_dict(
    curve: _BaseCurveOrIdOrIdDict,
    allow_dict: Literal[False],
    allow_no_input: Literal[True],
) -> _BaseCurve_: ...


@overload
def _validate_base_curve_or_dict(
    curve: _BaseCurveOrIdOrIdDict,
    allow_dict: Literal[False],
    allow_no_input: Literal[False],
) -> _BaseCurve: ...


def _validate_base_curve_or_dict(
    curve: _BaseCurveOrIdOrIdDict,
    allow_dict: bool,
    allow_no_input: bool,
) -> _BaseCurveOrDict_:
    """
    Validate that a curve input is an object and not a string id.
    """
    if isinstance(curve, dict):
        if not allow_dict:
            raise ValueError("Cannot supply a dict type object as this `curve`.")
        else:
            return {
                k: _validate_base_curve(v, allow_no_input=allow_no_input)  # type: ignore[call-overload]
                for k, v in curve.items()
            }
    else:
        return _validate_base_curve(curve, allow_no_input=allow_no_input)  # type: ignore[no-any-return, call-overload]


@overload
def _validate_base_curve(curve: _BaseCurveOrId, allow_no_input: Literal[False]) -> _BaseCurve: ...


@overload
def _validate_base_curve(curve: _BaseCurveOrId, allow_no_input: Literal[True]) -> _BaseCurve_: ...


def _validate_base_curve(curve: _BaseCurveOrId, allow_no_input: bool) -> _BaseCurve_:
    if isinstance(curve, str):
        if allow_no_input:
            return NoInput(0)
        else:
            raise ValueError(
                f"`curves` must contain _BaseCurve, not str, if `solver` not given. "
                f"Got id: '{curve}'"
            )
    return curve


@overload
def _get_curve_from_solver(
    curve: _BaseCurveOrIdOrIdDict, solver: Solver, allow_dict: Literal[True]
) -> _BaseCurveOrDict: ...


@overload
def _get_curve_from_solver(
    curve: _BaseCurveOrIdOrIdDict, solver: Solver, allow_dict: Literal[False]
) -> _BaseCurve: ...


def _get_curve_from_solver(
    curve: _BaseCurveOrIdOrIdDict, solver: Solver, allow_dict: bool
) -> _BaseCurveOrDict:
    """
    Maps a "Curve | str | dict[str, Curve | str]" to a "Curve | dict[str, Curve]" via a Solver.

    If curve input involves strings get objects directly from solver curves mapping.

    This is the explicit variety which does not handle NoInput.
    """
    if isinstance(curve, dict):
        if not allow_dict:
            raise ValueError("Cannot supply a dict type object as this `curve`.")
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
        return curve  # type: ignore[no-any-return]  # mypy error
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


# Solver and FX Vol mapping


@overload
def _get_fx_vol(
    allow_numeric: Literal[True],
    allow_no_input: Literal[True],
    vol: _Vol,
    vol_meta: _Vol,
    solver: Solver_,
) -> _FXVolObj | DualTypes | NoInput: ...


@overload
def _get_fx_vol(
    allow_numeric: Literal[True],
    allow_no_input: Literal[False],
    vol: _Vol,
    vol_meta: _Vol,
    solver: Solver_,
) -> _FXVolObj | DualTypes: ...


@overload
def _get_fx_vol(
    allow_numeric: Literal[False],
    allow_no_input: Literal[True],
    vol: _Vol,
    vol_meta: _Vol,
    solver: Solver_,
) -> _FXVolObj | NoInput: ...


@overload
def _get_fx_vol(
    allow_numeric: Literal[False],
    allow_no_input: Literal[False],
    vol: _Vol,
    vol_meta: _Vol,
    solver: Solver_,
) -> _FXVolObj: ...


def _get_fx_vol(
    allow_numeric: bool,
    allow_no_input: bool,
    vol: _Vol,
    vol_meta: _Vol,
    solver: Solver_,
) -> _FXVolObj | DualTypes | NoInput:
    fx_vol_ = _drb(vol_meta.fx_vol, vol.fx_vol)
    if isinstance(fx_vol_, NoInput) or fx_vol_ is None:
        if allow_no_input:
            return NoInput(0)
        else:
            raise ValueError("`fx_vol` must be provided. Got NoInput.")
    elif isinstance(fx_vol_, float | Dual | Dual2 | Variable):
        if allow_numeric:
            return fx_vol_
        else:
            raise ValueError("`fx_vol` must be an object. Got numeric quantity.")
    elif isinstance(solver, NoInput):
        return _validate_base_fx_vol(fx_vol=fx_vol_, allow_no_input=allow_no_input)  # type: ignore[no-any-return, call-overload]
    else:
        return _get_fx_vol_from_solver(fx_vol=fx_vol_, solver=solver)


@overload
def _validate_base_fx_vol(fx_vol: _FXVolObj | str, allow_no_input: Literal[False]) -> _FXVolObj: ...


@overload
def _validate_base_fx_vol(
    fx_vol: _FXVolObj | str, allow_no_input: Literal[True]
) -> _FXVolObj | NoInput: ...


def _validate_base_fx_vol(fx_vol: _FXVolObj | str, allow_no_input: bool) -> _FXVolObj | NoInput:
    if isinstance(fx_vol, str):
        if allow_no_input:
            return NoInput(0)
        else:
            raise ValueError(
                f"`fx_vol` must contain FXVol object, not str, if `solver` not given. "
                f"Got id: '{fx_vol}'"
            )
    return fx_vol


def _get_fx_vol_from_solver(fx_vol: _FXVolObj | str, solver: Solver) -> _FXVolObj:
    if isinstance(fx_vol, str):
        return solver._get_pre_fxvol(fx_vol)

    try:
        # it is a safeguard to load curves from solvers when a solver is
        # provided and multiple curves might have the same id
        __: _FXVolObj = solver._get_pre_fxvol(fx_vol.id)
        if id(__) != id(fx_vol):  # Python id() is a memory id, not a string label id.
            raise ValueError(
                "An FXVol object has been supplied, as part of ``vol``, which has the same "
                f"`id` ('{fx_vol.id}'),\nas one of the curves available as part of the "
                "Solver's collection but is not the same object.\n"
                "This is ambiguous and cannot price.\n"
                "Either refactor the arguments as follows:\n"
                "1) remove the conflicting object: [vol=[..], solver=<Solver>] -> "
                "[vol=None, solver=<Solver>]\n"
                "2) change the `id` of the supplied FXVol object and ensure the rateslib.defaults "
                "option 'curve_not_in_solver' is set to 'ignore'.\n"
                "   This will remove the ability to accurately price risk metrics.",
            )
        return __
    except AttributeError:
        raise AttributeError(
            "FXVol object has no attribute `id`, likely it is not a valid object, got: "
            f"{fx_vol}.\nSince a solver is provided have you missed labelling the `curves` "
            f"of the instrument or supplying `curves` directly?",
        )
    except KeyError:
        if defaults.curve_not_in_solver == "ignore":
            return fx_vol
        elif defaults.curve_not_in_solver == "warn":
            warnings.warn("FXVol object not found in `solver`.", UserWarning)
            return fx_vol
        else:
            raise ValueError("FXVol object must be in `solver`.")


# Solver and IR Vol mapping


def _maybe_get_ir_vol_maybe_from_solver(
    vol_meta: _Vol,
    vol: _Vol,
    # name: str, = "fx_vol"
    solver: Solver_,
) -> _IRVolOption_:
    ir_vol_ = _drb(vol_meta.ir_vol, vol.ir_vol)
    if isinstance(ir_vol_, NoInput | float | Dual | Dual2 | Variable):
        return ir_vol_
    elif isinstance(solver, NoInput):
        return _validate_ir_vol_is_not_id(ir_vol=ir_vol_)
    else:
        return _get_ir_vol_from_solver(ir_vol=ir_vol_, solver=solver)


def _get_ir_vol_from_solver(
    ir_vol: _BaseIRSmile | _BaseIRCube[Any] | str, solver: Solver
) -> _BaseIRSmile | _BaseIRCube[Any]:
    if isinstance(ir_vol, str):
        return solver._get_pre_irvol(ir_vol)

    try:
        # it is a safeguard to load curves from solvers when a solver is
        # provided and multiple curves might have the same id
        __: _IRVolObj = solver._get_pre_irvol(ir_vol.id)
        if id(__) != id(ir_vol):  # Python id() is a memory id, not a string label id.
            raise ValueError(
                "An IRVol object has been supplied, as part of ``vol``, which has the same "
                f"`id` ('{ir_vol.id}'),\nas one of the curves available as part of the "
                "Solver's collection but is not the same object.\n"
                "This is ambiguous and cannot price.\n"
                "Either refactor the arguments as follows:\n"
                "1) remove the conflicting object: [vol=[..], solver=<Solver>] -> "
                "[vol=None, solver=<Solver>]\n"
                "2) change the `id` of the supplied IRVol object and ensure the rateslib.defaults "
                "option 'curve_not_in_solver' is set to 'ignore'.\n"
                "   This will remove the ability to accurately price risk metrics.",
            )
        return __
    except AttributeError:
        raise AttributeError(
            "IRVol object has no attribute `id`, likely it is not a valid object, got: "
            f"{ir_vol}.\nSince a solver is provided have you missed labelling the `curves` "
            f"of the instrument or supplying `curves` directly?",
        )
    except KeyError:
        if defaults.curve_not_in_solver == "ignore":
            return ir_vol
        elif defaults.curve_not_in_solver == "warn":
            warnings.warn("FXVol object not found in `solver`.", UserWarning)
            return ir_vol
        else:
            raise ValueError("FXVol object must be in `solver`.")


def _validate_ir_vol_is_not_id(ir_vol: _IRVolObj | str) -> _IRVolObj:
    if isinstance(ir_vol, str):  # curve is a str ID
        raise ValueError(
            f"`vol` must contain IRVol object, not str, if `solver` not given. Got id: '{ir_vol}'"
        )
    return ir_vol


# FX and Solver mapping


def _get_fx_forwards_maybe_from_solver(solver: Solver_, fx: FXForwards_) -> FXForwards_:
    if isinstance(fx, NoInput):
        if isinstance(solver, NoInput):
            fx_: FXForwards_ = NoInput(0)
        else:
            if isinstance(solver.fx, NoInput):
                fx_ = NoInput(0)
            else:
                # TODO disallow `fx` on Solver as FXRates. Only allow FXForwards.
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


def _get_fx_maybe_from_solver(solver: Solver_, fx: FXForwards_) -> FXForwards_:
    if isinstance(fx, NoInput):
        if isinstance(solver, NoInput):
            fx_: FX_ = NoInput(0)
        else:
            if isinstance(solver.fx, NoInput):
                fx_ = NoInput(0)
            else:
                fx_ = solver._get_fx()  # will validate the state
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

    return fx_  # type: ignore[return-value]
