from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pandas import DataFrame

from rateslib import FXDeltaVolSmile, FXDeltaVolSurface, defaults
from rateslib.curves._parsers import _get_curves_maybe_from_solver, _validate_curve_not_no_input
from rateslib.default import NoInput
from rateslib.dual import Dual, Dual2, Variable
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXSabrSmile, FXSabrSurface

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Curves_,
        Curves_DiscTuple,
        FXVol,
        FXVol_,
        FXVolOption_,
        Solver_,
        str_,
    )


def _get_base_maybe_from_fx(fx: FX_, base: str_, local_ccy: str_) -> str_:
    if isinstance(fx, NoInput | float) and isinstance(base, NoInput):
        # base will not be inherited from a 2nd level inherited object, i.e.
        # from solver.fx, to preserve single currency instruments being defaulted
        # to their local currency.
        base_ = local_ccy
    elif isinstance(fx, FXRates | FXForwards) and isinstance(base, NoInput):
        base_ = fx.base
    else:
        base_ = base
    return base_


def _get_fx_maybe_from_solver(solver: Solver_, fx: FX_) -> FX_:
    if isinstance(fx, NoInput):
        if isinstance(solver, NoInput):
            fx_: FX_ = NoInput(0)
            # fx_ = 1.0
        else:  # solver is not NoInput:
            if isinstance(solver.fx, NoInput):
                fx_ = NoInput(0)
                # fx_ = 1.0
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


def _get_curves_fx_and_base_maybe_from_solver(
    curves_attr: Curves_,
    solver: Solver_,
    curves: Curves_,
    fx: FX_,
    base: str_,
    local_ccy: str_,
) -> tuple[Curves_DiscTuple, FX_, str_]:
    """
    Parses the ``solver``, ``curves``, ``fx`` and ``base`` arguments in combination.

    Parameters
    ----------
    curves_attr
        The curves attribute attached to the class.
    solver
        The solver argument passed in the outer method.
    curves
        The curves argument passed in the outer method.
    fx
        The fx argument agrument passed in the outer method.

    Returns
    -------
    tuple : (leg1 forecasting, leg1 discounting, leg2 forecasting, leg2 discounting), fx, base

    Notes
    -----
    If only one curve is given this is used as all four curves.

    If two curves are given the forecasting curve is used as the forecasting
    curve on both legs and the discounting curve is used as the discounting
    curve for both legs.

    If three curves are given the single discounting curve is used as the
    discounting curve for both legs.
    """
    # First process `base`.
    base_ = _get_base_maybe_from_fx(fx, base, local_ccy)
    # Second process `fx`
    fx_ = _get_fx_maybe_from_solver(solver, fx)
    # Third process `curves`
    curves_ = _get_curves_maybe_from_solver(curves_attr, solver, curves)
    return curves_, fx_, base_


def _get_fxvol_maybe_from_solver(vol_attr: FXVol_, vol: FXVol_, solver: Solver_) -> FXVolOption_:
    """
    Try to retrieve a general vol input from a solver or the default vol object associated with
    instrument.

    If the resolved input is a Sequence then return that directly. This aids with recursive
    calculation in FXOptionStrats.

    Parameters
    ----------
    vol_attr: DualTypes, str or FXDeltaVolSmile
        The vol attribute associated with the object at initialisation.
    vol: DualTypes, str of FXDeltaVolSMile
        The specific vol argument supplied at price time. Will take precendence.
    solver: Solver, optional
        A solver object

    Returns
    -------
    DualTypes, FXDeltaVolSmile or NoInput.blank
    """
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


def _get_fxvol_curves_fx_and_base_maybe_from_solver(
    curves_attr: Curves_,
    vol_attr: FXVol_,
    solver: Solver_,
    curves: Curves_,
    fx: FX_,
    base: str_,
    vol: FXVol_,
    local_ccy: str_,
) -> tuple[Curves_DiscTuple, FX_, str_, FXVolOption_]:
    """
    Parses the inputs including the instrument's attributes and also validates them
    """
    curves_, fx_, base_ = _get_curves_fx_and_base_maybe_from_solver(
        curves_attr,
        solver,
        curves,
        fx,
        base,
        local_ccy,
    )
    vol_ = _get_fxvol_maybe_from_solver(vol_attr, vol, solver)
    if isinstance(vol_, FXDeltaVolSmile | FXDeltaVolSurface):
        curves_1 = _validate_curve_not_no_input(curves_[1])
        if vol_.meta.eval_date != curves_1.nodes.initial:
            raise ValueError(
                "The `eval_date` on the FXDeltaVolSmile and the Curve do not align.\n"
                "Aborting calculation to avoid pricing errors.",
            )
    return curves_, fx_, base_, vol_


def _get(kwargs: dict[str, Any], leg: int = 1, filter: tuple[str, ...] = ()) -> dict[str, Any]:  # noqa: A002
    """
    A parser to return kwarg dicts for relevant legs.
    Internal structuring only.
    Will return kwargs relevant to leg1 OR leg2.
    Does not return keys that are specified in the filter.
    """
    if leg == 1:
        _: dict[str, Any] = {k: v for k, v in kwargs.items() if "leg2" not in k and k not in filter}
    else:
        _ = {k[5:]: v for k, v in kwargs.items() if "leg2_" in k and k not in filter}
    return _


def _push(spec: str_, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Push user specified kwargs to a default specification.
    Values from the `spec` dict will not overwrite specific user values already in `kwargs`.
    """
    if isinstance(spec, NoInput):
        return kwargs
    else:
        try:
            spec_kwargs = defaults.spec[spec.lower()]
        except KeyError:
            raise ValueError(f"Given `spec`, '{spec}', cannot be found in defaults.")

        user = {k: v for k, v in kwargs.items() if not isinstance(v, NoInput)}
        return {**kwargs, **spec_kwargs, **user}


def _update_not_noinput(base_kwargs: dict[str, Any], new_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Update the `base_kwargs` with `new_kwargs` (user values) unless those new values are NoInput.
    """
    updaters = {
        k: v for k, v in new_kwargs.items() if k not in base_kwargs or not isinstance(v, NoInput)
    }
    return {**base_kwargs, **updaters}


def _update_with_defaults(
    base_kwargs: dict[str, Any], default_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Update the `base_kwargs` with `default_kwargs` if the base_values are NoInput.blank.
    """
    updaters = {
        k: v
        for k, v in default_kwargs.items()
        if k in base_kwargs and base_kwargs[k] is NoInput.blank
    }
    return {**base_kwargs, **updaters}


def _inherit_or_negate(kwargs: dict[str, Any], ignore_blank: bool = False) -> dict[str, Any]:
    """Amend the values of leg2 kwargs if they are defaulted to inherit or negate from leg1."""

    def _replace(k: str, v: Any) -> Any:
        # either inherit or negate the value in leg2 from that in leg1
        if "leg2_" in k:
            if not isinstance(v, NoInput):
                return v  # do nothing if the attribute is an input

            try:
                leg1_v = kwargs[k[5:]]
            except KeyError:
                return v

            if leg1_v is NoInput.blank:
                if ignore_blank:
                    return v  # this allows an inheritor or negator to be called a second time
                else:
                    return NoInput(0)

            if v is NoInput(-1):
                return leg1_v * -1.0
            elif v is NoInput(1):
                return leg1_v
        return v  # do nothing to leg1 attributes

    return {k: _replace(k, v) for k, v in kwargs.items()}


def _lower(val: str_) -> str_:
    if isinstance(val, str):
        return val.lower()
    return val


def _upper(val: str_) -> str_:
    if isinstance(val, str):
        return val.upper()
    return val


def _composit_fixings_table(df_result: DataFrame, df: DataFrame) -> DataFrame:
    """
    Add a DataFrame to an existing fixings table by extending or adding to relevant columns.

    Parameters
    ----------
    df_result: The main DataFrame that will be updated
    df: The incoming DataFrame with new data to merge

    Returns
    -------
    DataFrame
    """
    # reindex the result DataFrame
    if df_result.empty:
        return df
    else:
        df_result = df_result.reindex(index=df_result.index.union(df.index))

    # update existing columns with missing data from the new available data
    for c in [c for c in df.columns if c in df_result.columns and c[1] in ["dcf", "rates"]]:
        df_result[c] = df_result[c].combine_first(df[c])

    # merge by addition existing values with missing filled to zero
    m = [c for c in df.columns if c in df_result.columns and c[1] in ["notional", "risk"]]
    if len(m) > 0:
        df_result[m] = df_result[m].add(df[m], fill_value=0.0)

    # append new columns without additional calculation
    a = [c for c in df.columns if c not in df_result.columns]
    if len(a) > 0:
        df_result[a] = df[a]

    # df_result.columns = MultiIndex.from_tuples(df_result.columns)
    return df_result
