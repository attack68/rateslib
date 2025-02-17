from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from rateslib import defaults
from rateslib.curves import Curve
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.default import NoInput, _drb
from rateslib.dual.utils import _dual_float
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Curve,
        CurveOption_,
        DataFrame,
        DualTypes,
        FXVolOption_,
        datetime,
        datetime_,
        str_,
    )


def _validate_float_args(
    fixing_method: str | NoInput,
    method_param: int | NoInput,
    spread_compound_method: str | NoInput,
) -> tuple[str, int, str]:
    """
    Validate the argument input to float periods.

    Returns
    -------
    tuple
    """
    fixing_method_: str = _drb(defaults.fixing_method, fixing_method).lower()
    if fixing_method_ not in [
        "ibor",
        "rfr_payment_delay",
        "rfr_observation_shift",
        "rfr_lockout",
        "rfr_lookback",
        "rfr_payment_delay_avg",
        "rfr_observation_shift_avg",
        "rfr_lockout_avg",
        "rfr_lookback_avg",
    ]:
        raise ValueError(
            "`fixing_method` must be in {'rfr_payment_delay', "
            "'rfr_observation_shift', 'rfr_lockout', 'rfr_lookback', 'ibor'}, "
            f"got '{fixing_method_}'.",
        )

    method_param_: int = _drb(defaults.fixing_method_param[fixing_method_], method_param)
    if method_param_ != 0 and fixing_method_ == "rfr_payment_delay":
        raise ValueError(
            "`method_param` should not be used (or a value other than 0) when "
            f"using a `fixing_method` of 'rfr_payment_delay', got {method_param_}. "
            f"Configure the `payment_lag` option instead to have the "
            f"appropriate effect.",
        )
    elif fixing_method_ == "rfr_lockout" and method_param_ < 1:
        raise ValueError(
            f'`method_param` must be >0 for "rfr_lockout" `fixing_method`, got {method_param_}',
        )

    spread_compound_method_: str = _drb(
        defaults.spread_compound_method, spread_compound_method
    ).lower()
    if spread_compound_method_ not in [
        "none_simple",
        "isda_compounding",
        "isda_flat_compounding",
    ]:
        raise ValueError(
            "`spread_compound_method` must be in {'none_simple', "
            "'isda_compounding', 'isda_flat_compounding'}, "
            f"got {spread_compound_method_}",
        )
    return fixing_method_, method_param_, spread_compound_method_


def _get_fx_and_base(
    currency: str,
    fx: FX_ = NoInput(0),
    base: str_ = NoInput(0),
) -> tuple[DualTypes, str_]:
    """
    From a local currency and potentially FX Objects determine the conversion rate between
    `currency` and `base`. If `base` is not given it is inferred from the FX Objects.
    """
    # TODO these can be removed when no traces of None remain.
    if fx is None:
        raise NotImplementedError("TraceBack for NoInput")  # pragma: no cover
    if base is None:
        raise NotImplementedError("TraceBack for NoInput")  # pragma: no cover

    if isinstance(fx, FXRates | FXForwards):
        base_: str | NoInput = fx.base if isinstance(base, NoInput) else base.lower()
        if base_ == currency:
            fx_: DualTypes = 1.0
        else:
            fx_ = fx.rate(pair=f"{currency}{base_}")
    elif not isinstance(base, NoInput):  # and fx is then a float or None
        base_ = base
        if isinstance(fx, NoInput):
            if base.lower() != currency.lower():
                raise ValueError(
                    f"`base` ({base}) cannot be requested without supplying `fx` as a "
                    "valid FXRates or FXForwards object to convert to "
                    f"currency ({currency}).\n"
                    "If you are using a `Solver` with multi-currency instruments have you "
                    "forgotten to attach the FXForwards in the solver's `fx` argument?",
                )
            fx_ = 1.0
        else:
            if abs(fx - 1.0) < 1e-10:
                pass  # no warning when fx == 1.0
            else:
                warnings.warn(
                    f"`base` ({base}) should not be given when supplying `fx` as numeric "
                    f"since it will not be used.\nIt may also be interpreted as giving "
                    f"wrong results.\nBest practice is to instead supply `fx` as an "
                    f"FXRates (or FXForwards) object.\n"
                    f"Reformulate: [fx={fx}, base='{base}'] -> "
                    f"[fx=FXRates({{'{currency}{base}': {fx}}}), base='{base}'].",
                    UserWarning,
                )
            fx_ = fx
    else:  # base is None and fx is float or None.
        base_ = NoInput(0)
        if isinstance(fx, NoInput):
            fx_ = 1.0
        else:
            if abs(fx - 1.0) < 1e-12:
                pass  # no warning when fx == 1.0
            else:
                warnings.warn(
                    "It is not best practice to provide `fx` as numeric since this can "
                    "cause errors of output when dealing with multi-currency derivatives,\n"
                    "and it also fails to preserve FX rate sensitivity in calculations.\n"
                    "Instead, supply a 'base' currency and use an "
                    "FXRates or FXForwards object.\n"
                    f"Reformulate: [fx={fx}, base=None] -> "
                    f"[fx=FXRates({{'{currency}bas': {fx}}}), base='bas'].",
                    UserWarning,
                )
            fx_ = fx

    return fx_, base_


def _maybe_local(
    value: DualTypes,
    local: bool,
    currency: str,
    fx: FX_,
    base: str | NoInput,
) -> dict[str, DualTypes] | DualTypes:
    """
    Return NPVs in scalar form or dict form.
    """
    if local:
        return {currency: value}
    else:
        return _maybe_fx_converted(value, currency, fx, base)


def _maybe_fx_converted(
    value: DualTypes,
    currency: str,
    fx: FX_,
    base: str | NoInput,
) -> DualTypes:
    fx_, _ = _get_fx_and_base(currency, fx, base)
    return value * fx_


def _float_or_none(val: DualTypes | None) -> float | None:
    if val is None:
        return None
    else:
        return _dual_float(val)


def _get_ibor_curve_from_dict(months: int, d: dict[str, Curve]) -> Curve:
    try:
        return d[f"{months}m"]
    except KeyError:
        try:
            return d[f"{months}M"]
        except KeyError:
            raise ValueError(
                "If supplying `curve` as dict must provide a tenor mapping key and curve for"
                f"the frequency of the given Period. The missing mapping is '{months}m'."
            )


def _maybe_get_rfr_curve_from_dict(curve: Curve | dict[str, Curve] | NoInput) -> Curve | NoInput:
    if isinstance(curve, dict):
        return _get_rfr_curve_from_dict(curve)
    else:
        return curve


def _get_rfr_curve_from_dict(d: dict[str, Curve]) -> Curve:
    for s in ["rfr", "RFR", "Rfr"]:
        try:
            ret: Curve = d[s]
        except KeyError:
            continue
        else:
            return ret
    raise ValueError(
        "A `curve` supplied as dict to an RFR based period must contain a key entry 'rfr'."
    )


def _trim_df_by_index(df: DataFrame, left: datetime_, right: datetime_) -> DataFrame:
    """
    Used by fixings_tables to constrict the view to a left and right bound
    """
    if len(df.index) == 0 or (isinstance(left, NoInput) and isinstance(right, NoInput)):
        return df
    elif isinstance(left, NoInput):
        return df[:right]  # type: ignore[misc]
    elif isinstance(right, NoInput):
        return df[left:]  # type: ignore[misc]
    else:
        return df[left:right]  # type: ignore[misc]


def _get_vol_smile_or_value(vol: FXVolOption_, expiry: datetime) -> FXDeltaVolSmile | DualTypes:
    if isinstance(vol, FXDeltaVolSurface):
        return vol.get_smile(expiry)
    else:
        return _validate_obj_not_no_input(vol, "vol")  # type: ignore[return-value]


def _get_vol_smile_or_raise(vol: FXVolOption_, expiry: datetime) -> FXDeltaVolSmile:
    if isinstance(vol, FXDeltaVolSurface):
        return vol.get_smile(expiry)
    elif isinstance(vol, FXDeltaVolSmile):
        return vol
    else:
        raise ValueError("Must supply FXDeltaVolSmile/Surface as `vol` not numeric value.")


def _get_vol_delta_type(vol: FXVolOption_, default_delta_type: str) -> str:
    if not isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
        return default_delta_type
    else:
        return vol.delta_type


def _validate_credit_curves(curve: CurveOption_, disc_curve: CurveOption_) -> tuple[Curve, Curve]:
    # used by Credit type Periods to narrow inputs
    if not isinstance(curve, Curve):
        raise TypeError(
            "`curves` have not been supplied correctly.\n"
            "`curve`for a CreditPremiumPeriod must be supplied as a Curve type."
        )
    if not isinstance(disc_curve, Curve):
        raise TypeError(
            "`curves` have not been supplied correctly.\n"
            "`disc_curve` for a CreditPremiumPeriod must be supplied as a Curve type."
        )
    return curve, disc_curve


def _validate_fx_as_forwards(fx: FX_) -> FXForwards:
    if isinstance(fx, NoInput):
        raise ValueError(
            "An FXForwards object for `fx` is required for FXOption pricing.\n"
            "If this instrument is part of a Solver, have you omitted the `fx` input?",
        )
    elif not isinstance(fx, FXForwards):
        raise ValueError(
            "An FXForwards object for `fx` is required for FXOption prcing.\n"
            f"The given type, '{type(fx).__name__}', cannot be used here."
        )
    else:
        return fx


# def _validate_broad_delta_bounds(phi, delta, delta_type):
#     if phi < 0 and "_pa" in delta_type:
#         assert delta <= 0.0
#     elif phi < 0:
#         assert -1.0 <= delta <= 0.0
#     elif
