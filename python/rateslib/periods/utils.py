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
from datetime import datetime
from typing import TYPE_CHECKING

import rateslib.errors as err
from rateslib.curves._parsers import _validate_obj_not_no_input
from rateslib.curves.curves import _BaseCurve
from rateslib.enums.generics import Err, NoInput, Ok, Result
from rateslib.enums.parameters import FXDeltaMethod
from rateslib.fx import FXForwards, FXRates
from rateslib.fx_volatility import FXDeltaVolSmile, FXDeltaVolSurface, FXSabrSmile, FXSabrSurface

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        Any,
        CurveOption_,
        DualTypes,
        FXForwards_,
        _BaseCurve,
        _BaseCurve_,
        _FXVolOption_,
        datetime_,
        str_,
    )


def _maybe_local(
    value: DualTypes,
    local: bool,
    currency: str,
    fx: FXForwards_,
    base: str_,
    forward: datetime_,
) -> dict[str, DualTypes] | DualTypes:
    """
    Return NPVs in scalar form or dict form.
    """
    if local:
        return {currency: value}
    else:
        return _maybe_fx_converted(
            value=value, currency=currency, fx=fx, base=base, forward=forward
        )


def _maybe_fx_converted(
    value: DualTypes,
    currency: str,
    fx: FXForwards_,
    base: str_,
    forward: datetime_,
) -> DualTypes:
    """Take an input Value and maybe FX convert it depending on the inputs"""
    fx_, base = _get_immediate_fx_scalar_and_base(currency=currency, fx=fx, base=base)
    if isinstance(forward, datetime) and base != currency:
        fx_ = fx.rate(f"{currency}{base}", settlement=forward)  # type: ignore[union-attr]
    return value * fx_


def _get_immediate_fx_scalar_and_base(
    currency: str,
    fx: FXForwards_,
    base: str_,
) -> tuple[DualTypes, str]:
    """
    From a local currency and potentially FX Objects determine the conversion rate between
    `currency` and `base`. If `base` is not given it is set as `currency` and the returned
    FX rate is 1.0
    """
    if isinstance(base, NoInput) or base is None:
        if isinstance(fx, NoInput | FXRates | FXForwards):
            return 1.0, currency
        else:  # fx is DualTypes
            if abs(fx - 1.0) < 1e-10:  # type: ignore[operator]
                return fx, currency  # type: ignore[return-value]  # base is assumed
            else:
                warnings.warn(
                    "It is not best practice to provide `fx` as numeric since this can "
                    "cause errors of output when dealing with multi-currency derivatives,\n"
                    "and it also fails to preserve FX rate sensitivity in calculations.\n"
                    "Instead, supply a 'base' currency and use an "
                    "FXRates or, for best practice, an FXForwards object.\n"
                    f"Reformulate: [fx={fx}, base=None] -> "
                    f"[fx=FXRates({{'{currency}bas': {fx}}}), base='bas'].",
                    UserWarning,
                )
            return fx, "Unspecified"  # type: ignore[return-value]  # base is unknown
    else:  # base is str
        if isinstance(fx, NoInput):
            if base != currency:
                raise ValueError(
                    f"`base` ({base}) cannot be requested without supplying `fx` as a "
                    "valid FXRates or FXForwards object to convert from "
                    f"currency ({currency}).\n"
                    "If you are using a `Solver` with multi-currency instruments have you "
                    "forgotten to attach the FXForwards in the solver's `fx` argument?",
                )
            return 1.0, currency
        elif isinstance(fx, FXRates | FXForwards):
            if base == currency:
                return 1.0, currency
            else:
                return fx.rate(pair=f"{currency}{base}"), base
        else:  # FX is DualTypes
            if abs(fx - 1.0) < 1e-10:  # type: ignore[operator]
                pass  # no warning when fx == 1.0
            elif base == currency:
                raise ValueError(
                    "`fx` is given as numeric when `base` and `currency` are the same but the value"
                    "is not equal to 1.0, which it must be by definition."
                )
            else:
                warnings.warn(
                    f"Supplying `fx` as numeric is ambiguous, particularly with "
                    f"multi-currency Instruments, and may lead to forced errors. `base` ({base}) "
                    f"will also be ignored.\n"
                    f"Future versions will likely remove this ability altogether.\n"
                    f"Best practice is to supply `fx` as an FXRates (or FXForwards) object.\n"
                    f"Reformulate the arguments directly: [fx={fx}, base='{base}'] -> "
                    f"[fx=FXRates({{'{currency}{base}': {fx}}}), base='{base}'].",
                    DeprecationWarning,
                )
            return fx, base  # type: ignore[return-value]


def _get_vol_maybe_from_obj(
    fx_vol: _FXVolOption_,
    fx: FXForwards,
    rate_curve: _BaseCurve_,
    strike: DualTypes,
    pair: str,
    delivery: datetime,
    expiry: datetime,
) -> DualTypes:
    """Return a volatility for the option from a given FX Vol object.

    ``rate_curve`` is used as the curve on the LHS rate_curve to convert between spot and delivery
    delta. This is not a 'discount curve' because it is not used to discount cashflows.
    """
    # FXOption can have a `strike` that is NoInput, however this internal function should
    # only be performed after a `strike` has been set to number, temporarily or otherwise.

    if isinstance(fx_vol, FXDeltaVolSmile | FXDeltaVolSurface):
        # fx_vol is a Vol object
        rate_curve_: _BaseCurve = _validate_base_curve(rate_curve)
        spot = fx.pairs_settlement[pair]
        f = fx.rate(pair, delivery)
        _: tuple[Any, DualTypes, Any] = fx_vol.get_from_strike(
            k=strike,
            f=f,
            z_w=rate_curve_[delivery] / rate_curve_[spot],
            expiry=expiry,
        )
        vol_: DualTypes = _[1]
    elif isinstance(fx_vol, FXSabrSmile | FXSabrSurface):
        # fx_vol is a Vol object
        f = fx.rate(pair, delivery)
        _ = fx_vol.get_from_strike(
            k=strike,
            f=f,
            expiry=expiry,
        )
        vol_ = _[1]
    elif isinstance(fx_vol, NoInput):
        raise ValueError("`fx_vol` cannot be NoInput when provided to pricing function.")
    else:
        # fx_vol is a given scalar
        vol_ = fx_vol

    return vol_


def _get_vol_smile_or_value(vol: _FXVolOption_, expiry: datetime) -> FXDeltaVolSmile | DualTypes:
    if isinstance(vol, FXDeltaVolSurface):
        return vol.get_smile(expiry)
    else:
        return _validate_obj_not_no_input(vol, "vol")  # type: ignore[return-value]


def _get_vol_smile_or_raise(vol: _FXVolOption_, expiry: datetime) -> FXDeltaVolSmile:
    if isinstance(vol, FXDeltaVolSurface):
        return vol.get_smile(expiry)
    elif isinstance(vol, FXDeltaVolSmile):
        return vol
    else:
        raise ValueError("Must supply FXDeltaVolSmile/Surface as `vol` not numeric value.")


def _get_vol_delta_type(vol: _FXVolOption_, default_delta_type: FXDeltaMethod) -> FXDeltaMethod:
    if not isinstance(vol, FXDeltaVolSmile | FXDeltaVolSurface):
        return default_delta_type
    else:
        return vol.meta.delta_type


def _validate_fx_as_forwards(fx: FX_) -> FXForwards:
    return _try_validate_fx_as_forwards(fx).unwrap()


def _try_validate_fx_as_forwards(fx: FX_) -> Result[FXForwards]:
    if isinstance(fx, NoInput):
        return Err(ValueError(err.VE_NEEDS_FX_FORWARDS))
    elif not isinstance(fx, FXForwards):
        raise ValueError(err.VE_NEEDS_FX_FORWARDS_BAD_TYPE.format(type(fx).__name__))
    else:
        return Ok(fx)


def _try_validate_base_curve(curve: CurveOption_) -> Result[_BaseCurve]:
    if not isinstance(curve, _BaseCurve):
        return Err(
            TypeError(
                "`curves` have not been supplied correctly.\n"
                f"A _BaseCurve type object is required. Got: {type(curve).__name__}"
            )
        )
    return Ok(curve)


def _validate_base_curve(curve: CurveOption_) -> _BaseCurve:
    if not isinstance(curve, _BaseCurve):
        raise TypeError(
            "`curves` have not been supplied correctly.\n"
            f"A _BaseCurve type object is required. Got: {type(curve).__name__}"
        )
    return curve


def _validate_credit_curves(
    rate_curve: CurveOption_, disc_curve: CurveOption_
) -> Result[tuple[_BaseCurve, _BaseCurve]]:
    # used by Credit type Periods to narrow inputs
    if not isinstance(rate_curve, _BaseCurve):
        return Err(
            TypeError(
                "`curves` have not been supplied correctly.\n"
                "`curve`for a CreditPremiumPeriod must be supplied as a Curve type."
            )
        )
    if not isinstance(disc_curve, _BaseCurve):
        return Err(
            TypeError(
                "`curves` have not been supplied correctly.\n"
                "`disc_curve` for a CreditPremiumPeriod must be supplied as a Curve type."
            )
        )
    return Ok((rate_curve, disc_curve))


def _get_rfr_curve_from_dict(d: dict[str, _BaseCurve]) -> _BaseCurve:
    for s in ["rfr", "RFR", "Rfr"]:
        try:
            ret: _BaseCurve = d[s]
        except KeyError:
            continue
        else:
            return ret
    raise ValueError(
        "A `rate_curve` supplied as dict to an RFR based calculation must contain a key "
        "entry 'rfr'."
    )
