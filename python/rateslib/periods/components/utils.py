from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from rateslib.enums.generics import NoInput
from rateslib.fx import FXForwards, FXRates

if TYPE_CHECKING:
    from rateslib.typing import (
        FX_,
        DualTypes,
        str_,
    )


def _maybe_local(
    value: DualTypes,
    local: bool,
    currency: str,
    fx: FX_,
    base: str_,
) -> dict[str, DualTypes] | DualTypes:
    """
    Return NPVs in scalar form or dict form.
    """
    if local:
        return {currency: value}
    else:
        return _maybe_fx_converted(value=value, currency=currency, fx=fx, base=base)


def _maybe_fx_converted(
    value: DualTypes,
    currency: str,
    fx: FX_,
    base: str | NoInput,
) -> DualTypes:
    """Take an input Value and maybe FX convert it depending on the inputs"""
    fx_, _ = _get_immediate_fx_scalar_and_base(currency=currency, fx=fx, base=base)
    return value * fx_


def _get_immediate_fx_scalar_and_base(
    currency: str,
    fx: FX_,
    base: str_,
) -> tuple[DualTypes, str]:
    """
    From a local currency and potentially FX Objects determine the conversion rate between
    `currency` and `base`. If `base` is not given it is set as `currency` and the returned
    FX rate is 1.0
    """
    if isinstance(base, NoInput) or base is None:
        if isinstance(fx, NoInput | (FXRates | FXForwards)):
            return 1.0, currency
        else:  # fx is DualTypes
            if abs(fx - 1.0) < 1e-10:
                return fx, currency  # base is assumed
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
            return fx, "Unspecified"  # base is unknown
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
            if abs(fx - 1.0) < 1e-10:
                pass  # no warning when fx == 1.0
            elif base == currency:
                raise ValueError(
                    "`fx` is given as numeric when `base` and `currency` are the same but the value"
                    "is not equal to 1.0, which it must be by definition."
                )
            else:
                warnings.warn(
                    f"`base` ({base}) should not be given when supplying `fx` as numeric "
                    f"since it will not be used.\n"
                    f"Future versions may remove this ability since it may also be interpreted "
                    f"as giving wrong results.\nBest practice is to instead supply `fx` as an "
                    f"FXRates (or FXForwards) object.\n"
                    f"Reformulate the arguments directly: [fx={fx}, base='{base}'] -> "
                    f"[fx=FXRates({{'{currency}{base}': {fx}}}), base='{base}'].",
                    DeprecationWarning,
                )
            return fx, base
