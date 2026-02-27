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

from enum import Enum
from typing import TYPE_CHECKING, Never

from rateslib.rs import FloatFixingMethod, IROptionMetric, LegIndexBase

if TYPE_CHECKING:
    pass


class OptionType(float, Enum):
    """
    Enumerable type to define option directions.
    """

    Put = -1.0
    Call = 1.0


class FXOptionMetric(Enum):
    """
    Enumerable type for FXOption metrics.
    """

    Pips = 0
    Percent = 1


class SwaptionSettlementMethod(Enum):
    """
    Enumerable type for swaption settlement methods.
    """

    Physical = 0
    CashParTenor = 1
    CashCollateralized = 2


class LegMtm(Enum):
    """
    Enumerable type to define :class:`~rateslib.data.fixings.FXFixing` dates for non-deliverable
    *Legs*.

    For further information see non-deliverability **Notes** of :class:`~rateslib.legs.FixedLeg`.
    """

    Initial = 0
    XCS = 1
    Payment = 2


class IndexMethod(Enum):
    """
    Enumerable type to define determining the index value on some reference value date.

    Notes
    -----
    ``Curve`` variant derives an index value directly from a *Curve* by using its discount factors
    and its index base date.
    """

    Daily = 0
    Monthly = 1
    Curve = 2

    def __str__(self) -> str:
        return self.name


class SpreadCompoundMethod(Enum):
    """
    Enumerable type to define spread compounding methods for floating rates.
    """

    NoneSimple = 0
    ISDACompounding = 1
    ISDAFlatCompounding = 2

    def __str__(self) -> str:
        return self.name


class FXDeltaMethod(Enum):
    """
    Enumerable type to define the delta expression of an FX option.
    """

    Forward = 0
    Spot = 1
    ForwardPremiumAdjusted = 2
    SpotPremiumAdjusted = 3

    def __str__(self) -> str:
        return self.name


_SWAPTION_SETTLEMENT_MAP = {
    "physical": SwaptionSettlementMethod.Physical,
    "cash_par_tenor": SwaptionSettlementMethod.CashParTenor,
    "cash_collateralized": SwaptionSettlementMethod.CashCollateralized,
    # aliases
    "cashcollateralized": SwaptionSettlementMethod.CashCollateralized,
    "cashpartenor": SwaptionSettlementMethod.CashParTenor,
}


def _get_swaption_settlement_method(
    method: str | SwaptionSettlementMethod,
) -> SwaptionSettlementMethod:
    if isinstance(method, SwaptionSettlementMethod):
        return method
    else:
        try:
            return _SWAPTION_SETTLEMENT_MAP[method.lower()]
        except KeyError:
            raise ValueError(
                f"`swaption_settlement_method` as string: '{method}' is not a valid option. "
                f"Please consult docs."
            )


_LEG_MTM_MAP = {
    "initial": LegMtm.Initial,
    "xcs": LegMtm.XCS,
    "payment": LegMtm.Payment,
}


def _get_leg_mtm(leg_mtm: str | LegMtm) -> LegMtm:
    if isinstance(leg_mtm, LegMtm):
        return leg_mtm
    else:
        try:
            return _LEG_MTM_MAP[leg_mtm.lower()]
        except KeyError:
            raise ValueError(
                f"`mtm` as string: '{leg_mtm}' is not a valid option. Please consult docs."
            )


_INDEX_METHOD_MAP = {
    "daily": IndexMethod.Daily,
    "monthly": IndexMethod.Monthly,
    "curve": IndexMethod.Curve,
}


def _get_index_method(index_method: str | IndexMethod) -> IndexMethod:
    if isinstance(index_method, IndexMethod):
        return index_method
    else:
        try:
            return _INDEX_METHOD_MAP[index_method.lower()]
        except KeyError:
            raise ValueError(
                f"`index_method` as string: '{index_method}' is not a valid option. "
                f"Please consult docs."
            )


_FIXING_METHOD_MAP: dict[str, type[FloatFixingMethod]] = {
    "ibor": FloatFixingMethod.IBOR,
    "rfrpaymentdelay": FloatFixingMethod.RFRPaymentDelay,
    "rfrobservationshift": FloatFixingMethod.RFRObservationShift,
    "rfrlockout": FloatFixingMethod.RFRLockout,
    "rfrlookback": FloatFixingMethod.RFRLookback,
    "rfrpaymentdelayaverage": FloatFixingMethod.RFRPaymentDelayAverage,
    "rfrobservationshiftaverage": FloatFixingMethod.RFRObservationShiftAverage,
    "rfrlockoutaverage": FloatFixingMethod.RFRLockoutAverage,
    "rfrlookbackaverage": FloatFixingMethod.RFRLookbackAverage,
    # legacy compatibility
    "rfr_payment_delay": FloatFixingMethod.RFRPaymentDelay,
    "rfr_observation_shift": FloatFixingMethod.RFRObservationShift,
    "rfr_lockout": FloatFixingMethod.RFRLockout,
    "rfr_lookback": FloatFixingMethod.RFRLookback,
    "rfr_payment_delay_avg": FloatFixingMethod.RFRPaymentDelayAverage,
    "rfr_observation_shift_avg": FloatFixingMethod.RFRObservationShiftAverage,
    "rfr_lockout_avg": FloatFixingMethod.RFRLockoutAverage,
    "rfr_lookback_avg": FloatFixingMethod.RFRLookbackAverage,
}


def _get_float_fixing_method(method: str | FloatFixingMethod) -> FloatFixingMethod:
    if isinstance(method, FloatFixingMethod):
        return method
    else:
        if method.lower() in ["rfrpaymentdelay", "rfr_payment_delay"]:
            return FloatFixingMethod.RFRPaymentDelay()
        elif method.lower() in ["rfrpaymentdelayaverage", "rfr_payment_delay_avg"]:
            return FloatFixingMethod.RFRPaymentDelayAverage()

        if not ("(" in method and method[-1] == ")"):
            raise ValueError(
                f"`fixing_method` as string: '{method}' must have an associated parameter "
                f"contained in parentheses, for example 'ibor(2)' or 'rfr_observation_shift(5)'. "
            )

        method_, number_part = method[:-1].split("(")
        number = int(number_part)
        try:
            enum_ = _FIXING_METHOD_MAP[method_.lower()]
        except KeyError:
            raise ValueError(
                f"`fixing_method` as string: '{method_}' is not a valid FloatFixingMethod."
            )
        return enum_(number)  # type: ignore[call-arg]


_SPREAD_COMPOUNDING_METHOD_MAP = {
    "nonesimple": SpreadCompoundMethod.NoneSimple,
    "isdacompounding": SpreadCompoundMethod.ISDACompounding,
    "isdaflatcompounding": SpreadCompoundMethod.ISDAFlatCompounding,
    # legacy compatibility
    "none_simple": SpreadCompoundMethod.NoneSimple,
    "isda_compounding": SpreadCompoundMethod.ISDACompounding,
    "isda_flat_compounding": SpreadCompoundMethod.ISDAFlatCompounding,
}


def _get_spread_compound_method(method: str | SpreadCompoundMethod) -> SpreadCompoundMethod:
    if isinstance(method, SpreadCompoundMethod):
        return method
    else:
        try:
            return _SPREAD_COMPOUNDING_METHOD_MAP[method.lower()]
        except KeyError:
            raise ValueError(
                f"`spread_compound_method` as string: '{method}' is not a valid option. "
                f"Please consult docs."
            )


_FX_DELTA_TYPE_MAP = {
    "forward": FXDeltaMethod.Forward,
    "spot": FXDeltaMethod.Spot,
    "forward_pa": FXDeltaMethod.ForwardPremiumAdjusted,
    "spot_pa": FXDeltaMethod.SpotPremiumAdjusted,
    "forwardpremkiumadjusted": FXDeltaMethod.ForwardPremiumAdjusted,
    "spotpremiumadjusted": FXDeltaMethod.SpotPremiumAdjusted,
}


def _get_fx_delta_type(method: str | FXDeltaMethod) -> FXDeltaMethod:
    if isinstance(method, FXDeltaMethod):
        return method
    else:
        try:
            return _FX_DELTA_TYPE_MAP[method.lower()]
        except KeyError:
            raise ValueError(
                f"`delta_type` as string: '{method}' is not a valid option. Please consult docs."
            )


_FX_METRIC_MAP = {
    "pips": FXOptionMetric.Pips,
    "percent": FXOptionMetric.Percent,
}


def _get_fx_option_metric(method: str | FXOptionMetric) -> FXOptionMetric:
    if isinstance(method, FXOptionMetric):
        return method
    else:
        try:
            return _FX_METRIC_MAP[method.lower()]
        except KeyError:
            raise ValueError(
                f"FXOption `metric` as string: '{method}' is not a valid option. Please consult "
                f"docs."
            )


_IR_METRIC_MAP: dict[str, type[IROptionMetric]] = {
    "normal_vol": IROptionMetric.NormalVol,
    "log_normal_vol": IROptionMetric.LogNormalVol,
    "cash": IROptionMetric.Cash,
    "percent_notional": IROptionMetric.PercentNotional,
    "black_vol_shift": IROptionMetric.BlackVolShift,
    # aliases
    "normalvol": IROptionMetric.NormalVol,
    "lognormalvol": IROptionMetric.LogNormalVol,
    "percentnotional": IROptionMetric.PercentNotional,
    "blackvolshift": IROptionMetric.BlackVolShift,
}


def _get_ir_option_metric(method: str | IROptionMetric) -> IROptionMetric:
    if isinstance(method, IROptionMetric):
        return method
    else:
        method = method.lower()
        if "shift" in method:
            idx = method.rfind("_")
            if idx < 0:
                raise ValueError(
                    "The 'BlackVolShift' metric must have an underscore and shift, e.g. "
                    "'black_vol_shift_100"
                )
            else:
                args: tuple[Never, ...] | tuple[int]= (int(method[idx + 1 :]),)
            method = method[:idx]
        else:
            args = tuple()

        try:
            return _IR_METRIC_MAP[method](*args)
        except KeyError:
            raise ValueError(
                f"IROption `metric` as string: '{method}' is not a valid option. Please consult "
                f"documentation."
            )


__all__ = [
    "SpreadCompoundMethod",
    "FloatFixingMethod",
    "FXDeltaMethod",
    "FXOptionMetric",
    "IROptionMetric",
    "LegMtm",
    "LegIndexBase",
    "OptionType",
    "IndexMethod",
]
