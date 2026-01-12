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

from typing import TYPE_CHECKING

from pandas import Series

from rateslib import NoInput
from rateslib.data.fixings import _get_float_rate_series_or_blank, _IBORRate, _RFRRate
from rateslib.enums.generics import Err, Ok, _drb
from rateslib.enums.parameters import (
    FloatFixingMethod,
    SpreadCompoundMethod,
    _get_float_fixing_method,
    _get_spread_compound_method,
)
from rateslib.periods.utils import _get_rfr_curve_from_dict
from rateslib.scheduling.frequency import _get_frequency

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CurveOption_,
        DualTypes,
        DualTypes_,
        FloatRateSeries,
        Frequency,
        Result,
        Series,
        _BaseCurve_,
        datetime,
        str_,
    )


def rate_value(
    start: datetime,
    end: datetime,
    rate_curve: CurveOption_ = NoInput(0),
    *,
    rate_fixings: DualTypes_ | str = NoInput(0),
    frequency: Frequency | str_ = NoInput(0),
    rate_series: FloatRateSeries | str_ = NoInput(0),
    fixing_method: FloatFixingMethod | str = FloatFixingMethod.RFRPaymentDelay,
    method_param: int = 0,
    spread_compound_method: SpreadCompoundMethod | str = SpreadCompoundMethod.NoneSimple,
    float_spread: DualTypes = 0.0,
    stub: bool = False,
) -> DualTypes:
    return try_rate_value(
        start=start,
        end=end,
        rate_curve=rate_curve,
        rate_series=rate_series,
        frequency=frequency,
        rate_fixings=rate_fixings,
        fixing_method=fixing_method,
        method_param=method_param,
        spread_compound_method=spread_compound_method,
        float_spread=float_spread,
        stub=stub,
    ).unwrap()


def try_rate_value(
    start: datetime,
    end: datetime,
    rate_curve: CurveOption_ = NoInput(0),
    *,
    rate_series: FloatRateSeries | str_ = NoInput(0),
    frequency: Frequency | str_ = NoInput(0),
    rate_fixings: DualTypes | Series[DualTypes] | str_ = NoInput(0),  # type: ignore[type-var]
    fixing_method: FloatFixingMethod | str = FloatFixingMethod.RFRPaymentDelay,
    method_param: int = 0,
    spread_compound_method: SpreadCompoundMethod | str = SpreadCompoundMethod.NoneSimple,
    float_spread: DualTypes = 0.0,
    stub: bool = False,
) -> Result[DualTypes]:
    """
    Derive a floating rate value from a combination of market inputs.


    """
    fm = _get_float_fixing_method(fixing_method)
    scm = _get_spread_compound_method(spread_compound_method)
    rs = _get_float_rate_series_or_blank(rate_series)
    if fm == FloatFixingMethod.IBOR:
        return _IBORRate._rate(
            start=start,
            end=end,
            rate_curve=rate_curve,
            rate_fixings=rate_fixings,
            method_param=method_param,
            float_spread=_drb(0.0, float_spread),
            stub=stub,
            rate_series=rs,
            frequency=_get_frequency(frequency, NoInput(0), NoInput(0)),
        )
    else:  #  RFR based
        if isinstance(rate_curve, dict):
            rate_curve_: _BaseCurve_ = _get_rfr_curve_from_dict(rate_curve)
        else:
            rate_curve_ = rate_curve
        r_result = _RFRRate._rate(
            start=start,
            end=end,
            rate_curve=rate_curve_,
            rate_fixings=rate_fixings,
            fixing_method=fm,
            method_param=method_param,
            spread_compound_method=scm,
            float_spread=float_spread,
            rate_series=rs,
        )
        if isinstance(r_result, Err):
            return r_result
        else:
            return Ok(r_result.unwrap()[0])
