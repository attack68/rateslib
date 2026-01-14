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

from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from pandas import Series

from rateslib import defaults
from rateslib.data.fixings import (
    FloatRateIndex,
    FloatRateSeries,
    IBORFixing,
    IBORStubFixing,
    RFRFixing,
    _get_float_rate_series,
    _RFRRate,
)
from rateslib.enums.generics import (
    NoInput,
    _drb,
)
from rateslib.enums.parameters import (
    FloatFixingMethod,
    SpreadCompoundMethod,
    _get_float_fixing_method,
    _get_spread_compound_method,
)
from rateslib.scheduling import Convention, Frequency
from rateslib.scheduling.adjuster import _convert_to_adjuster
from rateslib.scheduling.frequency import _get_frequency, _get_tenor_from_frequency

if TYPE_CHECKING:
    from rateslib.typing import (
        Adjuster,
        Adjuster_,
        CalTypes,
        DualTypes,
        DualTypes_,
        PeriodFixings,
        datetime,
        int_,
        str_,
    )


def _init_FloatRateParams(
    _float_spread: DualTypes_,
    _rate_fixings: PeriodFixings,
    _fixing_method: FloatFixingMethod | str_,
    _method_param: int_,
    _spread_compound_method: SpreadCompoundMethod | str_,
    _fixing_frequency: Frequency | str_,
    _fixing_series: FloatRateSeries | str_,
    _accrual_start: datetime,
    _accrual_end: datetime,
    _period_calendar: CalTypes,
    _period_adjuster: Adjuster_,
    _period_convention: Convention,
    _period_frequency: Frequency,
    _period_stub: bool,
) -> _FloatRateParams:
    fixing_method: FloatFixingMethod = _get_float_fixing_method(
        _drb(defaults.fixing_method, _fixing_method)
    )
    spread_compound_method = _get_spread_compound_method(
        _drb(defaults.spread_compound_method, _spread_compound_method)
    )
    if isinstance(_method_param, NoInput):
        method_param = defaults.fixing_method_param[fixing_method]
    else:
        method_param = _method_param

    if isinstance(_fixing_series, NoInput):
        # modifier is defaulted to days only type if RFR based
        fixing_series = FloatRateSeries(
            lag=method_param,
            calendar=_period_calendar,
            convention=_period_convention,
            modifier=_convert_to_adjuster(
                modifier=_drb(defaults.modifier, _period_adjuster),
                settlement=False,
                mod_days=fixing_method != FloatFixingMethod.IBOR,
            ),
            eom=defaults.eom,
        )
    else:
        fixing_series = _get_float_rate_series(_fixing_series)

    float_spread = _drb(0.0, _float_spread)
    if isinstance(_fixing_frequency, NoInput):
        if fixing_method == FloatFixingMethod.IBOR:
            fixing_frequency = _period_frequency
        else:
            fixing_frequency = Frequency.BusDays(1, fixing_series.calendar)
    else:
        fixing_frequency = _get_frequency(
            frequency=_fixing_frequency, roll=NoInput(0), calendar=fixing_series.calendar
        )

    fixing_index = FloatRateIndex(
        frequency=fixing_frequency,
        series=fixing_series,
    )

    if fixing_method == FloatFixingMethod.IBOR and not _period_stub:
        if isinstance(_rate_fixings, Series):
            result = IBORFixing._lookup(
                timeseries=_rate_fixings,
                date=fixing_index.calendar.lag_bus_days(_accrual_start, -fixing_index.lag, False),
                bounds=None,
            )
            rate_fixing: IBORFixing | IBORStubFixing | RFRFixing = IBORFixing(
                rate_index=fixing_index,
                accrual_start=_accrual_start,
                date=NoInput(0),
                value=result,
                identifier=NoInput(0),
            )
        else:
            if isinstance(_rate_fixings, str):
                identifier: str_ = _rate_fixings + "_" + _get_tenor_from_frequency(fixing_frequency)
            else:
                identifier = NoInput(0)
            rate_fixing = IBORFixing(
                rate_index=fixing_index,
                accrual_start=_accrual_start,
                date=NoInput(0),
                value=_rate_fixings if not isinstance(_rate_fixings, str) else NoInput(0),
                identifier=identifier,
            )
    elif fixing_method == FloatFixingMethod.IBOR and _period_stub:
        if isinstance(_rate_fixings, Series):
            result = IBORFixing._lookup(
                timeseries=_rate_fixings,
                date=fixing_index.calendar.lag_bus_days(_accrual_start, -fixing_index.lag, False),
                bounds=None,
            )
            rate_fixing = IBORStubFixing(
                rate_series=fixing_series,
                accrual_start=_accrual_start,
                accrual_end=_accrual_end,
                date=NoInput(0),
                value=result,
                identifier=NoInput(0),
            )
        else:
            rate_fixing = IBORStubFixing(
                rate_series=fixing_series,
                accrual_start=_accrual_start,
                accrual_end=_accrual_end,
                date=NoInput(0),
                value=_rate_fixings if not isinstance(_rate_fixings, str) else NoInput(0),
                identifier=_rate_fixings if isinstance(_rate_fixings, str) else NoInput(0),
            )
    else:
        if isinstance(_rate_fixings, Series):
            dates_obs, dates_dcf = RFRFixing._get_date_bounds(
                accrual_start=_accrual_start,
                accrual_end=_accrual_end,
                fixing_method=fixing_method,
                method_param=method_param,
                fixing_calendar=fixing_index.calendar,
            )
            dcfs_dcf = _RFRRate._get_dcf_values(
                dcf_dates=np.array(
                    fixing_index.calendar.bus_date_range(dates_dcf[0], dates_dcf[1])
                ),
                fixing_convention=fixing_index.convention,
                fixing_calendar=fixing_index.calendar,
            )
            result = RFRFixing._lookup(
                timeseries=_rate_fixings,
                fixing_method=fixing_method,
                method_param=method_param,
                spread_compound_method=spread_compound_method,
                float_spread=float_spread,
                dates_obs=np.array(
                    fixing_index.calendar.bus_date_range(dates_obs[0], dates_obs[1])
                ),
                dcfs_dcf=dcfs_dcf,
            )[0]
            rate_fixing = RFRFixing(
                rate_index=fixing_index,
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                accrual_start=_accrual_start,
                accrual_end=_accrual_end,
                fixing_method=fixing_method,
                method_param=method_param,
                value=result,
                identifier=NoInput(0),
            )
        else:
            if isinstance(_rate_fixings, str):
                identifier = _rate_fixings + "_" + _get_tenor_from_frequency(fixing_index.frequency)
            else:
                identifier = NoInput(0)
            rate_fixing = RFRFixing(
                rate_index=fixing_index,
                accrual_start=_accrual_start,
                accrual_end=_accrual_end,
                fixing_method=fixing_method,
                method_param=method_param,
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                value=_rate_fixings if not isinstance(_rate_fixings, str) else NoInput(0),
                identifier=identifier,
            )

    return _FloatRateParams(
        _float_spread=float_spread,
        _spread_compound_method=spread_compound_method,
        _fixing_series=fixing_series,
        _fixing_frequency=fixing_frequency,
        _fixing_method=fixing_method,
        _method_param=method_param,
        _rate_fixing=rate_fixing,
    )


class _CreditParams:
    """
    Parameters associated with credit related *Periods*.

    Parameters
    ----------
    _premium_accrued: bool
        Whether premium *Periods* pay accrued in the event of mid-period default.
    """

    _premium_accrued: bool

    def __init__(self, _premium_accrued: bool):
        self.__premium_accrued = _premium_accrued

    @property
    def premium_accrued(self) -> bool:
        """Whether premium *Periods* pay accrued in the event of mid-period default."""
        return self._premium_accrued


class _FixedRateParams:
    """
    Parameters for a *Period* containing a fixed rate.

    Parameters
    ----------
    _fixed_rate: float, Dual, Dual2, Variable, optional
        The fixed rate defining the *Period* cashflow.
    """

    def __init__(self, _fixed_rate: DualTypes_) -> None:
        self._fixed_rate = _fixed_rate

    @property
    def fixed_rate(self) -> DualTypes | NoInput:
        """The fixed rate defining the *Period* cashflow."""
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self._fixed_rate = value


class _FloatRateParams:
    """
    Parameters for a *Period* containing a floating rate.

    Parameters
    ----------
    _fixing_method: FloatFixingMethod
        The :class:`~rateslib.enums.parameters.FloatFixingMethod` describing the determination
        of the floating rate for the period.
    _method_param: int
        A specific parameter that is used by the specific ``fixing_method``.
    _fixing_series: FloatRateSeries,
        The :class:`~rateslib.enums.parameters.FloatRateSeries` of the
        :class:`~rateslib.enums.parameters.FloatRateIndex` defining the specific interest
        rate index and some of its calculation parameters.
    _fixing_frequency: Frequency,
        The :class:`~rateslib.scheduling.Frequency` of the
        :class:`~rateslib.enums.parameters.FloatRateIndex`.
    _float_spread: float, Dual, Dual2, Variable
        The amount (in bps) added to the rate in the period rate determination.
    _spread_compound_method: SpreadCompoundMethod
        The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation
        of the period rate when combining a ``_float_spread``. Used **only** with RFR type
        ``fixing_method``.
    _rate_fixing: IBORFixing, IBORStubFixing, RFRFixing
        The appropriate rate fixing class that is used to determine if known, published
        values are available for the *Period*.
    """

    def __init__(
        self,
        *,
        _fixing_method: FloatFixingMethod,
        _method_param: int,
        _fixing_series: FloatRateSeries,
        _fixing_frequency: Frequency,
        _float_spread: DualTypes,
        _spread_compound_method: SpreadCompoundMethod,
        _rate_fixing: IBORFixing | IBORStubFixing | RFRFixing,
    ) -> None:
        self._fixing_method: FloatFixingMethod = _fixing_method
        self._spread_compound_method: SpreadCompoundMethod = _spread_compound_method
        self._fixing_series = _fixing_series
        self._fixing_index = FloatRateIndex(
            frequency=_fixing_frequency,
            series=_fixing_series,
        )
        self._float_spread: DualTypes = _float_spread
        self._method_param: int = _method_param
        self._rate_fixing: IBORFixing | IBORStubFixing | RFRFixing = _rate_fixing

        self._validate_combinations_args()

    @property
    def fixing_series(self) -> FloatRateSeries:
        """The :class:`~rateslib.enums.parameters.FloatRateSeries` of the
        :class:`~rateslib.enums.parameters.FloatRateIndex`."""
        return self._fixing_series

    @property
    def fixing_index(self) -> FloatRateIndex:
        """The :class:`~rateslib.enums.parameters.FloatRateIndex` assoociated with the
        determination of the floating rate for the *Period*."""
        return self._fixing_index

    @cached_property
    def fixing_date(self) -> datetime:
        """The relevant date of the (first) rate fixing for the *Period*."""
        if self.fixing_method in [
            FloatFixingMethod.RFRPaymentDelay,
            FloatFixingMethod.RFRPaymentDelayAverage,
            FloatFixingMethod.RFRLockout,
            FloatFixingMethod.RFRLockoutAverage,
        ]:
            return self.accrual_start
        else:
            return self.fixing_calendar.lag_bus_days(
                date=self.accrual_start, days=-self.fixing_series.lag, settlement=False
            )

    @property
    def fixing_convention(self) -> Convention:
        """The day count :class:`~rateslib.scheduling.Convention` of the
        :class:`~rateslib.enums.parameters.FloatRateIndex`."""
        return self.fixing_index.convention

    @property
    def fixing_modifier(self) -> Adjuster:
        """The date :class:`~rateslib.scheduling.Adjuster` of the
        :class:`~rateslib.enums.parameters.FloatRateIndex`."""
        return self.fixing_index.modifier

    @property
    def fixing_frequency(self) -> Frequency:
        """The date :class:`~rateslib.scheduling.Frequency` of the
        :class:`~rateslib.enums.parameters.FloatRateIndex`."""
        return self.fixing_index.frequency

    @property
    def fixing_identifier(self) -> str_:
        """The string identifier provided to ``rate_fixings`` to construct a *Fixings* object."""
        if isinstance(self.rate_fixing, RFRFixing):
            if isinstance(self.rate_fixing.identifier, str):
                return self.rate_fixing.identifier[:-3]  # strip out "_1B"
            return NoInput(0)
        elif isinstance(self.rate_fixing, IBORFixing):
            if isinstance(self.rate_fixing.identifier, str):
                if self.rate_fixing.identifier[-3] == "_":
                    return self.rate_fixing.identifier[:-3]
                else:  # [-4] == "_"
                    return self.rate_fixing.identifier[:-4]
            else:
                return NoInput(0)
        else:  # IBORStubFixing
            if isinstance(self.rate_fixing.identifier, str):
                return self.rate_fixing.identifier  # no suffix
            return NoInput(0)

    @property
    def accrual_start(self) -> datetime:
        """
        The accrual start date for the *Period*.

        Fixing dates will be measured relative to this date under appropriate calendars and
        ``method_param``
        """
        return self.rate_fixing.accrual_start

    @property
    def accrual_end(self) -> datetime:
        """The accrual end date for the *Period*.

        Final fixing dates (or IBOR stub weights) will be measured relative to this date under
        appropriate calendars and ``method_param``.
        """
        return self.rate_fixing.accrual_end

    @property
    def fixing_calendar(self) -> CalTypes:
        """The calendar of the :class:`~rateslib.enums.parameters.FloatRateIndex`."""
        return self.fixing_index.calendar

    @property
    def fixing_method(self) -> FloatFixingMethod:
        """The :class:`~rateslib.enums.parameters.FloatFixingMethod` defining the determination of
        the floating rate for the period."""
        return self._fixing_method

    @property
    def method_param(self) -> int:
        """A parameter used by the ``fixing_method``."""
        return self._method_param

    @property
    def float_spread(self) -> DualTypes:
        """The amount (in bps) added to the rate in the period rate determination."""
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        self._float_spread = value
        self.rate_fixing.reset()

    @property
    def spread_compound_method(self) -> SpreadCompoundMethod:
        """The :class:`~rateslib.enums.parameters.SpreadCompoundMethod` used in the calculation."""
        return self._spread_compound_method

    @property
    def rate_fixing(self) -> IBORFixing | IBORStubFixing | RFRFixing:
        """The :class:`~rateslib.data.fixings.IBORFixing`,
        :class:`~rateslib.data.fixings.IBORStubFixing`, or :class:`~rateslib.data.fixings.RFRFixing`
        appropriate for the *Period*."""
        return self._rate_fixing

    def _validate_combinations_args(self) -> None:
        """
        Validate the argument input to float periods.

        Returns
        -------
        tuple
        """
        if self.method_param != 0 and self.fixing_method in [
            FloatFixingMethod.RFRPaymentDelay,
            FloatFixingMethod.RFRPaymentDelayAverage,
        ]:
            raise ValueError(
                "`method_param` should not be used (or a value other than 0) when "
                f"using a `fixing_method` of 'RFRPaymentDelay' type, got {self.method_param}. "
                f"Configure the `payment_lag` option instead to have the "
                f"appropriate effect.",
            )
        elif self.method_param < 1 and self.fixing_method in [
            FloatFixingMethod.RFRLockout,
            FloatFixingMethod.RFRLockoutAverage,
        ]:
            raise ValueError(
                f'`method_param` must be >0 for "RFRLockout" type `fixing_method`, '
                f"got {self.method_param}",
            )
