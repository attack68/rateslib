from __future__ import annotations

import warnings
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from pandas import Series

import rateslib.errors as err
from rateslib import defaults, fixings
from rateslib.enums.generics import (
    Err,
    NoInput,
    _drb,
)
from rateslib.enums.parameters import (
    FloatFixingMethod,
    SpreadCompoundMethod,
    _get_float_fixing_method,
    _get_spread_compound_method,
)
from rateslib.fixing_data import FixingRangeError
from rateslib.periods.components.float_rate import _RFRRate
from rateslib.periods.components.parameters.base_fixing import _BaseFixing
from rateslib.scheduling import Convention, Frequency, add_tenor
from rateslib.scheduling.adjuster import _convert_to_adjuster
from rateslib.scheduling.float_rate_index import (
    FloatRateIndex,
    FloatRateSeries,
    _get_float_rate_series,
)
from rateslib.scheduling.frequency import _get_frequency, _get_tenor_from_frequency

if TYPE_CHECKING:
    from rateslib.typing import (
        Adjuster,
        Adjuster_,
        Arr1dF64,
        Arr1dObj,
        CalTypes,
        DualTypes,
        DualTypes_,
        datetime,
        datetime_,
        int_,
        str_,
    )


def _init_FloatRateParams(
    _float_spread: DualTypes_,
    _rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
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
    _premium_accrued: bool

    def __init__(self, _premium_accrued: bool):
        self.__premium_accrued = _premium_accrued

    @property
    def premium_accrued(self) -> bool:
        return self._premium_accrued


class _CashflowRateParams:
    pass


class _FixedRateParams:
    def __init__(self, _fixed_rate: DualTypes_) -> None:
        self._fixed_rate = _fixed_rate

    @property
    def fixed_rate(self) -> DualTypes | NoInput:
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: DualTypes_) -> None:
        self._fixed_rate = value


class _FloatRateParams:
    _fixing_method: FloatFixingMethod
    _method_param: int
    _fixing_index: FloatRateIndex
    _float_spread: DualTypes
    _spread_compound_method: SpreadCompoundMethod
    _rate_fixing: IBORFixing | IBORStubFixing | RFRFixing

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
        self._spread_compound_method = _spread_compound_method
        self._fixing_series = _fixing_series
        self._fixing_index = FloatRateIndex(
            frequency=_fixing_frequency,
            series=_fixing_series,
        )
        self._float_spread = _float_spread
        self._method_param: int = _method_param
        self._rate_fixing = _rate_fixing

        self._validate_combinations_args()

    @property
    def fixing_series(self) -> FloatRateSeries:
        return self._fixing_series

    @property
    def fixing_index(self) -> FloatRateIndex:
        return self._fixing_index

    @cached_property
    def fixing_date(self) -> datetime:
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
        return self.fixing_index.convention

    @property
    def fixing_modifier(self) -> Adjuster:
        return self.fixing_index.modifier

    @property
    def fixing_frequency(self) -> Frequency:
        return self.fixing_index.frequency

    @property
    def accrual_start(self) -> datetime:
        return self.rate_fixing.accrual_start

    @property
    def accrual_end(self) -> datetime:
        return self.rate_fixing.accrual_end

    @property
    def fixing_calendar(self) -> CalTypes:
        return self.fixing_index.calendar

    @property
    def fixing_method(self) -> FloatFixingMethod:
        return self._fixing_method

    @property
    def method_param(self) -> int:
        return self._method_param

    @property
    def float_spread(self) -> DualTypes:
        return self._float_spread

    @float_spread.setter
    def float_spread(self, value: DualTypes) -> None:
        self._float_spread = value
        self.rate_fixing.reset()

    @property
    def spread_compound_method(self) -> SpreadCompoundMethod:
        return self._spread_compound_method

    @property
    def rate_fixing(self) -> IBORFixing | IBORStubFixing | RFRFixing:
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


class IBORFixing(_BaseFixing):
    """
    A rate fixing value referencing a tenor-IBOR type calculation.

    Parameters
    ----------
    rate_index: FloatRateIndex
        The parameters associated with the floating rate index.
    accrual_start: datetime
        The start accrual date for the period of the floating rate.
    date: datetime
        The date of relevance for the fixing, which is its **publication** date. This can
        be determined by a ``lag`` parameter of the ``rate_index`` measured from the
        ``accrual_start``.
    value: float, Dual, Dual2, Variable, optional
        The initial value for the fixing to adopt. Most commonly this is not given and it is
        determined from a timeseries of published FX rates.
    identifier: str, optional
        The string name of the timeseries to be loaded by the *Fixings* object.

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.periods.components.parameters import IBORFixing
       from rateslib.scheduling.float_rate_index import FloatRateIndex
       from rateslib import fixings, dt
       from pandas import Series

    .. ipython:: python

       fixings.add("EURIBOR_3M", Series(index=[dt(2000, 1, 3), dt(2000, 2, 4)], data=[1.651, 1.665]))
       ibor_fix = IBORFixing(
           accrual_start=dt(2000, 1, 5),
           identifier="Euribor_3m",
           rate_index=FloatRateIndex(frequency="Q", series="eur_ibor")
       )
       ibor_fix.date
       ibor_fix.value

    .. ipython:: python
       :suppress:

       fixings.pop("Euribor_3m")

    """  # noqa: E501

    _accrual_start: datetime
    _accrual_end: datetime
    _rate_index: FloatRateIndex

    def __init__(
        self,
        *,
        rate_index: FloatRateIndex,
        accrual_start: datetime,
        date: datetime_ = NoInput(0),
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ) -> None:
        super().__init__(date=date, value=value, identifier=identifier)  # type: ignore[arg-type]
        self._accrual_start = accrual_start
        self._rate_index = rate_index
        self._date = _drb(
            self.index.calendar.lag_bus_days(self.accrual_start, -self.index.lag, False),
            date,
        )
        self._accrual_end = add_tenor(
            start=self.accrual_start,
            tenor=self.index.frequency,
            modifier=self.index.modifier,
            calendar=self.index.calendar,
        )

    @property
    def index(self) -> FloatRateIndex:
        """The definitions for the :class:`FloatRateIndex` of the fixing."""
        return self._rate_index

    @property
    def series(self) -> FloatRateSeries:
        """The :class:`FloatRateSeries` for defining the fixing."""
        return self.index.series

    @property
    def accrual_start(self) -> datetime:
        """The start accrual date for the defined period of the floating rate."""
        return self._accrual_start

    @property
    def accrual_end(self) -> datetime:
        """The end accrual date for the defined period of the floating rate."""
        return self._accrual_end

    def _lookup_and_calculate(
        self,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        bounds: tuple[datetime, datetime] | None,
    ) -> DualTypes_:
        return self._lookup(timeseries=timeseries, bounds=bounds, date=self.date)

    @classmethod
    def _lookup(
        cls,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        date: datetime,
        bounds: tuple[datetime, datetime] | None = None,
    ) -> DualTypes_:
        result = fixings.__base_lookup__(
            fixing_series=timeseries,
            lookup_date=date,
            bounds=bounds,
        )
        if isinstance(result, Err):
            if isinstance(result._exception, FixingRangeError):
                return NoInput(0)
            result.unwrap()
        else:
            return result.unwrap()


class IBORStubFixing(_BaseFixing):
    """
    A rate fixing value referencing an interpolated tenor-IBOR type calculation.

    Parameters
    ----------
    rate_series: FloatRateSeries
        The parameters associated with the floating rate index.
    accrual_start: datetime
        The start accrual date for the period.
    accrual_end: datetime
        The end accrual date for the period..
    date: datetime, optional
        The date of relevance for the fixing, which is its **publication** date. This can
        be determined by a ``lag`` parameter of the ``rate_series`` measured from the
        ``accrual_start``.
    value: float, Dual, Dual2, Variable, optional
        The initial value for the fixing to adopt. Most commonly this is not given and it is
        determined from a timeseries of published FX rates.
    identifier: str, optional
        The string name of the timeseries to be loaded by the *Fixings* object. This is a
        *series* identifier, e.g. "Euribor", which will be extended to derive the full
        version, e.g. "Euribor_3m" based on available and necessary tenors.

    Notes
    -----
    An interpolated tenor-IBOR type calculation depends upon two tenors being available from
    the *Fixings* object. Appropriate tenors will be automatically selected based on the
    ``accrual_end`` date. If only one tenor is available, this will be used as the single
    ``fixing1`` value.

    Examples
    --------

    This fixing automatically identifies it must be interpolated between the available 3M and 6M
    tenors.

    .. ipython:: python
       :suppress:

       from rateslib.periods.components.parameters import IBORStubFixing
       from rateslib.scheduling.float_rate_index import FloatRateSeries
       from rateslib import fixings, dt
       from pandas import Series

    .. ipython:: python

       fixings.add("EURIBOR_1M", Series(index=[dt(2000, 1, 3), dt(2000, 2, 4)], data=[1.651, 1.665]))
       fixings.add("EURIBOR_2M", Series(index=[dt(2000, 1, 3), dt(2000, 2, 4)], data=[2.651, 2.665]))
       fixings.add("EURIBOR_3M", Series(index=[dt(2000, 1, 3), dt(2000, 2, 4)], data=[3.651, 3.665]))
       fixings.add("EURIBOR_6M", Series(index=[dt(2000, 1, 3), dt(2000, 2, 4)], data=[4.651, 4.665]))
       ibor_fix = IBORStubFixing(
           accrual_start=dt(2000, 1, 5),
           accrual_end=dt(2000, 5, 17),
           identifier="Euribor",
           rate_series=FloatRateSeries(
               lag=2,
               modifier="MF",
               calendar="tgt",
               convention="act360",
               eom=False,
           )
       )
       ibor_fix.date
       ibor_fix.value

    .. ipython:: python
       :suppress:

       fixings.pop("Euribor_1m")
       fixings.pop("Euribor_2m")
       fixings.pop("Euribor_3m")
       fixings.pop("Euribor_6m")

    This fixing can only be determined from a single tenor, which is quite distinct from the
    stub tenor in this case.

    .. ipython:: python

       fixings.add("NIBOR_6M", Series(index=[dt(2000, 1, 3), dt(2000, 2, 4)], data=[4.651, 4.665]))
       ibor_fix = IBORStubFixing(
           accrual_start=dt(2000, 1, 5),
           accrual_end=dt(2000, 1, 17),
           identifier="Nibor",
           rate_series=FloatRateSeries(
               lag=2,
               modifier="MF",
               calendar="osl",
               convention="act360",
               eom=True,
           )
       )
       ibor_fix.date
       ibor_fix.value
       ibor_fix.fixing2

    The following fixing cannot identify any tenor indices in the *Fixings* object, and will
    log a *UserWarning* before proceeding to yield *NoInput* for all values.

    .. ipython:: python
       :okwarning:

       ibor_fix = IBORStubFixing(
           accrual_start=dt(2000, 1, 5),
           accrual_end=dt(2000, 1, 17),
           identifier="Unavailable_Identifier",
           rate_series=FloatRateSeries(
               lag=2,
               modifier="MF",
               calendar="nyc",
               convention="act360",
               eom=True,
           )
       )
       ibor_fix.date
       ibor_fix.value
       ibor_fix.fixing1
       ibor_fix.fixing2

    """  # noqa: E501

    _accrual_start: datetime
    _accrual_end: datetime
    _series: FloatRateSeries
    _fixing1: IBORFixing | NoInput
    _fixing2: IBORFixing | NoInput

    def __init__(
        self,
        *,
        rate_series: FloatRateSeries,
        accrual_start: datetime,
        accrual_end: datetime,
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
        date: datetime_ = NoInput(0),
    ) -> None:
        super().__init__(value=value, date=date, identifier=identifier)  # type: ignore[arg-type]
        self._accrual_start = accrual_start
        self._accrual_end = accrual_end
        self._series = rate_series
        self._date = _drb(
            self.series.calendar.lag_bus_days(self.accrual_start, -self.series.lag, False),
            date,
        )

        if isinstance(value, NoInput):
            if isinstance(identifier, NoInput):
                self._fixing2 = NoInput(0)
                self._fixing1 = NoInput(0)
            else:
                # then populate additional required information
                tenors = self._stub_tenors()
                if len(tenors[0]) in [1, 2]:
                    self._fixing1 = IBORFixing(
                        rate_index=FloatRateIndex(
                            series=self.series,
                            frequency=_get_frequency(tenors[0][0], NoInput(0), NoInput(0)),
                        ),
                        accrual_start=self.accrual_start,
                        date=date,
                        value=NoInput(0),
                        identifier=identifier + "_" + tenors[0][0],
                    )
                    if len(tenors[0]) == 2:
                        self._fixing2 = IBORFixing(
                            rate_index=FloatRateIndex(
                                series=self._series,
                                frequency=_get_frequency(tenors[0][1], NoInput(0), NoInput(0)),
                            ),
                            date=date,
                            accrual_start=self.accrual_start,
                            value=NoInput(0),
                            identifier=identifier + "_" + tenors[0][1],
                        )
                    else:
                        self._fixing2 = NoInput(0)
                else:
                    warnings.warn(err.UW_NO_TENORS.format(identifier))
                    self._fixing2 = NoInput(0)
                    self._fixing1 = NoInput(0)
        else:
            self._value = value

    @property
    def date(self) -> datetime:
        """The date of relevance for the fixing, which is its **publication** date."""
        return self._date

    @property
    def fixing1(self) -> IBORFixing | NoInput:
        """The shorter tenor :class:`IBORFixing` making up part of the calculation."""
        return self._fixing1

    @property
    def fixing2(self) -> IBORFixing | NoInput:
        """The longer tenor :class:`IBORFixing` making up part of the calculation."""
        return self._fixing2

    @property
    def value(self) -> DualTypes_:
        if not isinstance(self._value, NoInput):
            return self._value
        elif isinstance(self.fixing1, NoInput) or isinstance(self.fixing1.value, NoInput):
            return NoInput(0)
        else:
            if isinstance(self.fixing2, NoInput):
                self._value = self.fixing1.value
                return self._value
            elif isinstance(self.fixing2.value, NoInput):
                return NoInput(0)
            else:
                self._value = (
                    self.weights[0] * self.fixing1.value + self.weights[1] * self.fixing2.value
                )
                return self._value

    @cached_property
    def weights(self) -> tuple[float, float]:
        """Scalar multiplier to apply to each tenor fixing for the interpolation."""
        if isinstance(self.fixing2, NoInput):
            if isinstance(self.fixing1, NoInput):
                raise ValueError(
                    "The IBORStubFixing has no individual IBORFixings to determine weights."
                )
            return 1.0, 0.0
        else:
            e1 = self.fixing1.accrual_end  # type: ignore[union-attr]
            e2 = self.fixing2.accrual_end
            e = self.accrual_end
            return (e2 - e) / (e2 - e1), (e - e1) / (e2 - e1)

    @property
    def series(self) -> FloatRateSeries:
        """The :class:`FloatRateSeries` for defining the fixing."""
        return self._series

    @property
    def accrual_start(self) -> datetime:
        """The start accrual date for the defined accrual period."""
        return self._accrual_start

    @property
    def accrual_end(self) -> datetime:
        """The end accrual date for the defined accrual period."""
        return self._accrual_end

    def _lookup_and_calculate(
        self,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        bounds: tuple[datetime, datetime] | None,
    ) -> DualTypes_:
        raise RuntimeError("This method should be unused due to overloaded properties")

    def _stub_tenors(self) -> tuple[list[str], list[datetime]]:
        """
        Return the tenors available in the :class:`~rateslib.defaults.Fixings` object for
        determining an IBOR type stub period.

        Returns
        -------
        tuple of list[string tenors] and list[evaluated end dates]
        """
        from rateslib.scheduling import add_tenor

        left: tuple[str | None, datetime] = (None, datetime(1, 1, 1))
        right: tuple[str | None, datetime] = (None, datetime(9999, 1, 1))

        for tenor in [
            "1D",
            "1B",
            "2B",
            "1W",
            "2W",
            "3W",
            "4W",
            "1M",
            "2M",
            "3M",
            "4M",
            "5M",
            "6M",
            "7M",
            "8M",
            "9M",
            "10M",
            "11M",
            "12M",
            "1Y",
        ]:
            try:
                _ = fixings.__getitem__(f"{self.identifier}_{tenor}")
            except Exception:  # noqa: S112
                continue
            else:
                sample_end = add_tenor(
                    start=self.accrual_start,
                    tenor=tenor,
                    modifier=self.series.modifier,
                    calendar=self.series.calendar,
                )
                if sample_end <= self.accrual_end and sample_end > left[1]:
                    left = (tenor, sample_end)
                if sample_end >= self.accrual_end and sample_end < right[1]:
                    right = (tenor, sample_end)
                    break

        ret: tuple[list[str], list[datetime]] = ([], [])
        if left[0] is not None:
            ret[0].append(left[0])
            ret[1].append(left[1])
        if right[0] is not None:
            ret[0].append(right[0])
            ret[1].append(right[1])
        return ret


class RFRFixing(_BaseFixing):
    """
    A rate fixing value representing an RFR type calculating involving multiple RFR publications.

    Parameters
    ----------
    rate_index: FloatRateIndex
        The parameters associated with the floating rate index.
    accrual_start: datetime
        The start accrual date for the period.
    accrual_end: datetime
        The end accrual date for the period.
    value: float, Dual, Dual2, Variable, optional
        The initial value for the fixing to adopt. Most commonly this is not given and it is
        determined from a timeseries of published FX rates.
    identifier: str, optional
        The string name of the timeseries to be loaded by the *Fixings* object. For alignment with
        internal structuring these should have the suffix "_1B", e.g. "ESTR_1B".
    fixing_method: FloatFixingMethod
        The :class:`FloatFixingMethod` object used to combine multiple RFR fixings.
    method_param: int
        A parameter required by the ``fixing_method``.
    spread_compound_method: SpreadCompoundMethod
        A :class:`SpreadCompoundMethod` object used define the calculation of the addition of the
        ``float_spread``.
    float_spread: float, DUal, Dual2, Variable
        An additional amount added to the calculation to determine the final period rate.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib.enums.parameters import SpreadCompoundMethod, FloatFixingMethod
       from rateslib.periods.components.parameters import RFRFixing
       from rateslib.scheduling.float_rate_index import FloatRateIndex
       from rateslib import fixings, dt
       from pandas import Series

    The below is a fully determined *RFRFixing* with populated rates.

    .. ipython:: python

       fixings.add("SOFR_1B", Series(index=[
            dt(2025, 1, 8), dt(2025, 1, 9), dt(2025, 1, 10), dt(2025, 1, 13), dt(2025, 1, 14)
          ], data=[1.1, 2.2, 3.3, 4.4, 5.5]))

       rfr_fix = RFRFixing(
           accrual_start=dt(2025, 1, 9),
           accrual_end=dt(2025, 1, 15),
           identifier="SOFR_1B",
           spread_compound_method=SpreadCompoundMethod.NoneSimple,
           fixing_method=FloatFixingMethod.RFRPaymentDelay,
           method_param=0,
           float_spread=0.0,
           rate_index=FloatRateIndex(frequency="1B", series="usd_rfr")
       )
       rfr_fix.value
       rfr_fix.populated

    This second example is a partly undetermined period, and will result in *NoInput* for its
    value but has recorded partial population of its individual RFRs.

    .. ipython:: python

       rfr_fix2 = RFRFixing(
           accrual_start=dt(2025, 1, 9),
           accrual_end=dt(2025, 1, 21),
           identifier="SOFR_1B",
           spread_compound_method=SpreadCompoundMethod.NoneSimple,
           fixing_method=FloatFixingMethod.RFRPaymentDelay,
           method_param=0,
           float_spread=0.0,
           rate_index=FloatRateIndex(frequency="1B", series="usd_rfr")
       )
       rfr_fix2.value
       rfr_fix2.populated

    .. ipython:: python
       :suppress:

       fixings.pop("SOFR_1B")

    """

    _populated: Series[DualTypes]  # type: ignore[type-var]
    _dates_obs: list[datetime] | None
    _dates_dcf: list[datetime] | None
    _float_spread: DualTypes
    _fixing_index: FloatRateIndex
    _accrual_start: datetime
    _accrual_end: datetime
    _fixing_method: FloatFixingMethod
    _spread_compound_method: SpreadCompoundMethod
    _method_param: int

    def __init__(
        self,
        *,
        rate_index: FloatRateIndex,
        accrual_start: datetime,
        accrual_end: datetime,
        fixing_method: FloatFixingMethod,
        method_param: int,
        spread_compound_method: SpreadCompoundMethod,
        float_spread: DualTypes,
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ):
        self._identifier = identifier if isinstance(identifier, NoInput) else identifier.upper()
        self._value = value
        self._state = 0

        self._float_spread = float_spread
        self._spread_compound_method = spread_compound_method
        self._rate_index = rate_index
        self._value = value
        self._accrual_start = accrual_start
        self._accrual_end = accrual_end
        self._fixing_method = fixing_method
        self._method_param = method_param
        self._populated = Series(index=[], data=[], dtype=float)

    @property
    def fixing_method(self) -> FloatFixingMethod:
        """The :class:`FloatFixingMethod` object used to combine multiple RFR fixings."""
        return self._fixing_method

    @property
    def method_param(self) -> int:
        """A parameter required by the ``fixing_method``."""
        return self._method_param

    @property
    def float_spread(self) -> DualTypes:
        """The spread value incorporated into the fixing calculation using the compound method."""
        return self._float_spread

    @property
    def spread_compound_method(self) -> SpreadCompoundMethod:
        """A :class:`SpreadCompoundMethod` object used define the calculation of the addition of the
        ``float_spread``."""
        return self._spread_compound_method

    @property
    def accrual_start(self) -> datetime:
        """The accrual start date for the underlying float rate period."""
        return self._accrual_start

    @property
    def accrual_end(self) -> datetime:
        """The accrual end date for the underlying float rate period."""
        return self._accrual_end

    @property
    def value(self) -> DualTypes_:
        if not isinstance(self._value, NoInput):
            return self._value
        else:
            if isinstance(self._identifier, NoInput):
                return NoInput(0)
            else:
                state, timeseries, bounds = fixings.__getitem__(self._identifier)
                if state == self._state:
                    return NoInput(0)
                else:
                    self._state = state
                    v = self._lookup_and_calculate(timeseries, bounds)
                    self._value = v
                    return v

    @property
    def populated(self) -> Series[DualTypes]:  # type: ignore[type-var]
        """The looked up fixings as part of the calculation after a ``value`` calculation."""
        return self._populated

    def _lookup_and_calculate(
        self,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        bounds: tuple[datetime, datetime] | None,
    ) -> DualTypes_:
        value, populated = self._lookup(
            timeseries=timeseries,
            fixing_method=self.fixing_method,
            method_param=self.method_param,
            dates_obs=self.dates_obs,
            dcfs_dcf=self.dcfs_dcf,
            float_spread=self.float_spread,
            spread_compound_method=self.spread_compound_method,
        )
        self._populated = populated
        return value

    @classmethod
    def _lookup(
        cls,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        # bounds: tuple[datetime, datetime] | None,
        # accrual_start: datetime,
        # accrual_end: datetime,
        fixing_method: FloatFixingMethod,
        method_param: int,
        dates_obs: Arr1dObj,
        # dates_dcf: list[datetime] | None,
        # dcfs_obs: Arr1dF64,
        dcfs_dcf: Arr1dF64,
        float_spread: DualTypes,
        spread_compound_method: SpreadCompoundMethod,
    ) -> tuple[DualTypes_, Series[DualTypes]]:  # type: ignore[type-var]
        fixing_rates: Series[DualTypes] = Series(index=dates_obs[:-1], data=np.nan, dtype=object)  # type: ignore[type-var, assignment]
        # populate Series with values
        fixing_rates, populated, unpopulated = (
            _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
                fixing_rates=fixing_rates,
                rate_fixings=timeseries,
            )
        )
        if len(unpopulated) > 0:
            return NoInput(0), populated
        else:
            result = _RFRRate._inefficient_calculation(
                fixing_rates=fixing_rates,
                fixing_dcfs=dcfs_dcf,
                fixing_method=fixing_method,
                method_param=method_param,
                spread_compound_method=spread_compound_method,
                float_spread=float_spread,
            )
            if isinstance(result, Err):
                result.unwrap()  # will raise
            return result.unwrap(), populated

    @property
    def rate_index(self) -> FloatRateIndex:
        """The :class:`FloatRateIndex` defining the parameters of the RFR interest rate index."""
        return self._rate_index

    @cached_property
    def dates_obs(self) -> Arr1dObj:
        """A sequence of dates defining the individual **observation** rates for the period."""
        start, end = self.bounds[0]
        return np.array(self.rate_index.calendar.bus_date_range(start, end))

    @cached_property
    def dates_dcf(self) -> Arr1dObj:
        """A sequence of dates defining the individual **DCF** dates for the period."""
        start, end = self.bounds[1]
        return np.array(self.rate_index.calendar.bus_date_range(start, end))

    @cached_property
    def dcfs_obs(self) -> Arr1dF64:
        """A sequence of floats defining the individual **DCF** values associated with
        the **observation** dates."""
        return _RFRRate._get_dcf_values(
            dcf_dates=self.dates_obs,
            fixing_convention=self.rate_index.convention,
            fixing_calendar=self.rate_index.calendar,
        )

    @cached_property
    def dcfs_dcf(self) -> Arr1dF64:
        """A sequence of floats defining the individual **DCF** values associated with
        the **DCF** dates."""
        return _RFRRate._get_dcf_values(
            dcf_dates=self.dates_dcf,
            fixing_convention=self.rate_index.convention,
            fixing_calendar=self.rate_index.calendar,
        )

    @cached_property
    def bounds(self) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
        """The fixing method adjusted start and end dates for the **observation** dates and
        the **dcf** dates."""
        return self._get_date_bounds(
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            fixing_method=self.fixing_method,
            method_param=self.method_param,
            fixing_calendar=self.rate_index.calendar,
        )

    @staticmethod
    def _get_date_bounds(
        accrual_start: datetime,
        accrual_end: datetime,
        fixing_method: FloatFixingMethod,
        method_param: int,
        fixing_calendar: CalTypes,
    ) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
        """
        For each different RFR fixing method adjust the start and end date of the associated
        period to return adjusted start and end dates for the fixing set as well as the
        DCF set.

        For all methods except 'lookback', these dates will align with each other.
        For 'lookback' the observed RFRs are applied over different DCFs that do not naturally
        align.
        """
        # Depending upon method get the observation dates and dcf dates
        if fixing_method in [
            FloatFixingMethod.RFRPaymentDelay,
            FloatFixingMethod.RFRPaymentDelayAverage,
            FloatFixingMethod.RFRLockout,
            FloatFixingMethod.RFRLockoutAverage,
        ]:
            start_obs, end_obs = accrual_start, accrual_end
            start_dcf, end_dcf = accrual_start, accrual_end
        elif fixing_method in [
            FloatFixingMethod.RFRObservationShift,
            FloatFixingMethod.RFRObservationShiftAverage,
        ]:
            start_obs = fixing_calendar.lag_bus_days(accrual_start, -method_param, settlement=False)
            end_obs = fixing_calendar.lag_bus_days(accrual_end, -method_param, settlement=False)
            start_dcf, end_dcf = start_obs, end_obs
        else:
            # fixing_method in [
            #    FloatFixingMethod.RFRLookback,
            #    FloatFixingMethod.RFRLookbackAverage,
            # ]:
            start_obs = fixing_calendar.lag_bus_days(accrual_start, -method_param, settlement=False)
            end_obs = fixing_calendar.lag_bus_days(accrual_end, -method_param, settlement=False)
            start_dcf, end_dcf = accrual_start, accrual_end

        return (start_obs, end_obs), (start_dcf, end_dcf)
