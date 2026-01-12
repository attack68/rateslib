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
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from functools import cached_property
from math import prod
from typing import TYPE_CHECKING

import numpy as np
import rateslib.errors as err
from pandas import Series, isna
from rateslib import defaults, fixings
from rateslib.curves.curves import _BaseCurve, _index_value_from_series_no_curve
from rateslib.curves.interpolation import index_left
from rateslib.curves.utils import _CurveType
from rateslib.data.loader import FixingMissingForecasterError, FixingRangeError
from rateslib.dual import Dual, Dual2, Variable
from rateslib.enums.generics import Err, NoInput, Ok, _drb, _validate_obj_not_no_input
from rateslib.enums.parameters import (
    FloatFixingMethod,
    SpreadCompoundMethod,
    _get_float_fixing_method,
    _get_index_method,
    _get_spread_compound_method,
)
from rateslib.rs import Adjuster
from rateslib.scheduling.adjuster import _get_adjuster
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.convention import Convention, _get_convention
from rateslib.scheduling.dcfs import dcf
from rateslib.scheduling.frequency import _get_frequency, _get_tenor_from_frequency, add_tenor
from rateslib.utils.calendars import _get_first_bus_day

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        Any,
        Arr1dF64,
        Arr1dObj,
        CalInput,
        CalTypes,
        CurveOption_,
        DualTypes,
        DualTypes_,
        Frequency,
        FXForwards,
        FXForwards_,
        FXIndex_,
        IndexMethod,
        LegFixings,
        PeriodFixings,
        Result,
        _BaseCurve_,
        bool_,
        datetime_,
        int_,
        str_,
    )


class _BaseFixing(metaclass=ABCMeta):
    """
    Abstract base class for core financial fixing implementation.

    Parameters
    ----------
    date: datetime
        The date of relevance for the financial fixing, e.g. the publication date for an
        *IBORFixing* or the reference date for an *IndexFixing*.
    value: float, Dual, Dual2, Variable, optional
        The initial value for the fixing to adopt. Most commonly this is not given and it is
        determined from a timeseries.
    identifier: str, optional
        The string name of the timeseries to be loaded by the *Fixings* object.
    """

    _identifier: str_
    _value: DualTypes_
    _state: int
    _date: datetime

    def __init__(
        self,
        *,
        date: datetime,
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ) -> None:
        self._identifier = identifier if isinstance(identifier, NoInput) else identifier.upper()
        self._value = value
        self._state = 0
        self._date = date

    def reset(self, state: int_ = NoInput(0)) -> None:
        """
        Sets the ``value`` attribute to :class:`~rateslib.enums.generics.NoInput`, which allows it
        to be redetermined from a timeseries.

        .. rubric:: Examples

        .. ipython:: python
           :suppress:

           from rateslib import fixings, dt, NoInput, FXFixing
           from pandas import Series

        .. ipython:: python

           fx_fixing1 = FXFixing(publication=dt(2021, 1, 1), fx_index="eurusd", identifier="A")
           fx_fixing2 = FXFixing(publication=dt(2021, 1, 1), fx_index="gbpusd", identifier="B")
           fixings.add("A_eurusd", Series(index=[dt(2021, 1, 1)], data=[1.1]), state=100)
           fixings.add("B_gbpusd", Series(index=[dt(2021, 1, 1)], data=[1.4]), state=200)

           # data is populated from the available Series
           fx_fixing1.value
           fx_fixing2.value

           # fixings are reset according to the data state
           fx_fixing1.reset(state=100)
           fx_fixing2.reset(state=100)

           # only the private data for fixing1 is removed because of its link to the data state
           fx_fixing1._value
           fx_fixing2._value

        .. role:: green

        Parameters
        ----------
        state: int, :green:`optional`
            If given only fixings whose state matches this value will be reset. If no state is
            given then the value will be reset.

        Returns
        -------
        None
        """
        if isinstance(state, NoInput) or self._state == state:
            self._value = NoInput(0)
            self._state = 0

    @property
    def value(self) -> DualTypes_:
        """
        The fixing value.

        If this value is :class:`rateslib.enums.generics.NoInput`, then each request will attempt a
        lookup from a timeseries to obtain a new fixing value.

        Once this value is determined it is restated indefinitely, unless :meth:`_BaseFixing.reset`
        is called.
        """
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
    def date(self) -> datetime:
        """The date of relevance for the fixing, e.g. the publication date of an IBORFixing."""
        return self._date

    @property
    def identifier(self) -> str_:
        """The string name of the timeseries to be loaded by the *Fixings* object."""
        return self._identifier

    @abstractmethod
    def _lookup_and_calculate(
        self,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        bounds: tuple[datetime, datetime] | None,
    ) -> DualTypes_:
        pass

    def __repr__(self) -> str:
        return f"<rl.{type(self).__name__} at {hex(id(self))}>"


class IndexFixing(_BaseFixing):
    """
    An index fixing value for settlement of indexed cashflows.

    Parameters
    ----------
    index_lag: int
        The number months by which the reference date is lagged to derive an index value.
    index_method: IndexMethod
        The method used for calculating the index value. See
        :class:`~rateslib.enums.parameters.IndexMethod`.
    date: datetime
        The date of relevance for the index fixing, which is its **reference value** date.
    value: float, Dual, Dual2, Variable, optional
        The initial value for the fixing to adopt. Most commonly this is not given and it is
        determined from a timeseries of published FX rates.
    identifier: str, optional
        The string name of the timeseries to be loaded by the *Fixings* object.

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib.data.fixings import IndexFixing
       from rateslib.enums.parameters import IndexMethod
       from rateslib import fixings, dt
       from pandas import Series

    .. ipython:: python

       fixings.add("UK-CPI", Series(index=[dt(2000, 1, 1), dt(2000, 2, 1)], data=[100, 110.0]))
       index_fix = IndexFixing(date=dt(2000, 4, 15), identifier="UK-CPI", index_lag=3, index_method=IndexMethod.Daily)
       index_fix.value

    .. ipython:: python
       :suppress:

       fixings.pop("UK-CPI")

    """  # noqa: E501

    _index_lag: int
    _index_method: IndexMethod

    def __init__(
        self,
        *,
        index_lag: int,
        index_method: IndexMethod | str,
        date: datetime,
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ) -> None:
        super().__init__(date=date, value=value, identifier=identifier)

        self._index_lag = index_lag
        self._index_method = _get_index_method(index_method)

    @property
    def index_method(self) -> IndexMethod:
        """The :class:`~rateslib.enums.parameters.IndexMethod` used for calculating the
        index value."""
        return self._index_method

    @property
    def index_lag(self) -> int:
        """The number months by which the reference date is lagged to derive an index value."""
        return self._index_lag

    def _lookup_and_calculate(
        self,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        bounds: tuple[datetime, datetime] | None,
    ) -> DualTypes_:
        return self._lookup(
            index_lag=self.index_lag,
            index_method=self.index_method,
            date=self.date,
            timeseries=timeseries,
            bounds=bounds,
        )

    @classmethod
    def _lookup(
        cls,
        index_lag: int,
        index_method: IndexMethod,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        date: datetime,
        bounds: tuple[datetime, datetime] | None = None,
    ) -> DualTypes_:
        result = _index_value_from_series_no_curve(
            index_lag=index_lag,
            index_method=index_method,
            index_fixings=timeseries,
            index_date=date,
            index_fixings_boundary=bounds,
        )
        if isinstance(result, Err):
            if isinstance(result._exception, FixingRangeError):
                return NoInput(0)
            result.unwrap()
        else:
            return result.unwrap()


class FXIndex:
    """
    Define the parameters of a specific FX pair and fixing index.

    This object acts as a container to store market conventions for different FX pairs.
    This allows the determination of dates under different methodologies, e.g. ISDA MTM fixings or
    spot settlement dates.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.data.fixings import FXIndex

    .. ipython:: python

       fxi = FXIndex(
           pair="eurusd",
           calendar="tgt|fed",      # <- Spot FX measures settlement dates according to this calendar
           settle=2,
           isda_mtm_calendar="ldn", # <- MTM XCS FX fixing dates are determined according to this calendar
           isda_mtm_settle=-2,
       )
       fxi.delivery(dt(2025, 7, 3))
       fxi.isda_fixing_date(dt(2025, 7, 3))

    .. role:: red

    .. role:: green

    Parameters
    ----------
    pair: str, :red:`required`
        The currency pair of the FX fixing. 6-digit iso code.
    calendar: Calendar, str, :red:`required`
        The calendar associated with the FX settlement date determination.
    settle:  Adjuster, int, str :green:`optional (set by 'defaults')`
        The delivery lag applied to any FX quotation to adjust 'today' to a delivery date, under
        the given ``calendar``. If int is assumed to be settleable business days.
    isda_mtm_calendar: Calendar, str, :green:`optional`
        The calendar associated with the MTM fixing date determination.
    isda_mtm_settle: Adjuster, str, int_, :green:`optional`,
        The adjustment applied to determine the MTM fixing date.
    allow_cross: bool, :green:`optional (set as True)`
        This allows sub-division of the fixing into its *majors* as defined by WMR
        benchmark methodology. For an example of using a *cross* see the documentation for
        an :class:`FXFixing`.

    """  # noqa: E501

    def __init__(
        self,
        pair: str,
        calendar: CalTypes | str,
        settle: Adjuster | str | int,
        isda_mtm_calendar: CalInput = NoInput(0),
        isda_mtm_settle: Adjuster | str | int_ = NoInput(0),
        allow_cross: bool_ = NoInput(0),
    ) -> None:
        self._pair: str = pair.lower()
        self._calendar: CalTypes = get_calendar(calendar)
        self._settle: Adjuster = _get_adjuster(settle)
        self._allow_cross: bool = _drb(True, allow_cross)

        if isinstance(isda_mtm_calendar, NoInput):
            self._isda_mtm_calendar: CalTypes | NoInput = NoInput(0)
        else:
            self._isda_mtm_calendar = get_calendar(isda_mtm_calendar)

        if isinstance(isda_mtm_settle, NoInput):
            self._isda_mtm_settle: Adjuster | NoInput = NoInput(0)
        else:
            self._isda_mtm_settle = _get_adjuster(isda_mtm_settle)

    def __repr__(self) -> str:
        return f"<rl.FXIndex:{self.pair} at {id(self)}>"

    @property
    def pair(self) -> str:
        """The currency pair of the FX fixing."""
        return self._pair

    @property
    def calendar(self) -> CalTypes:
        """The calendar associated with the settlement delivery date determination."""
        return self._calendar

    @property
    def settle(self) -> Adjuster:
        """
        The :class:`~rateslib.scheduling.Adjuster` associated with determining
        the settlement delivery date.
        """
        return self._settle

    @property
    def isda_mtm_calendar(self) -> CalTypes | NoInput:
        """The calendar associated with the MTM fixing date determination."""
        return self._isda_mtm_calendar

    @property
    def isda_mtm_settle(self) -> Adjuster | NoInput:
        """
        The :class:`~rateslib.scheduling.Adjuster` associated with the MTM fixing
        date determination.
        """
        return self._isda_mtm_settle

    def isda_fixing_date(self, delivery: datetime) -> datetime:
        """
        Return the MTM FX fixing date under ISDA conventions.

        Parameters
        ----------
        delivery: datetime
            The delivery date of the notional exchange.

        Returns
        -------
        datetime

        Notes
        -----
        If ``isda`` attributes are not fully qualified on the object then uses the ``reverse``
        method to reverse engineer the FX quotation date as a proxy.
        """
        if isinstance(self.isda_mtm_calendar, NoInput) or isinstance(self.isda_mtm_settle, NoInput):
            # Fallback method for determining fixing date when ISDA fixing details not available.
            # This may be due to instruments that only use for non-deliverability as a feature
            # but do not technically have a published fixing, i.e. a physically settled
            # FXForward or an FXOption.
            # In these cases do the best to estimate a respectable date.
            alternatives: list[datetime] = []
            counter: int = 0
            while len(alternatives) == 0:
                alternatives = self.publications(delivery + timedelta(days=counter))
                counter += 1
            return _get_first_bus_day(alternatives, self.calendar)
        else:
            return self.isda_mtm_settle.adjust(delivery, self.isda_mtm_calendar)

    def delivery(self, date: datetime) -> datetime:
        """
        Return the settlement delivery date associated with the publication date.

        Parameters
        ----------
        date: datetime
            The publication date of the quotation.

        Returns
        -------
        datetime
        """
        return self.settle.adjust(date, self.calendar)

    def publications(self, delivery: datetime) -> list[datetime]:
        """
        Return the potential publication dates that result in a given settlement delivery date.

        Parameters
        ----------
        delivery: datetime
            The settlement delivery date of the publication.

        Returns
        -------
        list[datetime]
        """
        return self.settle.reverse(delivery, self.calendar)

    @property
    def allow_cross(self) -> bool:
        """Whether to allow FXFixings which sub-divide into USD or EUR crosses."""
        return self._allow_cross


class _FXFixingMajor(_BaseFixing):
    """
    An FX fixing value for cross currency settlement.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.data.fixings import _FXFixingMajor, FXIndex
       from rateslib import fixings, dt
       from pandas import Series

    .. ipython:: python

       fixings.add("Custom_CADSEK", Series(index=[dt(1999, 12, 29)], data=[8.7]))
       fxfix = _FXFixingMajor(
           delivery=dt(2000, 1, 4),
           fx_index=FXIndex(
               pair="cadsek",
               calendar="tro,stk|fed",
               settle=2,
               isda_mtm_calendar="tro,stk,ldn,nyc",
               isda_mtm_settle=-2,
           ),
           identifier="Custom"
       )
       fxfix.publication  #  <--  derived from isda attributes
       fxfix.value        #  <--  should be 8.7

    .. ipython:: python
       :suppress:

       fixings.pop("Custom_CADSEK")

    .. role:: red

    .. role:: green

    Parameters
    ----------
    fx_index: FXIndex, str, :red:`required`
        The :class:`~rateslib.data.fixings.FXIndex` defining the FX pair and its conventions.
    publication: datetime, :green:`optional`
        The publication date of the fixing. If not given, must provide ``delivery`` in order to
        derive the *publication date*.
    delivery: datetime, :green:`optional`
        The settlement delivery date of the cashflow. Can be used to derive the *publication date*.
        If not given is derived from the ``publication``.
    value: float, Dual, Dual2, Variable, optional
        The initial value for the fixing to adopt. Most commonly this is not given and it is
        determined from a timeseries of published FX rates.
    identifier: str, optional
        The string name of the series to be loaded by the *Fixings* object. Will be
        appended with "_{pair}" to derive the full timeseries key.

    Notes
    ------
    The *FXFixingMajor* is a class designed to lookup and return FX fixings directly from a
    Series in either the FX pair directly, or its inverse. This function depends upon what is
    populated to the datastore. That is, if *'GBPMXN'* is an available dataseries then *'MXNGBP'*
    would also be calculable as the inverse of *'GBPMXN'*.

    When forecasting the fixing from an :class:`~rateslib.fx.FXForwards` object, the rate pair
    will be looked up directly according to the ``delivery`` date.

    The use of the name **major**, does not imply that only *FX majors* can be used by this class.
    I.e. that it is only suitable for *'EURUSD'* and *'EURSEK'*, for example. Rather, the name
    *major* implies that this object treats the given FX pair as a major and does not perform any
    type of **cross**. This is, in fact, a sub-component of the more featureful
    :class:`~rateslib.data.fixings.FXFixing` class which adheres to the ``allow_cross`` argument
    on the :class:`~rateslib.data.fixings.FXIndex` in order to automatically handle different
    types of required behaviour.

    """

    def __init__(
        self,
        fx_index: FXIndex | str,
        publication: datetime_ = NoInput(0),
        delivery: datetime_ = NoInput(0),
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ) -> None:
        self._fx_index = _get_fx_index(fx_index)
        del fx_index

        if isinstance(delivery, NoInput) and isinstance(publication, NoInput):
            raise ValueError(
                "At least one date; a `delivery` or `publication` is required to derive the "
                "`date` used for the FX fixing."
            )
        elif isinstance(publication, NoInput) and isinstance(delivery, datetime):
            # then derive it under conventions
            date_ = self.fx_index.isda_fixing_date(delivery)
            self._delivery = delivery
            self._publication = date_
        elif isinstance(publication, datetime) and isinstance(delivery, NoInput):
            date_ = publication
            self._publication = date_
            self._delivery = self.fx_index.delivery(date_)
        elif isinstance(publication, datetime) and isinstance(delivery, datetime):
            date_ = publication
            self._publication = date_
            self._delivery = delivery
        else:
            raise TypeError(  # pragma: no cover
                "`delivery` and `publication` given as incorrect types.\n"
                f"Got {type(delivery).__name__} and {type(publication).__name__}."
            )

        super().__init__(date=date_, value=value, identifier=identifier)

    @property
    def fx_index(self) -> FXIndex:
        """The :class:`FXIndex` for the FX fixing."""
        return self._fx_index

    def _value_from_possible_inversion(self, identifier: str) -> DualTypes_:
        direct, inverted = self.pair, f"{self.pair[3:6]}{self.pair[0:3]}"
        try:
            state, timeseries, bounds = fixings.__getitem__(identifier + "_" + direct)
            exponent = 1.0
        except ValueError as e:
            try:
                state, timeseries, bounds = fixings.__getitem__(identifier + "_" + inverted)
                exponent = -1.0
            except ValueError:
                raise e

        if state == self._state:
            return NoInput(0)
        else:
            self._state = state
            v = self._lookup_and_calculate(timeseries, bounds)
            if isinstance(v, NoInput):
                return NoInput(0)
            self._value = v**exponent
            return self._value

    @property
    def publication(self) -> datetime:
        """The publication date of the fixing as specified directly, or implied from
        the :class:`~rateslib.data.fixings.FXIndex`."""
        return self._publication

    @property
    def delivery(self) -> datetime:
        """The settlement delivery date of the fixing as specified directly, or implied
        from the :class:`~rateslib.data.fixings.FXIndex`."""
        return self._delivery

    @property
    def value(self) -> DualTypes_:
        if not isinstance(self._value, NoInput):
            return self._value
        else:
            if isinstance(self._identifier, NoInput):
                return NoInput(0)
            else:
                return self._value_from_possible_inversion(identifier=self._identifier)

    def _lookup_and_calculate(
        self, timeseries: Series, bounds: tuple[datetime, datetime] | None
    ) -> DualTypes_:
        return self._lookup(timeseries=timeseries, date=self.date, bounds=bounds)

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

    @property
    def pair(self) -> str:
        """The currency pair related to the FX fixing."""
        return self.fx_index.pair

    def value_or_forecast(self, fx: FXForwards_) -> DualTypes:
        """
        Return the determined value of the fixing or forecast it if not available.

        Parameters
        ----------
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object to forecast the forward FX rate.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        if isinstance(self.value, NoInput):
            fx_: FXForwards = _validate_obj_not_no_input(fx, "FXForwards")
            return fx_.rate(pair=self.pair, settlement=self.delivery)
        else:
            return self.value

    def try_value_or_forecast(self, fx: FXForwards_) -> Result[DualTypes]:
        """
        Return the determined value of the fixing or forecast it if not available.

        Parameters
        ----------
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object to forecast the forward FX rate.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        if isinstance(self.value, NoInput):
            if isinstance(fx, NoInput):
                return Err(ValueError("Must provide `fx` argument to forecast FXFixing."))
            else:
                return Ok(fx.rate(pair=self.pair, settlement=self.delivery))
        else:
            return Ok(self.value)

    def __repr__(self) -> str:
        return f"<rl.FXFixingMajor:{self.pair} at {hex(id(self))}>"


def _clone_isda_mtm(pair: FXIndex | str, isda_index: FXIndex) -> FXIndex:
    """
    Attempt to lookup the conventions of pair, but maintain the original ISDA index conventions
    from the given isda_index
    """
    if isinstance(pair, str):
        try:
            fx_index = _get_fx_index(pair)  # lookup the conventions from STATIC directly
        except ValueError:
            fx_index = FXIndex(
                pair=pair,
                settle=isda_index.settle,
                calendar=isda_index.calendar,
            )
    else:
        fx_index = pair

    return FXIndex(
        pair=fx_index.pair,
        settle=fx_index.settle,
        calendar=fx_index.calendar,
        isda_mtm_settle=isda_index.isda_mtm_settle,
        isda_mtm_calendar=isda_index.isda_mtm_calendar,
    )


def _fx_index_set_cross(pair: FXIndex, allow_cross: bool) -> FXIndex:
    return FXIndex(
        pair=pair.pair,
        settle=pair.settle,
        calendar=pair.calendar,
        isda_mtm_settle=pair.isda_mtm_settle,
        isda_mtm_calendar=pair.isda_mtm_calendar,
        allow_cross=allow_cross,
    )


class _UnitFixing(_BaseFixing):
    """
    A :class:`~rateslib.data.fixings._BaseFixing` permanently adopting value 1.0.
    Used as a placeholder.
    """

    def __init__(
        self, *, date: datetime, value: DualTypes_ = NoInput(0), identifier: str_ = NoInput(0)
    ) -> None:
        self._value = 1.0
        self._state = 0
        self._date = date
        self._identifier = identifier

    @property
    def value(self) -> DualTypes_:
        """Returns 1.0."""
        return 1.0

    def value_or_forecast(self, *args: Any, **kwargs: Any) -> DualTypes:
        """Returns 1.0."""
        return 1.0

    def __repr__(self) -> str:
        return f"<rl._UnitFixing at {id(self)}>"

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Does nothing."""
        pass

    def _lookup_and_calculate(self, *args: Any, **kwargs: Any) -> DualTypes_:
        return 1.0


_WMR_EUR_BASE = ["czk", "dkk", "huf", "nok", "pln", "ron", "sek"]
_WMR_USD_INVERTED = ["gbp", "eur", "aud", "nzd", "iep", "bwp", "sbd", "top", "wst", "xeu"]


class _WMRClassification(Enum):
    """
    WMR FX Benchmarks classification. Either the currency is USD or EUR or it is a third currency
    whose base is measured versus USD or EUR
    """

    USD = 0
    EUR = 1
    BASE_USD = 2
    BASE_EUR = 3

    @classmethod
    def classify(cls, value: str) -> _WMRClassification:
        if value == "usd":
            return _WMRClassification.USD
        elif value == "eur":
            return _WMRClassification.EUR
        elif value in _WMR_EUR_BASE:
            return _WMRClassification.BASE_EUR
        else:
            return _WMRClassification.BASE_USD


class FXFixing(_BaseFixing):
    """
    An FX fixing value for cross-currency or non-deliverable settlement.

    .. rubric:: Examples

    .. ipython:: python
       :suppress:

       from rateslib.data.fixings import FXFixing, FXIndex
       from rateslib import fixings, dt, FXForwards, FXRates, Curve
       from pandas import Series

    .. ipython:: python

       fixings.add("WMR_10AMTYO_USDJPY", Series(index=[dt(1999, 12, 29)], data=[155.00]))
       fixings.add("WMR_10AMTYO_AUDUSD", Series(index=[dt(1999, 12, 29)], data=[1.260]))
       fxfix = FXFixing(
           delivery=dt(2000, 1, 4),
           fx_index=FXIndex(
               pair="audjpy",
               calendar="syd,tyo|fed",
               settle=2,
               isda_mtm_calendar="syd,tyo,ldn",
               isda_mtm_settle=-2,
               allow_cross=True,
           ),
           identifier="WMR_10AMTYO"
       )
       fxfix.publication  #  <--  derived from isda attributes
       fxfix.value        #  <--  should be from the cross 1.26 * 155 = 195.3

    .. ipython:: python
       :suppress:

       fixings.pop("WMR_10AMTYO_USDJPY")
       fixings.pop("WMR_10AMTYO_AUDUSD")

    .. role:: red

    .. role:: green

    Parameters
    ----------
    fx_index: FXIndex, str, :red:`required`
        The :class:`~rateslib.data.fixings.FXIndex` defining the FX pair and its conventions.
    publication: datetime, :green:`optional`
        The publication date of the fixing. If not given, must provide ``delivery`` in order to
        derive the *publication date*.
    delivery: datetime, :green:`optional`
        The settlement delivery date of the cashflow. Can be used to derive the *publication date*.
        If not given is derived from the ``publication``.
    value: float, Dual, Dual2, Variable, optional
        The initial value for the fixing to adopt. Most commonly this is not given and it is
        determined from a timeseries of published FX rates.
    identifier: str, optional
        The string name of the series to be loaded by the *Fixings* object. Will be
        appended with "_{pair}" to derive the full timeseries key.

    Notes
    -----
    This object is designed to systematically handle FX fixings across variety of conventions
    and is typically used for non-deliverable and MTM-XCS settlement.

    If the :class:`~rateslib.data.fixings.FXIndex` is configured to ``allow_cross`` (which is
    the general default) then it will adopt the `WMR Benchmark Methodology <https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/wmr-fx-methodology.pdf>`__
    and assume cross rates against base USD, except if the currency is one of the European
    currencies defined as having a base EUR, by that methodology.

    Suppose one transacted a *CADSEK mtm-XCS* with a CAD mtm *Leg*. The ISDA MTM fixing date could
    be defined as being 2 business days prior to the cashflow under the Stockholm, New York, London
    and Toronto calendars. Under WMR, CAD has a USD base, and SEK has a EUR base, so the
    determination of this FX Fixing will be a 3-way cross: CADUSD * USDEUR * EURSEK.

    WMR ignores market settlement convention and local calendars in the determination of the cross.
    So a cashflow due on 8th Jan '26 will determine a publication date as 5th Jan '26 (since the
    6th Jan is holiday in Stockholm). All three WMR publication will have different
    settlement (delivery dates) for the publication date on the 5th Jan:

    .. ipython:: python

       fxfix = FXFixing(
           fx_index=FXIndex(
               pair="cadsek",
               calendar="tro,stk|fed",
               settle=2,
               isda_mtm_calendar="tro,ldn,stk,nyc",
               isda_mtm_settle=-2,
               allow_cross=True,
           ),
           delivery=dt(2026, 1, 8),
       )
       fxfix.publication  #  <-- is 5th Jan determnined from the isda specications
       fxfix.fx_fixing1.delivery  #  <-- USDCAD is T+1 under "tro|fed" defined by defaults
       fxfix.fx_fixing2.delivery  #  <-- EURUSD is T+2 under "tgt|fed" defined by defaults
       fxfix.fx_fixing3.delivery  #  <-- EURSEK is T+2 under "tgt,stk|fed" defined by defaults

    This has implications towards the forecasting of these fixing values. In order to properly
    forecast the above an :class:`~rateslib.fx.FXForwards` with all four currencies is required.

    .. ipython:: python
       :suppress:

       sek = Curve({dt(2026, 1, 1): 1.0, dt(2027, 1, 1): 0.98})
       eur = Curve({dt(2026, 1, 1): 1.0, dt(2027, 1, 1): 0.981})
       cad = Curve({dt(2026, 1, 1): 1.0, dt(2027, 1, 1): 0.97})
       usd = Curve({dt(2026, 1, 1): 1.0, dt(2027, 1, 1): 0.965})

       fxf = FXForwards(
           fx_rates=[
               FXRates({"usdcad": 1.38}, settlement=dt(2026, 1, 2)),
               FXRates({"eurusd": 1.165, "eursek": 10.75}, settlement=dt(2026, 1, 3))
           ],
           fx_curves={
               "seksek": sek, "sekusd": sek, "eureur": eur, "eurusd": eur, "cadcad": cad, "cadusd": cad, "usdusd": usd
           }
       )

    .. ipython:: python

       fxfix.value_or_forecast(fx=fxf)  #  <-  FXForwards:usd,cad,eur,sek
       fxf.rate("cadusd", dt(2026, 1, 6)) * fxf.rate("usdeur", dt(2026, 1, 7)) * fxf.rate("eursek", dt(2026, 1, 8))

    Note that this is different to the **actual** *CADSEK* forecast FX rate and this is due to
    those milaligned crosses and calendars.

    .. ipython:: python

       fxfix = FXFixing(
           fx_index=FXIndex(
               pair="cadsek",
               calendar="tro,stk|fed",
               settle=2,
               isda_mtm_calendar="tro,ldn,stk,nyc",
               isda_mtm_settle=-2,
               allow_cross=False,      #  <-  Everything the same except no crossing allowed
           ),
           delivery=dt(2026, 1, 8),
       )
       fxfix.value_or_forecast(fx=fxf)
       fxf.rate("cadsek", dt(2026, 1, 8))

    """  # noqa: E501

    def __init__(
        self,
        fx_index: FXIndex | str,
        publication: datetime_ = NoInput(0),
        delivery: datetime_ = NoInput(0),
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ) -> None:
        self._fx_index = _get_fx_index(fx_index)
        del fx_index

        if isinstance(delivery, NoInput) and isinstance(publication, NoInput):
            raise ValueError(
                "At least one date; a `delivery` or `publication` is required to derive the "
                "`date` used for the FX fixing."
            )
        elif isinstance(publication, NoInput) and isinstance(delivery, datetime):
            # then derive it under conventions
            date_ = self.fx_index.isda_fixing_date(delivery)
            self._delivery = delivery
            self._publication = date_
        elif isinstance(publication, datetime) and isinstance(delivery, NoInput):
            date_ = publication
            self._publication = date_
            self._delivery = self.fx_index.delivery(date_)
        elif isinstance(publication, datetime) and isinstance(delivery, datetime):
            date_ = publication
            self._publication = date_
            self._delivery = delivery
        else:
            raise TypeError(  # pragma: no cover
                "`delivery` and `publication` given as incorrect types.\n"
                f"Got {type(delivery).__name__} and {type(publication).__name__}."
            )

        self._identifier = identifier if isinstance(identifier, NoInput) else identifier.upper()
        self._value = value
        self._date = date_

        if not self.allow_cross:
            self._fx_fixing1: _FXFixingMajor = _FXFixingMajor(
                fx_index=self.fx_index,
                publication=self.publication,
                delivery=self.delivery,
                value=value,
                identifier=identifier,
            )
            self._fx_fixing2: _FXFixingMajor | _UnitFixing = _UnitFixing(
                date=self.publication, identifier=identifier
            )
            self._fx_fixing3: _FXFixingMajor | _UnitFixing = _UnitFixing(
                date=self.publication, identifier=identifier
            )
        else:
            ccy1, ccy2 = self.fx_index.pair[:3], self.fx_index.pair[3:]

            match (
                _WMRClassification.classify(self.pair[:3]),
                _WMRClassification.classify(self.pair[3:]),
            ):
                case (_WMRClassification.USD, _WMRClassification.USD):
                    raise ValueError("An FXFixing between 'usd' and 'usd' is not valid.")
                case (_WMRClassification.EUR, _WMRClassification.EUR):
                    raise ValueError("An FXFixing between 'eur' and 'eur' is not valid.")
                case (
                    (_WMRClassification.USD, _WMRClassification.EUR)
                    | (_WMRClassification.EUR, _WMRClassification.USD)
                    | (_WMRClassification.USD, _WMRClassification.BASE_USD)
                    | (_WMRClassification.BASE_USD, _WMRClassification.USD)
                    | (_WMRClassification.EUR, _WMRClassification.BASE_EUR)
                    | (_WMRClassification.BASE_EUR, _WMRClassification.EUR)
                ):
                    # then the pair is a direct major determined by WMR
                    self._fx_fixing1 = _FXFixingMajor(
                        fx_index=self.fx_index,
                        publication=self.publication,
                        delivery=self.delivery,
                        identifier=identifier,
                    )
                    self._fx_fixing2 = _UnitFixing(date=self.publication, identifier=identifier)
                    self._fx_fixing3 = _UnitFixing(date=self.publication, identifier=identifier)
                case (
                    (_WMRClassification.USD, _WMRClassification.BASE_EUR)
                    | (_WMRClassification.BASE_EUR, _WMRClassification.USD)
                    | (_WMRClassification.BASE_EUR, _WMRClassification.BASE_EUR)
                ):
                    # then must be a 2 pair cross involving EUR
                    self._fx_fixing1 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"{ccy1}eur", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing2 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"eur{ccy2}", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing3 = _UnitFixing(date=self.publication, identifier=identifier)
                case (
                    (_WMRClassification.BASE_USD, _WMRClassification.EUR)
                    | (_WMRClassification.EUR, _WMRClassification.BASE_USD)
                    | (_WMRClassification.BASE_USD, _WMRClassification.BASE_USD)
                ):
                    # then must be a 2 pair cross involving USD
                    self._fx_fixing1 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"{ccy1}usd", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing2 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"usd{ccy2}", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing3 = _UnitFixing(date=self.publication, identifier=identifier)
                case (_WMRClassification.BASE_USD, _WMRClassification.BASE_EUR):
                    # then must be a 4 currency cross involving EUR and USD
                    self._fx_fixing1 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"{ccy1}usd", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing2 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm("usdeur", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing3 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"eur{ccy2}", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                case (_WMRClassification.BASE_EUR, _WMRClassification.BASE_USD):
                    # then must be a 4 currency cross involving EUR and USD
                    self._fx_fixing1 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"{ccy1}eur", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing2 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm("eurusd", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )
                    self._fx_fixing3 = _FXFixingMajor(
                        fx_index=_clone_isda_mtm(f"usd{ccy2}", self.fx_index),
                        publication=self.publication,
                        identifier=identifier,
                    )

    @property
    def _state(self) -> int:  # type: ignore[override]
        return hash(self.fx_fixing1._state + self.fx_fixing2._state + self.fx_fixing3._state)

    @property
    def fx_fixing1(self) -> _FXFixingMajor:
        """
        The first (or only) :class:`~rateslib.data.fixings._FXFixingMajor` required by the fixing.
        """
        return self._fx_fixing1

    @property
    def fx_fixing2(self) -> _FXFixingMajor | _UnitFixing:
        """
        The second :class:`~rateslib.data.fixings._FXFixingMajor` required by the fixing if crossed.
        """
        return self._fx_fixing2

    @property
    def fx_fixing3(self) -> _FXFixingMajor | _UnitFixing:
        """
        The third :class:`~rateslib.data.fixings._FXFixingMajor` required by the fixing if crossed.
        """
        return self._fx_fixing3

    @property
    def allow_cross(self) -> bool:
        """Whether the fixing uses WMR base currencies and majors or directly looks up the given
        pair."""
        return self.fx_index.allow_cross

    @property
    def fx_index(self) -> FXIndex:
        """The :class:`FXIndex` for the FX fixing."""
        return self._fx_index

    @property
    def publication(self) -> datetime:
        """The publication date of the fixing as specified directly, or implied from
        the :class:`~rateslib.data.fixings.FXIndex`."""
        return self._publication

    @property
    def delivery(self) -> datetime:
        """The settlement delivery date of the fixing as specified directly, or implied
        from the :class:`~rateslib.data.fixings.FXIndex`."""
        return self._delivery

    @property
    def value(self) -> DualTypes_:
        if not isinstance(self._value, NoInput):
            return self._value
        else:
            if (
                isinstance(self.fx_fixing1.value, NoInput)
                or isinstance(self.fx_fixing2.value, NoInput)
                or isinstance(self.fx_fixing3.value, NoInput)
            ):
                return NoInput(0)
            else:
                self._value = self.fx_fixing1.value * self.fx_fixing2.value * self.fx_fixing3.value
                return self._value

    @property
    def pair(self) -> str:
        """The currency pair related to the FX fixing."""
        return self.fx_index.pair

    def value_or_forecast(self, fx: FXForwards_) -> DualTypes:
        """
        Return the determined value of the fixing or forecast it if not available.

        Parameters
        ----------
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object to forecast the forward FX rate.

        Returns
        -------
        float, Dual, Dual2, Variable
        """
        if isinstance(self.value, NoInput):
            fx_: FXForwards = _validate_obj_not_no_input(fx, "FXForwards")
            f1 = self.fx_fixing1.value_or_forecast(fx=fx_)
            f2 = self.fx_fixing2.value_or_forecast(fx=fx_)
            f3 = self.fx_fixing3.value_or_forecast(fx=fx_)
            return f1 * f2 * f3
        else:
            return self.value

    def try_value_or_forecast(self, fx: FXForwards_) -> Result[DualTypes]:
        """
        Return the determined value of the fixing or forecast it if not available.

        Parameters
        ----------
        fx: FXForwards, optional
            The :class:`~rateslib.fx.FXForwards` object to forecast the forward FX rate.

        Returns
        -------
        Result[float, Dual, Dual2, Variable]
        """
        if isinstance(self.value, NoInput):
            if isinstance(fx, NoInput):
                return Err(ValueError("Must provide `fx` argument to forecast FXFixing."))
            else:
                return Ok(fx.rate(pair=self.pair, settlement=self.delivery))
        else:
            return Ok(self.value)

    def _lookup_and_calculate(
        self, timeseries: Series, bounds: tuple[datetime, datetime] | None
    ) -> DualTypes_:
        raise NotImplementedError("FXFixing does not support lookup and calculation.")

    @classmethod
    def _lookup(
        cls,
        timeseries: Series[DualTypes],  # type: ignore[type-var]
        date: datetime,
        bounds: tuple[datetime, datetime] | None = None,
    ) -> DualTypes_:
        raise NotImplementedError("FXFixing does not support lookup.")
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

    def __repr__(self) -> str:
        _1 = self.fx_fixing1.pair
        _2 = ("/" + self.fx_fixing2.pair) if not isinstance(self.fx_fixing2, _UnitFixing) else ""
        _3 = ("/" + self.fx_fixing3.pair) if not isinstance(self.fx_fixing3, _UnitFixing) else ""
        return f"<rl.FXFixing:{_1}{_2}{_3} at {hex(id(self))}>"

    def reset(self, state: int_ = NoInput(0)) -> None:
        if (
            isinstance(state, NoInput)
            or self.fx_fixing1._state == state
            or self.fx_fixing2._state == state
            or self.fx_fixing3._state == state
        ):
            self._value = NoInput(0)
        self._fx_fixing1.reset(state)
        self._fx_fixing2.reset(state)
        self._fx_fixing3.reset(state)


# class FXFixing_Legacy(_BaseFixing):
#     """
#     An FX fixing value for cross currency settlement.
#
#     .. role:: red
#
#     .. role:: green
#
#     Parameters
#     ----------
#     fx_index: FXIndex, str, :red:`required`
#         The :class:`~rateslib.data.fixings.FXIndex` defining the FX pair and its conventions.
#     publication: datetime, :green:`optional`
#         The publication date of the fixing. If not given, must provide ``delivery`` in order to
#         derive the *publication date*.
#     delivery: datetime, :green:`optional`
#         The settlement delivery date of the cashflow. Can be used to derive the
#         *publication date*. If not given is derived from the ``publication``.
#     value: float, Dual, Dual2, Variable, optional
#         The initial value for the fixing to adopt. Most commonly this is not given and it is
#         determined from a timeseries of published FX rates.
#     identifier: str, optional
#         The string name of the series to be loaded by the *Fixings* object. Will be
#         appended with "_{pair}" to derive the full timeseries key.
#
#     Examples
#     --------
#
#     .. ipython:: python
#        :suppress:
#
#        from rateslib.data.fixings import FXFixing, FXIndex
#        from rateslib import fixings, dt
#        from pandas import Series
#
#     .. ipython:: python
#
#        fixings.add("WMRPSPOT01_USDJPY", Series(index=[dt(1999, 12, 29)], data=[155.00]))
#        fixings.add("WMRPSPOT01_AUDUSD", Series(index=[dt(1999, 12, 29)], data=[1.260]))
#        fxfix = FXFixing(
#            delivery=dt(2000, 1, 4),
#            fx_index=FXIndex(
#                pair="audjpy",
#                calendar="syd,tyo|fed",
#                settle=2,
#                isda_mtm_calendar="syd,tyo,ldn",
#                isda_mtm_settle=-2,
#            ),
#            identifier="WMRPSPOT01"
#        )
#        fxfix.publication  #  <--  derived from isda attributes
#        fxfix.value  #  <--  should be 1.26 * 155 = 202.5
#
#     .. ipython:: python
#        :suppress:
#
#        fixings.pop("WMRPSPOT01_USDJPY")
#        fixings.pop("WMRPSPOT01_AUDUSD")
#
#     """
#
#     def __init__(
#         self,
#         fx_index: FXIndex | str,
#         publication: datetime_ = NoInput(0),
#         delivery: datetime_ = NoInput(0),
#         value: DualTypes_ = NoInput(0),
#         identifier: str_ = NoInput(0),
#     ) -> None:
#         fx_index_: FXIndex = _get_fx_index(fx_index)
#         del fx_index
#         if isinstance(delivery, NoInput) and isinstance(publication, NoInput):
#             raise ValueError(
#                 "At least one date; a `delivery` or `publication` is required to derive the "
#                 "`date` used for the FX fixing."
#             )
#         elif isinstance(publication, NoInput) and isinstance(delivery, datetime):
#             # then derive it under conventions
#             date_ = fx_index_.isda_fixing_date(delivery)
#             self._delivery = delivery
#             self._publication = date_
#         elif isinstance(publication, datetime):
#             date_ = publication
#             self._publication = date_
#             if isinstance(delivery, NoInput):
#                 self._delivery = fx_index_.delivery(date_)
#
#         super().__init__(date=date_, value=value, identifier=identifier)
#         self._fx_index = fx_index_
#         self._is_cross = "usd" not in self.fx_index.pair
#
#     @property
#     def fx_index(self) -> FXIndex:
#         """The :class:`FXIndex` for the FX fixing."""
#         return self._fx_index
#
#     @property
#     def is_cross(self) -> bool:
#         """Whether the fixing is a cross rate derived from other USD dominated fixings."""
#         return self._is_cross
#
#     def _value_from_possible_inversion(self, identifier: str) -> DualTypes_:
#         direct, inverted = self.pair, f"{self.pair[3:6]}{self.pair[0:3]}"
#         try:
#             state, timeseries, bounds = fixings.__getitem__(identifier + "_" + direct)
#             exponent = 1.0
#         except ValueError as e:
#             try:
#                 state, timeseries, bounds = fixings.__getitem__(identifier + "_" + inverted)
#                 exponent = -1.0
#             except ValueError:
#                 raise e
#
#         if state == self._state:
#             return NoInput(0)
#         else:
#             self._state = state
#             v = self._lookup_and_calculate(timeseries, bounds)
#             if isinstance(v, NoInput):
#                 return NoInput(0)
#             self._value = v**exponent
#             return self._value
#
#     def _value_from_cross(self, identifier: str) -> DualTypes_:
#         lhs1, lhs2 = "usd" + self.pair[:3], self.pair[:3] + "usd"
#         try:
#             state_l, timeseries_l, bounds_l = fixings.__getitem__(identifier + "_" + lhs1)
#             exponent_l = -1.0
#         except ValueError:
#             try:
#                 state_l, timeseries_l, bounds_l = fixings.__getitem__(identifier + "_" + lhs2)
#                 exponent_l = 1.0
#             except ValueError:
#                 raise ValueError(
#                     "The LHS cross currency has no available fixing series, either "
#                     f"{identifier + '_' + lhs1} or {identifier + '_' + lhs2}"
#                 )
#
#         rhs1, rhs2 = "usd" + self.pair[3:], self.pair[3:] + "usd"
#         try:
#             state_r, timeseries_r, bounds_r = fixings.__getitem__(identifier + "_" + rhs1)
#             exponent_r = 1.0
#         except ValueError:
#             try:
#                 state_r, timeseries_r, bounds_r = fixings.__getitem__(identifier + "_" + rhs2)
#                 exponent_r = -1.0
#             except ValueError:
#                 raise ValueError(
#                     "The RHS cross currency has no available fixing series, either "
#                     f"{identifier + '_' + lhs1} or {identifier + '_' + lhs2}"
#                 )
#
#         if hash(state_l + state_r) == self._state:
#             return NoInput(0)
#         else:
#             self._state = hash(state_l + state_r)
#             v_l = self._lookup_and_calculate(timeseries_l, bounds_l)
#             v_r = self._lookup_and_calculate(timeseries_r, bounds_r)
#             if isinstance(v_l, NoInput) or isinstance(v_r, NoInput):
#                 return NoInput(0)
#             self._value = v_l**exponent_l * v_r**exponent_r
#             return self._value
#
#     @property
#     def publication(self) -> datetime:
#         return self._publication
#
#     @property
#     def delivery(self) -> datetime:
#         return self._delivery
#
#     @property
#     def value(self) -> DualTypes_:
#         if not isinstance(self._value, NoInput):
#             return self._value
#         else:
#             if isinstance(self._identifier, NoInput):
#                 return NoInput(0)
#             else:
#                 if self.is_cross:
#                     return self._value_from_cross(identifier=self._identifier)
#                 else:
#                     return self._value_from_possible_inversion(identifier=self._identifier)
#
#     def _lookup_and_calculate(
#         self, timeseries: Series, bounds: tuple[datetime, datetime] | None
#     ) -> DualTypes_:
#         return self._lookup(timeseries=timeseries, date=self.date, bounds=bounds)
#
#     @classmethod
#     def _lookup(
#         cls,
#         timeseries: Series[DualTypes],  # type: ignore[type-var]
#         date: datetime,
#         bounds: tuple[datetime, datetime] | None = None,
#     ) -> DualTypes_:
#         result = fixings.__base_lookup__(
#             fixing_series=timeseries,
#             lookup_date=date,
#             bounds=bounds,
#         )
#         if isinstance(result, Err):
#             if isinstance(result._exception, FixingRangeError):
#                 return NoInput(0)
#             result.unwrap()
#         else:
#             return result.unwrap()
#
#     @property
#     def pair(self) -> str:
#         """The currency pair related to the FX fixing."""
#         return self.fx_index.pair
#
#     def value_or_forecast(self, fx: FXForwards_) -> DualTypes:
#         """
#         Return the determined value of the fixing or forecast it if not available.
#
#         Parameters
#         ----------
#         fx: FXForwards, optional
#             The :class:`~rateslib.fx.FXForwards` object to forecast the forward FX rate.
#
#         Returns
#         -------
#         float, Dual, Dual2, Variable
#         """
#         if isinstance(self.value, NoInput):
#             fx_: FXForwards = _validate_obj_not_no_input(fx, "FXForwards")
#             return fx_.rate(pair=self.pair, settlement=self.delivery)
#         else:
#             return self.value
#
#     def try_value_or_forecast(self, fx: FXForwards_) -> Result[DualTypes]:
#         """
#         Return the determined value of the fixing or forecast it if not available.
#
#         Parameters
#         ----------
#         fx: FXForwards, optional
#             The :class:`~rateslib.fx.FXForwards` object to forecast the forward FX rate.
#
#         Returns
#         -------
#         Result[float, Dual, Dual2, Variable]
#         """
#         if isinstance(self.value, NoInput):
#             if isinstance(fx, NoInput):
#                 return Err(ValueError("Must provide `fx` argument to forecast FXFixing."))
#             else:
#                 return Ok(fx.rate(pair=self.pair, settlement=self.delivery))
#         else:
#             return Ok(self.value)


def _maybe_get_fx_index(val: FXIndex | str_) -> FXIndex_:
    if isinstance(val, NoInput):
        return NoInput(0)
    else:
        return _get_fx_index(val)


def _get_fx_index(val: FXIndex | str) -> FXIndex:
    if isinstance(val, FXIndex):
        return val
    else:
        pair = val.lower()
        try:
            return FXIndex(**defaults.fx_index[pair])
        except KeyError:
            try:
                reverse_fxi = FXIndex(**defaults.fx_index[f"{pair[3:]}{pair[:3]}"])
                return FXIndex(
                    pair=pair,
                    calendar=reverse_fxi.calendar,
                    settle=reverse_fxi.settle,
                    isda_mtm_calendar=reverse_fxi.isda_mtm_calendar,
                    isda_mtm_settle=reverse_fxi.isda_mtm_settle,
                )
            except KeyError:
                raise ValueError(
                    f"The FXIndex: '{pair}' was not found in `defaults`.\n"
                    "To add a default specification for the required FXIndex, for example, use:\n"
                    f"> defaults.fx_index['{pair}'] = {{ \n"
                    "      'pair': 'usdsek',\n"
                    "      'calendar': 'stk|fed',\n"
                    "      'settle': '2B',\n"
                    "      'isda_mtm_settle': '-2B',\n"
                    "      'isda_mtm_calendar': 'stk',\n"
                    "      'allow_cross': True,\n"
                    f"  }}\n"
                    "Alternatively, create an FXIndex directly and supply it to `pair`, "
                    "for example:\n> pair=FXIndex('usdsek', 'stk|fed\\, 2)"
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

       from rateslib.data.fixings import IBORFixing
       from rateslib.data.fixings import FloatRateIndex
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

       from rateslib.data.fixings import IBORStubFixing
       from rateslib.data.fixings import FloatRateSeries
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
        rate_series: FloatRateSeries | str,
        accrual_start: datetime,
        accrual_end: datetime,
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
        date: datetime_ = NoInput(0),
    ) -> None:
        super().__init__(value=value, date=date, identifier=identifier)  # type: ignore[arg-type]
        self._accrual_start = accrual_start
        self._accrual_end = accrual_end
        self._series = _get_float_rate_series(rate_series)
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

    def reset(self, state: int_ = NoInput(0)) -> None:
        if not isinstance(self._fixing1, NoInput):
            self._fixing1.reset(state=state)
        if not isinstance(self._fixing2, NoInput):
            self._fixing2.reset(state=state)
        self._value = NoInput(0)

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
                if sample_end > self.accrual_end and sample_end < right[1]:
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
    fixing_method: FloatFixingMethod or str
        The :class:`FloatFixingMethod` object used to combine multiple RFR fixings.
    method_param: int
        A parameter required by the ``fixing_method``.
    spread_compound_method: SpreadCompoundMethod or str
        A :class:`SpreadCompoundMethod` object used define the calculation of the addition of the
        ``float_spread``.
    float_spread: float, DUal, Dual2, Variable
        An additional amount added to the calculation to determine the final period rate.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib.enums.parameters import SpreadCompoundMethod, FloatFixingMethod
       from rateslib.data.fixings import RFRFixing
       from rateslib.data.fixings import FloatRateIndex
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
           spread_compound_method="NoneSimple",
           fixing_method="RFRPaymentDelay",
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
        fixing_method: FloatFixingMethod | str,
        method_param: int,
        spread_compound_method: SpreadCompoundMethod | str,
        float_spread: DualTypes,
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ):
        self._identifier = identifier if isinstance(identifier, NoInput) else identifier.upper()
        self._value = value
        self._state = 0

        self._float_spread = float_spread
        self._spread_compound_method = _get_spread_compound_method(spread_compound_method)
        self._rate_index = rate_index
        self._value = value
        self._accrual_start = accrual_start
        self._accrual_end = accrual_end
        self._fixing_method = _get_float_fixing_method(fixing_method)
        self._method_param = method_param
        self._populated = Series(index=[], data=[], dtype=float)  # type: ignore[assignment]

    def reset(self, state: int_ = NoInput(0)) -> None:
        if isinstance(state, NoInput) or self._state == state:
            self._populated = Series(index=[], data=[], dtype=float)  # type: ignore[assignment]
            self._value = NoInput(0)
            self._state = 0

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

    @property
    def unpopulated(self) -> Series[DualTypes]:  # type: ignore[type-var]
        """The fixings that are not published but are required to determine the period fixing."""
        return Series(index=self.dates_obs[:-1], data=np.nan, dtype=object).drop(  # type: ignore[return-value]
            self.populated.index
        )

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
                fixing_method=fixing_method,
                method_param=method_param,
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
        the method's **observation** dates."""
        return _RFRRate._get_dcf_values(
            dcf_dates=self.dates_obs,
            fixing_convention=self.rate_index.convention,
            fixing_calendar=self.rate_index.calendar,
        )

    @cached_property
    def dcfs_dcf(self) -> Arr1dF64:
        """A sequence of floats defining the individual **DCF** values associated with
        the **DCF** dates natural to the fixing rates."""
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


class FloatRateIndex:
    """
    Define the parameters of a specific interest rate index.

    Parameters
    ----------
    frequency : Frequency or str
        The specific tenor of the interest rate index.
    series : Series or str
        The general parameters applied to any tenor of this particular interest rate series.

    Examples
    --------
    None
    """

    _frequency: Frequency
    _series: FloatRateSeries

    def __init__(
        self,
        frequency: Frequency | str,
        series: FloatRateSeries | str,
    ) -> None:
        self._series = _get_float_rate_series(series)
        self._frequency = _get_frequency(frequency, NoInput(0), self.calendar)

    @property
    def frequency(self) -> Frequency:
        """The specific tenor of the interest rate index."""
        return self._frequency

    @property
    def series(self) -> FloatRateSeries:
        """The general parameters applied to any tenor of this particular interest rate series."""
        return self._series

    @property
    def lag(self) -> int:
        """The lag for the determining the publication date of the interest rate index."""
        return self.series.lag

    @property
    def calendar(self) -> CalTypes:
        """The calendar associated with the publication of the interest rate index."""
        return self.series.calendar

    @property
    def modifier(self) -> Adjuster:
        """The :class:`Adjuster` associated with the end accrual day of the interest rate index."""
        return self.series.modifier

    @property
    def eom(self) -> bool:
        """Whether the interest rate index adopts an end of month convention."""
        return self.series.eom

    @property
    def convention(self) -> Convention:
        """The :class:`Convention` associated with the publication of the interest rate index."""
        return self.series.convention


class FloatRateSeries:
    """
    Define the general parameters of multiple tenors of an interest rate series.

    .. role:: red

    .. role:: green

    Parameters
    ----------
    lag: int, :red:`required`
        The number of business days by which the fixing date is lagged to the accrual start date.
    calendar: Calendar, str :green:`required`
        The calendar associated with the floating rate's date determination.
    modifier: Adjuster, str, :red:`required`
        The :class:`Adjuster` associated with the end accrual day of the floating rate's date.
    convention: Convention, str, :green:`required`
        The day count :class:`~rateslib.scheduling.Convention` associated with the floating rate.
    eom: bool, :red:`required`
        Whether the interest rate index natively adopts EoM roll preference or not.

    """

    _lag: int
    _calendar: CalTypes
    _modifier: Adjuster
    _convention: Convention
    _eom: bool

    def __init__(
        self,
        lag: int,
        calendar: CalTypes | str,
        modifier: Adjuster | str,
        convention: Convention | str,
        eom: bool,
    ) -> None:
        self._lag = lag
        self._calendar = get_calendar(calendar)
        self._modifier = _get_adjuster(modifier)
        self._convention = _get_convention(convention)
        self._eom = eom

    @property
    def lag(self) -> int:
        return self._lag

    @property
    def calendar(self) -> CalTypes:
        return self._calendar

    @property
    def convention(self) -> Convention:
        return self._convention

    @property
    def modifier(self) -> Adjuster:
        return self._modifier

    @property
    def eom(self) -> bool:
        return self._eom


class _IBORRate:
    @staticmethod
    def _rate(
        *,
        rate_curve: _BaseCurve | dict[str, _BaseCurve] | NoInput,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        start: datetime,
        end: datetime,
        method_param: int,
        stub: bool,
        float_spread: DualTypes,
        rate_series: FloatRateSeries | NoInput,
        frequency: Frequency,
    ) -> Result[DualTypes]:
        rate_series_ = _maybe_get_rate_series_from_curve(
            rate_curve=rate_curve, rate_series=rate_series, method_param=method_param
        )
        fixing_date = rate_series_.calendar.lag_bus_days(start, -rate_series_.lag, settlement=False)
        if stub:
            # TODO: pass through tenor convention and modifier to the interpolated stub
            return _IBORRate._rate_interpolated_stub(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                float_spread=float_spread,
                rate_series=rate_series_,
            )
        else:
            return _IBORRate._rate_single_tenor(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )

    @staticmethod
    def _rate_interpolated_stub(
        rate_curve: _BaseCurve | dict[str, _BaseCurve] | NoInput,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
        rate_series: FloatRateSeries,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, NoInput):
            # will attempt to forecast stub period from rate_curve
            if isinstance(rate_curve, dict):
                return _IBORRate._rate_interpolated_stub_from_curve_dict(
                    rate_curve=rate_curve,
                    fixing_date=fixing_date,
                    start=start,
                    end=end,
                    float_spread=float_spread,
                )
            else:
                return _IBORRate._rate_stub_forecast_from_curve(
                    rate_curve=rate_curve,
                    fixing_date=fixing_date,
                    start=start,
                    end=end,
                    float_spread=float_spread,
                )
        else:
            # will maybe find relevant fixing values in Series
            return _IBORRate._rate_interpolated_stub_maybe_from_fixings(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                rate_series=rate_series,
                float_spread=float_spread,
            )

    @staticmethod
    def _rate_interpolated_stub_maybe_from_fixings(
        rate_curve: _BaseCurve_ | dict[str, _BaseCurve],
        rate_fixings: DualTypes | Series[DualTypes] | str,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
        rate_series: FloatRateSeries,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, str):
            tenors, dates, fixings_ = fixings.get_stub_ibor_fixings(
                value_start_date=start,
                value_end_date=end,
                fixing_calendar=rate_series.calendar,
                fixing_modifier=rate_series.modifier,
                fixing_identifier=rate_fixings,
                fixing_date=fixing_date,
            )
            if len(tenors) == 0:
                # nothing found
                return _IBORRate._rate_interpolated_stub(
                    rate_curve=rate_curve,
                    rate_fixings=NoInput(0),  # no fixings are found
                    fixing_date=fixing_date,
                    start=start,
                    end=end,
                    float_spread=float_spread,
                    rate_series=rate_series,
                )
            elif len(tenors) == 1:
                if fixings_[0] is None:
                    return _IBORRate._rate_interpolated_stub(
                        rate_curve=rate_curve,
                        rate_fixings=NoInput(0),  # no fixings are found
                        fixing_date=fixing_date,
                        start=start,
                        end=end,
                        float_spread=float_spread,
                        rate_series=rate_series,
                    )
                return Ok(fixings_[0] + float_spread / 100.0)
            else:
                if fixings_[0] is None or fixings_[1] is None:
                    # missing data exists
                    return _IBORRate._rate_interpolated_stub(
                        rate_curve=rate_curve,
                        rate_fixings=NoInput(0),  # no fixings are found
                        fixing_date=fixing_date,
                        start=start,
                        end=end,
                        float_spread=float_spread,
                        rate_series=rate_series,
                    )
                return Ok(
                    _IBORRate._interpolated_stub_rate(
                        left_date=dates[0],
                        right_date=dates[1],
                        left_rate=fixings_[0],
                        right_rate=fixings_[1],
                        maturity_date=end,
                        float_spread=float_spread,
                    )
                )
        elif isinstance(rate_fixings, Series):
            raise ValueError(err.VE_FIXINGS_BAD_TYPE)
        else:
            return Ok(rate_fixings + float_spread / 100.0)

    @staticmethod
    def _rate_interpolated_stub_from_curve_dict(
        rate_curve: dict[str, _BaseCurve],
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        """
        Get the rate on all available curves in dict and then determine the ones to interpolate.
        """

        def _rate(c: _BaseCurve, tenor: str) -> DualTypes:
            if c._base_type == _CurveType.dfs:
                return c._rate_with_raise(start, tenor)
            else:  # values
                return c._rate_with_raise(fixing_date, tenor)  # tenor is not used on LineCurve

        try:
            values = {
                add_tenor(start, k, v.meta.modifier, v.meta.calendar): _rate(v, k)
                for k, v in rate_curve.items()
            }
        except Exception as e:
            return Err(e)
        values = dict(sorted(values.items()))
        dates, rates = list(values.keys()), list(values.values())
        if end > dates[-1]:
            warnings.warn(
                "Interpolated stub period has a length longer than the provided "
                "IBOR curve tenors: using the longest IBOR value.",
                UserWarning,
            )
            ret: DualTypes = rates[-1]
        elif end < dates[0]:
            warnings.warn(
                "Interpolated stub period has a length shorter than the provided "
                "IBOR curve tenors: using the shortest IBOR value.",
                UserWarning,
            )
            ret = rates[0]
        else:
            i = index_left(dates, len(dates), end)
            ret = rates[i] + (rates[i + 1] - rates[i]) * (
                (end - dates[i]).days / (dates[i + 1] - dates[i]).days
            )
        return Ok(ret + float_spread / 100.0)

    @staticmethod
    def _rate_single_tenor(
        rate_curve: _BaseCurve | dict[str, _BaseCurve] | NoInput,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        frequency: Frequency,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, NoInput):
            return _IBORRate._rate_tenor_forecast_from_curve(
                rate_curve=rate_curve,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )
        else:
            return _IBORRate._rate_tenor_maybe_from_fixings(
                rate_curve=rate_curve,
                rate_fixings=rate_fixings,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )

    @staticmethod
    def _rate_tenor_maybe_from_fixings(
        rate_curve: _BaseCurve_ | dict[str, _BaseCurve],
        rate_fixings: DualTypes | Series[DualTypes] | str,  # type: ignore[type-var]
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        frequency: Frequency,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        if isinstance(rate_fixings, str | Series):
            if isinstance(rate_fixings, str):
                identifier = rate_fixings
                _, fixings_, bounds = fixings[identifier]
            else:
                identifier = "<SERIES_OBJECT>"
                fixings_ = rate_fixings
                bounds = (rate_fixings.index.min(), rate_fixings.index.max())

            if fixing_date <= bounds[1]:
                try:
                    fixing = fixings_.loc[fixing_date]
                    return Ok(fixing + float_spread / 100.0)
                except KeyError:
                    warnings.warn(
                        f"Fixings are provided in series: '{identifier}', but the value for "
                        f" date: {fixing_date} is not found.\nAttempting to forecast from "
                        f"the `rate_curve`.",
                    )
            return _IBORRate._rate_tenor_forecast_from_curve(
                rate_curve=rate_curve,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )
        else:
            # is just a scalar value so return directly.
            return Ok(rate_fixings + float_spread / 100.0)

    @staticmethod
    def _rate_tenor_forecast_from_curve(
        rate_curve: _BaseCurve_ | dict[str, _BaseCurve],
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        frequency: Frequency,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        tenor = _get_tenor_from_frequency(frequency)
        if isinstance(rate_curve, NoInput):
            return Err(ValueError(err.VE_NEEDS_RATE_TO_FORECAST_TENOR_IBOR))
        elif isinstance(rate_curve, dict):
            remapped_rate_curve = {k.lower(): v for k, v in rate_curve.items()}
            rate_curve_ = remapped_rate_curve[tenor.lower()]
            return _IBORRate._rate_tenor_forecast_from_curve(
                rate_curve=rate_curve_,
                fixing_date=fixing_date,
                start=start,
                end=end,
                frequency=frequency,
                float_spread=float_spread,
            )
        else:
            if rate_curve._base_type == _CurveType.dfs:
                try:
                    r = rate_curve._rate_with_raise(start, tenor) + float_spread / 100.0
                except Exception as e:
                    return Err(e)
                else:
                    return Ok(r)
            else:
                try:
                    r = rate_curve._rate_with_raise(fixing_date, NoInput(0)) + float_spread / 100.0
                except Exception as e:
                    return Err(e)
                else:
                    return Ok(r)

    @staticmethod
    def _rate_stub_forecast_from_curve(
        rate_curve: _BaseCurve_,
        fixing_date: datetime,
        start: datetime,
        end: datetime,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        if isinstance(rate_curve, NoInput):
            return Err(ValueError(err.VE_NEEDS_RATE_TO_FORECAST_STUB_IBOR))

        if rate_curve._base_type == _CurveType.dfs:
            try:
                r = rate_curve._rate_with_raise(start, end) + float_spread / 100.0
            except Exception as e:
                return Err(e)
            else:
                return Ok(r)
        else:
            try:
                r = rate_curve[fixing_date] + float_spread / 100.0
            except Exception as e:
                return Err(e)
            else:
                return Ok(r)

    @staticmethod
    def _interpolated_stub_rate(
        left_date: datetime,
        right_date: datetime,
        left_rate: DualTypes,
        right_rate: DualTypes,
        maturity_date: datetime,
        float_spread: DualTypes,
    ) -> DualTypes:
        return (
            left_rate
            + (maturity_date - left_date).days
            / (right_date - left_date).days
            * (right_rate - left_rate)
            + float_spread / 100.0
        )


class _RFRRate:
    """
    Class for maintaining methods related to calculating the period rate for an RFR compounded
    period. These periods have multiple branches depending upon;

    - which `fixing_method` has been selected.
    - which `spread_compound_method` has been selected (if the `float_spread` is non-zero).
    - whether there are any known fixings that must be populated to the calculation or unknown
      fixings must be forecast by some curve.

    """

    @staticmethod
    def _rate(
        start: datetime,
        end: datetime,
        rate_curve: _BaseCurve_,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        fixing_method: FloatFixingMethod,
        method_param: int,
        spread_compound_method: SpreadCompoundMethod,
        float_spread: DualTypes,
        rate_series: FloatRateSeries | NoInput,
    ) -> Result[  # type: ignore[type-var]
        tuple[
            DualTypes,
            tuple[datetime, datetime] | None,
            tuple[datetime, datetime] | None,
            Arr1dObj | None,
            Arr1dObj | None,
            Arr1dF64 | None,
            Arr1dF64 | None,
            Series[DualTypes] | None,
            Series[DualTypes] | None,
            Series[DualTypes] | None,
        ]
    ]:
        """
        To avoid repeated calculation, this function will pass back the data it calculates.
        In some short-circuited calculation not all data will have been calculated and returns
        None

        - 0: rate
        - 1: date_boundary_obs
        - 2: date_boundary_dcf
        - 3: dates_obs
        - 4: dates_dcf
        - 5: dcfs_obs
        - 6: dcfs_dcf
        - 7: fixing_rates
        - 8: populated
        - 9: unpopulated

        """

        if isinstance(rate_fixings, int | float | Dual | Dual2 | Variable):
            # a scalar value is assumed to have been pre-computed **including** the float spread
            # otherwise this information is of no use, since a computation including a
            # complicated float spread cannot be performed on just a compounded or average rate.
            return Ok((rate_fixings,) + (None,) * 9)

        rate_series_ = _maybe_get_rate_series_from_curve(
            rate_curve=rate_curve,
            rate_series=rate_series,
            method_param=method_param,
        )

        bounds_obs, bounds_dcf, is_matching = _RFRRate._adjust_dates(
            start=start,
            end=end,
            fixing_method=fixing_method,
            method_param=method_param,
            fixing_calendar=rate_series_.calendar,
        )

        # >>> short-circuit here before any complex calculation or date lookup is performed.
        # EFFICIENT CALCULATION:
        if _RFRRate._is_rfr_efficient(
            rate_curve=rate_curve,
            rate_fixings=rate_fixings,
            float_spread=float_spread,
            spread_compound_method=spread_compound_method,
            fixing_method=fixing_method,
        ):
            r_result = _RFRRate._efficient_calculation(
                rate_curve=rate_curve,  # type: ignore[arg-type]  # is pre-checked
                bounds_obs=bounds_obs,
                float_spread=float_spread,
            )
            if isinstance(r_result, Err):
                return r_result
            else:
                return Ok((r_result.unwrap(), bounds_obs, bounds_dcf) + (None,) * 7)

        dates_obs, dates_dcf, dcfs_obs, dcfs_dcf, populated, unpopulated, fixing_rates = (
            _RFRRate._get_dates_and_fixing_rates_from_fixings(
                rate_series=rate_series_,
                bounds_obs=bounds_obs,
                bounds_dcf=bounds_dcf,
                is_matching=is_matching,
                rate_fixings=rate_fixings,
                fixing_method=fixing_method,
                method_param=method_param,
            )
        )

        # >>> short circuit and perform a semi-efficient calculation splicing fixings with DFs
        # SEMI-EFFICIENT CALCULATION:
        if _RFRRate._is_rfr_efficient(
            rate_curve, NoInput(0), float_spread, spread_compound_method, fixing_method
        ):
            r = _RFRRate._semi_efficient_calculation(
                rate_curve=rate_curve,  # type: ignore[arg-type]  # guaranteed by if statement
                populated=populated,
                unpopulated=unpopulated,
                obs_date_boundary=bounds_obs,
                float_spread=float_spread,
                fixing_dcfs=dcfs_dcf,
            )
            return Ok(
                (
                    r,
                    bounds_obs,
                    bounds_dcf,
                    dates_obs,
                    dates_dcf,
                    dcfs_obs,
                    dcfs_dcf,
                    fixing_rates,
                    populated,
                    unpopulated,
                )
            )

        update = _RFRRate._forecast_fixing_rates_from_curve(
            unpopulated=unpopulated,
            populated=populated,
            fixing_rates=fixing_rates,
            rate_curve=rate_curve,
            dates_obs=dates_obs,
            dcfs_obs=dcfs_obs,
        )
        if isinstance(update, Err):
            return update

        # INEFFICIENT CALCULATION having derived all individual fixings.
        r_result = _RFRRate._inefficient_calculation(
            fixing_rates=fixing_rates,
            fixing_dcfs=dcfs_dcf,
            fixing_method=fixing_method,
            method_param=method_param,
            spread_compound_method=spread_compound_method,
            float_spread=float_spread,
        )
        if isinstance(r_result, Err):
            return r_result
        else:
            return Ok(
                (
                    r_result.unwrap(),
                    bounds_obs,
                    bounds_dcf,
                    dates_obs,
                    dates_dcf,
                    dcfs_obs,
                    dcfs_dcf,
                    fixing_rates,
                    populated,
                    unpopulated,
                )
            )

    @staticmethod
    def _efficient_calculation(
        rate_curve: _BaseCurve,  # discount factors only
        bounds_obs: tuple[datetime, datetime],
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        """
        Perform an efficient calculation only after the `_is_rfr_efficient` check is performed.

        This calculation uses only discount factors and does not calculate individual fixing rates.
        """
        try:
            r = (
                rate_curve._rate_with_raise(
                    effective=bounds_obs[0],
                    termination=bounds_obs[1],
                    # no other arguments are necessary following _is_efficient check
                )
                + float_spread / 100.0
            )
        except Exception as e:
            return Err(e)
        else:
            return Ok(r)

    @staticmethod
    def _semi_efficient_calculation(
        rate_curve: _BaseCurve,
        populated: Series[DualTypes],  # type: ignore[type-var]
        fixing_dcfs: Arr1dF64,
        unpopulated: Series[DualTypes],  # type: ignore[type-var]
        obs_date_boundary: tuple[datetime, datetime],
        float_spread: DualTypes,
    ) -> DualTypes:
        """
        Perform an efficient calculation only after the `_is_rfr_efficient` check is performed.

        This calculation combines some known fixing values with a forecast people calculated
        using discount factors and not by calculating a number of individual fixing rates.
        """
        populated_index = prod(
            [
                1.0 + d * r / 100.0
                for r, d in zip(populated, fixing_dcfs[: len(populated)], strict=False)
            ]
        )
        # TODO this is not date safe, i.e. a date maybe before the curve starts and DF is zero.
        if len(unpopulated) == 0:  # i.e. all fixings are known without needing to forecast
            unpopulated_index: DualTypes = 1.0
        else:
            unpopulated_index = rate_curve[unpopulated.index[0]] / rate_curve[obs_date_boundary[1]]
        rate: DualTypes = ((populated_index * unpopulated_index) - 1.0) * 100.0 / fixing_dcfs.sum()
        return rate + float_spread / 100.0

    @staticmethod
    def _inefficient_calculation(
        fixing_rates: Series,
        fixing_dcfs: Arr1dF64,
        fixing_method: FloatFixingMethod,
        method_param: int,
        spread_compound_method: SpreadCompoundMethod,
        float_spread: DualTypes,
    ) -> Result[DualTypes]:
        """
        Perform a full calculation forecasting every individual fixing rate and then compounding
        or averaging each of them up in turn, combining a float spread if necessary.
        """
        # overwrite with lockout rates: this is needed if rates have been forecast from curve.
        if fixing_method in [FloatFixingMethod.RFRLockout, FloatFixingMethod.RFRLockoutAverage]:
            # overwrite fixings
            if method_param >= len(fixing_rates):
                return Err(
                    ValueError(err.VE_LOCKOUT_METHOD_PARAM.format(method_param, fixing_rates))
                )
            for i in range(1, method_param + 1):
                fixing_rates.iloc[-i] = fixing_rates.iloc[-(method_param + 1)]

        if fixing_method in [
            FloatFixingMethod.RFRLockoutAverage,
            FloatFixingMethod.RFRLookbackAverage,
            FloatFixingMethod.RFRObservationShiftAverage,
            FloatFixingMethod.RFRPaymentDelayAverage,
        ]:
            return _RFRRate._calculator_rate_rfr_avg_with_spread(
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                rates=fixing_rates.to_numpy(),
                dcf_vals=fixing_dcfs,
            )
        else:
            return _RFRRate._calculator_rate_rfr_isda_compounded_with_spread(
                float_spread=float_spread,
                spread_compound_method=spread_compound_method,
                rates=fixing_rates.to_numpy(),
                dcf_vals=fixing_dcfs,
            )

    @staticmethod
    def _get_dates_and_fixing_rates_from_fixings(
        rate_series: FloatRateSeries,
        bounds_obs: tuple[datetime, datetime],
        bounds_dcf: tuple[datetime, datetime],
        is_matching: bool,
        rate_fixings: Series[DualTypes] | str_,  # type: ignore[type-var]
        fixing_method: FloatFixingMethod,
        method_param: int,
    ) -> tuple[  # type: ignore[type-var]
        Arr1dObj,
        Arr1dObj,
        Arr1dF64,
        Arr1dF64,
        Series[DualTypes],
        Series[DualTypes],
        Series[DualTypes],
    ]:
        """
        For an RFR period, construct the necessary fixing dates and DCF schedule.
        Populate fixings from a Series if any values are available to yield.
        Return Series objects.

        """
        dates_obs, dates_dcf, fixing_rates = _RFRRate._get_obs_and_dcf_dates(
            fixing_calendar=rate_series.calendar,
            fixing_convention=rate_series.convention,
            obs_date_boundary=bounds_obs,
            dcf_date_boundary=bounds_dcf,
            is_matching=is_matching,
        )
        dcfs_dcf = _RFRRate._get_dcf_values(
            dcf_dates=dates_dcf,
            fixing_convention=rate_series.convention,
            fixing_calendar=rate_series.calendar,
        )
        if is_matching:
            dcfs_obs = dcfs_dcf.copy()
        else:
            dcfs_obs = _RFRRate._get_dcf_values(
                dcf_dates=dates_obs,
                fixing_convention=rate_series.convention,
                fixing_calendar=rate_series.calendar,
            )

        # populate Series with values
        if isinstance(rate_fixings, NoInput):
            populated: Series[DualTypes] = Series(index=[], data=np.nan, dtype=object)  # type: ignore[type-var, assignment]
            unpopulated: Series[DualTypes] = Series(index=dates_obs[:-1], data=np.nan, dtype=object)  # type: ignore[type-var, assignment]
        elif isinstance(rate_fixings, str | Series):
            fixing_rates, populated, unpopulated = (
                _RFRRate._push_rate_fixings_as_series_to_fixing_rates(
                    fixing_rates=fixing_rates,
                    rate_fixings=rate_fixings,
                    fixing_method=fixing_method,
                    method_param=method_param,
                )
            )
        else:
            raise ValueError(err.VE_FIXINGS_BAD_TYPE)  # unknown fixings type fixings runtime issue

        return dates_obs, dates_dcf, dcfs_obs, dcfs_dcf, populated, unpopulated, fixing_rates

    @staticmethod
    def _forecast_fixing_rates_from_curve(
        unpopulated: Series[DualTypes],  # type: ignore[type-var]
        populated: Series[DualTypes],  # type: ignore[type-var]
        fixing_rates: Series[DualTypes],  # type: ignore[type-var]
        rate_curve: _BaseCurve_,
        dates_obs: Arr1dObj,
        dcfs_obs: Arr1dF64,
    ) -> Result[None]:
        # determine unpopulated fixings from the curve
        if len(unpopulated) > 0 and isinstance(rate_curve, NoInput):
            return Err(FixingMissingForecasterError())  # missing data - needs a rate_curve

        unpopulated_obs_dates = dates_obs[len(populated) :]
        if len(unpopulated_obs_dates) > 1:
            if isinstance(rate_curve, NoInput):
                return Err(ValueError(err.VE_NEEDS_RATE_TO_FORECAST_RFR))

            if rate_curve._base_type == _CurveType.values:
                try:
                    r = [
                        rate_curve._rate_with_raise(unpopulated_obs_dates[_], NoInput(0))
                        for _ in range(len(unpopulated))
                    ]
                except Exception as e:
                    return Err(e)
            else:
                v = np.array([rate_curve[_] for _ in unpopulated_obs_dates])
                r = (v[:-1] / v[1:] - 1) * 100 / dcfs_obs[len(populated) :]
            unpopulated = Series(
                index=unpopulated.index,
                data=r,
            )
        fixing_rates.update(unpopulated)

        return Ok(None)

    @staticmethod
    def _push_rate_fixings_as_series_to_fixing_rates(
        fixing_rates: Series[DualTypes],  # type: ignore[type-var]
        rate_fixings: str | Series[DualTypes],  # type: ignore[type-var]
        fixing_method: FloatFixingMethod,
        method_param: int,
    ) -> tuple[Series[DualTypes], Series[DualTypes], Series[DualTypes]]:  # type: ignore[type-var]
        """
        Populates an empty fixings_rates Series with values from a looked up fixings collection.
        """
        if isinstance(rate_fixings, str):
            fixing_series = fixings[rate_fixings][1]
        else:
            fixing_series = rate_fixings
        if fixing_rates.index[0] > fixing_series.index[-1]:
            # then no fixings in scope, so no changes
            return fixing_rates, Series(index=[], data=np.nan), fixing_rates.copy()  # type: ignore[return-value]
        else:
            fixing_rates.update(fixing_series)

        # push lockout rates if they are available
        if fixing_method in [FloatFixingMethod.RFRLockout, FloatFixingMethod.RFRLockoutAverage]:
            if method_param >= len(fixing_rates):
                raise ValueError(err.VE_LOCKOUT_METHOD_PARAM.format(method_param, fixing_rates))
            if not isna(fixing_rates.iloc[-(1 + method_param)]):  # type: ignore[arg-type]
                for i in range(method_param):
                    fixing_rates.iloc[-(1 + i)] = fixing_rates.iloc[-(1 + method_param)]

        # validate for missing and expected fixings in the fixing Series
        nans = isna(fixing_rates)
        populated, unpopulated = fixing_rates[~nans], fixing_rates[nans]
        if (
            len(unpopulated) > 0
            and len(populated) > 0
            and unpopulated.index[0] < populated.index[-1]
        ):
            raise ValueError(
                err.VE02_5.format(  # there is at least one missing fixing data item
                    rate_fixings,
                    fixing_rates[nans].index[0].strftime("%d-%m-%Y"),
                    fixing_rates[~nans].index[-1].strftime("%d-%m-%Y"),
                )
            )

        # validate for unexpected fixings provided in the fixings Series
        if 0 < len(populated) < len(fixing_series[populated.index[0] : populated.index[-1]]):
            # then fixing series contains an unexpected fixing.
            warnings.warn(
                err.W02_0.format(
                    rate_fixings,
                    populated.index[0].strftime("%d-%m-%Y"),
                    populated.index[-1].strftime("%d-%m-%Y"),
                ),
                UserWarning,
            )

        return fixing_rates, populated, unpopulated

    @staticmethod
    def _adjust_dates(
        start: datetime,
        end: datetime,
        fixing_method: FloatFixingMethod,
        method_param: int,
        fixing_calendar: CalTypes,
    ) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime], bool]:
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
            start_obs, end_obs = start, end
            start_dcf, end_dcf = start, end
            is_matching = True
        elif fixing_method in [
            FloatFixingMethod.RFRObservationShift,
            FloatFixingMethod.RFRObservationShiftAverage,
        ]:
            start_obs = fixing_calendar.lag_bus_days(start, -method_param, settlement=False)
            end_obs = fixing_calendar.lag_bus_days(end, -method_param, settlement=False)
            start_dcf, end_dcf = start_obs, end_obs
            is_matching = True
        else:
            # fixing_method in [
            #    FloatFixingMethod.RFRLookback,
            #    FloatFixingMethod.RFRLookbackAverage,
            # ]:
            start_obs = fixing_calendar.lag_bus_days(start, -method_param, settlement=False)
            end_obs = fixing_calendar.lag_bus_days(end, -method_param, settlement=False)
            start_dcf, end_dcf = start, end
            is_matching = False

        return (start_obs, end_obs), (start_dcf, end_dcf), is_matching

    @staticmethod
    def _get_obs_and_dcf_dates(
        fixing_calendar: CalTypes,
        fixing_convention: Convention,
        obs_date_boundary: tuple[datetime, datetime],
        dcf_date_boundary: tuple[datetime, datetime],
        is_matching: bool,
    ) -> tuple[Arr1dObj, Arr1dObj, Series[DualTypes]]:  # type: ignore[type-var]
        # construct empty Series for rates and DCFs
        obs_dates = np.array(fixing_calendar.bus_date_range(*obs_date_boundary))
        fixing_rates: Series[DualTypes] = Series(index=obs_dates[:-1], data=np.nan, dtype=object)  # type: ignore[type-var, assignment]
        if is_matching:
            dcf_dates = obs_dates
        else:
            dcf_dates = np.array(fixing_calendar.bus_date_range(*dcf_date_boundary))
        return obs_dates, dcf_dates, fixing_rates

    @staticmethod
    def _get_dcf_values(
        dcf_dates: Arr1dObj,
        fixing_convention: Convention,
        fixing_calendar: CalTypes,
    ) -> Arr1dF64:
        if fixing_convention == Convention.Act365F:
            days = np.fromiter((_.days for _ in dcf_dates[1:] - dcf_dates[:-1]), float)
            return days / 365.0
        elif fixing_convention == Convention.Act360:
            days = np.fromiter((_.days for _ in dcf_dates[1:] - dcf_dates[:-1]), float)
            return days / 360.0
        elif fixing_convention == Convention.Bus252:
            return np.array([1.0 / 252.0] * (len(dcf_dates) - 1))
        else:
            # this is unconventional fixing convention. Should maybe be avoided altogether.
            return np.array(
                [
                    dcf(
                        start=dcf_dates[i],
                        end=dcf_dates[i + 1],
                        convention=fixing_convention,
                        calendar=fixing_calendar,
                    )
                    for i in range(len(dcf_dates) - 1)
                ]
            )

    @staticmethod
    def _is_rfr_efficient(
        rate_curve: _BaseCurve_,
        rate_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        float_spread: DualTypes,
        spread_compound_method: SpreadCompoundMethod,
        fixing_method: FloatFixingMethod,
    ) -> bool:
        """
        Check all of the conditions to return an RFR rate directly from discount factors.

        - A rate curve must be available and be based on DFs.
        - There cannot be any known fixings that must be incorporated into the calculation.
        - Only PaymentDelay and ObservationShift fixing methods are suitable for this calculation.
        - Only NoneSimple spread compound method is suitable, or the float spread must be 0.0.

        """
        return (
            isinstance(rate_curve, _BaseCurve)
            and rate_curve._base_type == _CurveType.dfs
            and isinstance(rate_fixings, NoInput)
            and fixing_method
            in [FloatFixingMethod.RFRPaymentDelay, FloatFixingMethod.RFRObservationShift]
            and (float_spread == 0.0 or spread_compound_method == SpreadCompoundMethod.NoneSimple)
        )

    @staticmethod
    def _calculator_rate_rfr_avg_with_spread(
        float_spread: DualTypes,
        spread_compound_method: SpreadCompoundMethod,
        rates: Arr1dF64,
        dcf_vals: Arr1dF64,
    ) -> Result[DualTypes]:
        """
        Calculate all in rate with float spread under averaging.

        Parameters
        ----------
        rates : Series
            The rates which are expected for each daily period.
        dcf_vals : Series
            The weightings which are used for each rate in the compounding formula.

        Returns
        -------
        float, Dual, Dual2
        """
        if spread_compound_method != SpreadCompoundMethod.NoneSimple:
            return Err(ValueError(err.VE_SPREAD_METHOD_RFR.format(spread_compound_method)))
        else:
            _: DualTypes = (dcf_vals * rates).sum() / dcf_vals.sum() + float_spread / 100
            return Ok(_)

    @staticmethod
    def _calculator_rate_rfr_isda_compounded_with_spread(
        float_spread: DualTypes,
        spread_compound_method: SpreadCompoundMethod,
        rates: Arr1dObj,
        dcf_vals: Arr1dF64,
    ) -> Result[DualTypes]:
        """
        Calculate all in rates with float spread under different compounding methods.

        Parameters
        ----------
        rates : Series
            The rates which are expected for each daily period.
        dcf_vals : Series
            The weightings which are used for each rate in the compounding formula.

        Returns
        -------
        float, Dual, Dual2
        """
        if float_spread == 0 or spread_compound_method == SpreadCompoundMethod.NoneSimple:
            _: DualTypes = (
                (1 + dcf_vals * rates / 100).prod() - 1
            ) * 100 / dcf_vals.sum() + float_spread / 100
            return Ok(_)
        elif spread_compound_method == SpreadCompoundMethod.ISDACompounding:
            _ = (
                ((1 + dcf_vals * (rates / 100 + float_spread / 10000)).prod() - 1)
                * 100
                / dcf_vals.sum()
            )
            return Ok(_)
        else:  # spread_compound_method == SpreadCompoundMethod.ISDAFlatCompounding:
            sub_cashflows = (rates / 100 + float_spread / 10000) * dcf_vals
            C_i = 0.0
            for i in range(1, len(sub_cashflows)):
                C_i += sub_cashflows[i - 1]
                sub_cashflows[i] += C_i * rates[i] / 100 * dcf_vals[i]
            _ = sub_cashflows.sum() * 100 / dcf_vals.sum()
            return Ok(_)


def _get_float_rate_series(val: FloatRateSeries | str) -> FloatRateSeries:
    if isinstance(val, FloatRateSeries):
        return val
    else:
        try:
            return FloatRateSeries(**defaults.float_series[val.lower()])
        except KeyError:
            raise ValueError(
                f"The FloatRateSeries: '{val.lower()}' was not found in `defaults`.\n"
                "To add a default specification for a FloatRateSeries, for example, use:\n"
                f"> defaults.float_series['{val.lower()}'] = {{ \n"
                "      'lag': 2,\n"
                "      'calendar': 'nyc',\n"
                "      'modifier': 'MF',\n"
                "      'convention': 'Act360',\n"
                "      'eom': False,\n"
                f"  }}"
            )


def _get_float_rate_series_or_blank(val: FloatRateSeries | str_) -> FloatRateSeries | NoInput:
    if isinstance(val, NoInput):
        return val
    else:
        return _get_float_rate_series(val)


def _maybe_get_rate_series_from_curve(
    rate_curve: CurveOption_,
    rate_series: FloatRateSeries | NoInput,
    method_param: int,
) -> FloatRateSeries:
    """Get a rate fixing calendar and convention from a Curve or the alternatives if not given."""

    if isinstance(rate_curve, NoInput):
        if isinstance(rate_series, NoInput):
            raise ValueError(err.VE_NEEDS_CURVE_OR_INDEX)
        else:
            # get params from rate_index
            return rate_series
    else:
        if isinstance(rate_curve, dict):
            cal_ = list(rate_curve.values())[0].meta.calendar
            conv_ = list(rate_curve.values())[0].meta.convention
            mod_ = list(rate_curve.values())[0].meta.modifier
        else:
            cal_ = rate_curve.meta.calendar
            conv_ = rate_curve.meta.convention
            mod_ = rate_curve.meta.modifier

        if isinstance(rate_series, NoInput):
            # get params from rate_curve
            return FloatRateSeries(
                lag=method_param,
                calendar=cal_,
                convention=conv_,
                modifier=mod_,
                eom=False,  # TODO: un hard code this
            )
        else:
            if rate_series.convention != conv_:
                raise ValueError(
                    err.MISMATCH_RATE_INDEX_PARAMETERS.format(
                        "convention", conv_, rate_series.convention
                    )
                )
            # dual parameters may be specified
            # get params from rate_index
            return rate_series


def _leg_fixings_to_list(rate_fixings: LegFixings, n_periods: int) -> list[PeriodFixings]:
    """Perform a conversion of 'LegRateFixings' into a list of PeriodFixings."""
    if isinstance(rate_fixings, NoInput):
        # NoInput is converted to a list of NoInputs
        return [NoInput(0)] * n_periods
    elif isinstance(rate_fixings, tuple):
        # A tuple must be a 2-tuple which is converted to a first item and then multiplied.
        return [rate_fixings[0]] + [rate_fixings[1]] * (n_periods - 1)
    elif isinstance(rate_fixings, list):
        # A list is padded with NoInputs
        return rate_fixings + [NoInput(0)] * (n_periods - len(rate_fixings))
    elif isinstance(rate_fixings, str | Series):
        # A string or seried is multiplied
        return [rate_fixings] * n_periods
    else:
        # A scalar value is padded with NoInputs.
        return [rate_fixings] + [NoInput(0)] * (n_periods - 1)  # type: ignore[return-value]


__all__ = [
    "FloatRateSeries",
    "FloatRateIndex",
    "RFRFixing",
    "IBORFixing",
    "IBORStubFixing",
    "IndexFixing",
    "FXIndex",
    "FXFixing",
    "_FXFixingMajor",
    "_UnitFixing",
    "_BaseFixing",
]
