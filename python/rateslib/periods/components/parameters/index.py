from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pandas import Series

import rateslib.errors as err
from rateslib import defaults
from rateslib.curves.curves import _index_value_from_series_no_curve, _try_index_value
from rateslib.enums.generics import (
    Err,
    NoInput,
    Ok,
    _drb,
)
from rateslib.enums.parameters import (
    IndexMethod,
    _get_index_method,
)
from rateslib.fixings import FixingRangeError
from rateslib.periods.components.parameters.base_fixing import _BaseFixing

if TYPE_CHECKING:
    from rateslib.typing import (
        Any,
        DualTypes,
        DualTypes_,
        Result,
        _BaseCurve_,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


class _IndexParams:
    _index_lag: int
    _index_method: IndexMethod
    _index_fixing: IndexFixing
    _index_base: IndexFixing
    _index_only: bool

    def __init__(
        self,
        *,
        _index_method: IndexMethod,
        _index_lag: int,
        _index_base: DualTypes_,
        _index_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
        _index_base_date: datetime_,
        _index_reference_date: datetime_,
        _index_only: bool,
    ) -> None:
        self._index_method = _index_method
        self._index_lag = _index_lag
        self._index_only = _index_only

        if isinstance(_index_fixings, Series):
            warnings.warn(err.FW_FIXINGS_AS_SERIES, FutureWarning)

        if isinstance(_index_base, NoInput) and isinstance(_index_fixings, Series):
            _index_base_value = IndexFixing._lookup(
                index_lag=self.index_lag,
                index_method=self.index_method,
                timeseries=_index_fixings,
                date=_index_base_date,  # type: ignore[arg-type]  # argument combinations
            )
            self._index_base = IndexFixing(
                date=_index_base_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_base_value,
                identifier=NoInput(0),
            )
        else:
            self._index_base = IndexFixing(
                date=_index_base_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_base,
                identifier=_index_fixings if isinstance(_index_fixings, str) else NoInput(0),
            )

        if isinstance(_index_fixings, Series):
            _index_ref_value = IndexFixing._lookup(
                index_lag=self.index_lag,
                index_method=self.index_method,
                timeseries=_index_fixings,
                date=_index_reference_date,  # type: ignore[arg-type]  # argument combinations
            )
            self._index_fixing = IndexFixing(
                date=_index_reference_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_ref_value,
                identifier=NoInput(0),
            )
        else:
            self._index_fixing = IndexFixing(
                date=_index_reference_date,  # type: ignore[arg-type]  # argument combinations
                index_lag=self.index_lag,
                index_method=self.index_method,
                value=_index_fixings if not isinstance(_index_fixings, str) else NoInput(0),
                identifier=_index_fixings if isinstance(_index_fixings, str) else NoInput(0),
            )

    @property
    def index_base(self) -> IndexFixing:
        return self._index_base

    @index_base.setter
    def index_base(self, value: Any) -> None:
        raise ValueError(err.VE_ATTRIBUTE_IS_IMMUTABLE.format("index_base"))

    @property
    def index_fixing(self) -> IndexFixing:
        return self._index_fixing

    @index_fixing.setter
    def index_fixing(self, value: Any) -> None:
        raise ValueError(err.VE_ATTRIBUTE_IS_IMMUTABLE.format("index_fixing"))

    @property
    def index_only(self) -> bool:
        return self._index_only

    @property
    def index_lag(self) -> int:
        return self._index_lag

    @property
    def index_method(self) -> IndexMethod:
        return self._index_method

    def try_index_ratio(
        self,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> Result[tuple[DualTypes, DualTypes, DualTypes]]:
        """
        Calculate the index ratio for the *Period*, including the numerator and denominator.

        .. math::

           I(m) = \\frac{I_{val}(m)}{I_{base}}

        Parameters
        ----------
        index_curve : _BaseCurve, optional
            The curve from which index values are forecast if required.

        Returns
        -------
        Result of tuple of float, Dual, Dual2, Variable for the ratio, numerator, denominator.
        """
        denominator_ = _try_index_value(
            index_fixings=self.index_base.value,
            index_date=self.index_base.date,
            index_curve=index_curve,
            index_lag=self.index_lag,
            index_method=self.index_method,
        )
        if isinstance(denominator_, Err):
            return denominator_
        numerator_ = _try_index_value(
            index_fixings=self.index_fixing.value,
            index_date=self.index_fixing.date,
            index_curve=index_curve,
            index_lag=self.index_lag,
            index_method=self.index_method,
        )
        if isinstance(numerator_, Err):
            return numerator_

        n_, d_ = numerator_.unwrap(), denominator_.unwrap()
        return Ok((n_ / d_, n_, d_))

    def index_ratio(
        self,
        index_curve: _BaseCurve_ = NoInput(0),
    ) -> tuple[DualTypes, DualTypes, DualTypes]:
        """
        Calculate the index ratio for the *Period*, including the numerator and denominator.

        .. math::

           I(m) = \\frac{I_{val}(m)}{I_{base}}

        Parameters
        ----------
        index_curve : _BaseCurve, optional
            The curve from which index values are forecast if required.

        Returns
        -------
        tuple of float, Dual, Dual2, Variable for the ratio, numerator, denominator.
        """
        return self.try_index_ratio(index_curve=index_curve).unwrap()


class IndexFixing(_BaseFixing):
    """
    An index fixing value for settlement of indexed cashflows.

    Parameters
    ----------
    index_lag: int
        The number months by which the reference date is lagged to derive an index value.
    index_method: IndexMethod
        The method used for calculating the index value. See :class:`IndexMethod`.
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

       from rateslib.periods.components.parameters import IndexFixing
       from rateslib.enums.parameters import IndexMethod
       from rateslib import defaults, dt
       from pandas import Series

    .. ipython:: python

       defaults.fixings.add("UK-CPI", Series(index=[dt(2000, 1, 1), dt(2000, 2, 1)], data=[100, 110.0]))
       index_fix = IndexFixing(date=dt(2000, 4, 15), identifier="UK-CPI", index_lag=3, index_method=IndexMethod.Daily)
       index_fix.value

    .. ipython:: python
       :suppress:

       defaults.fixings.pop("UK-CPI")

    """  # noqa: E501

    _index_lag: int
    _index_method: IndexMethod

    def __init__(
        self,
        *,
        index_lag: int,
        index_method: IndexMethod,
        date: datetime,
        value: DualTypes_ = NoInput(0),
        identifier: str_ = NoInput(0),
    ) -> None:
        super().__init__(date=date, value=value, identifier=identifier)

        self._index_lag = index_lag
        self._index_method = index_method

    @property
    def index_method(self) -> IndexMethod:
        """The :class:`IndexMethod` used for calculating the index value."""
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


def _init_or_none_IndexParams(
    _index_base: DualTypes_,
    _index_lag: int_,
    _index_method: IndexMethod | str_,
    _index_fixings: DualTypes | Series[DualTypes] | str_,  # type: ignore[type-var]
    _index_only: bool_,
    _index_base_date: datetime_,
    _index_reference_date: datetime_,
) -> _IndexParams | None:
    if all(
        isinstance(_, NoInput)
        for _ in (
            _index_base,
            _index_lag,
            _index_method,
            _index_fixings,
        )
    ):
        return None
    else:
        if isinstance(_index_base, str):
            raise ValueError(err.VE_INDEX_BASE_NO_STR)
        return _IndexParams(
            _index_base=_index_base,
            _index_lag=_drb(defaults.index_lag, _index_lag),
            _index_method=_get_index_method(_drb(defaults.index_method, _index_method)),
            _index_fixings=_index_fixings,
            _index_base_date=_index_base_date,
            _index_reference_date=_index_reference_date,
            _index_only=_drb(False, _index_only),
        )
