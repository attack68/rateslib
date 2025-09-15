from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from pandas import Series

from rateslib import fixings
from rateslib.enums.generics import NoInput

if TYPE_CHECKING:
    from rateslib.typing import DualTypes, DualTypes_, str_


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

    def reset(self) -> None:
        """
        Sets the ``value`` attribute to :class:`rateslib.enums.generics.NoInput`, which allows it
        to be redetermined from a timeseries.

        Returns
        -------
        None
        """
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
