from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from rateslib.enums.generics import (
    NoInput,
)
from rateslib.scheduling import Convention, Frequency, dcf

if TYPE_CHECKING:
    from rateslib.typing import (
        Adjuster_,
        CalTypes,
        datetime,
        datetime_,
    )


class _PeriodParams:
    _start: datetime
    _end: datetime
    _frequency: Frequency
    _convention: Convention
    _termination: datetime_
    _calendar: CalTypes
    _stub: bool
    _adjuster: Adjuster_

    def __init__(
        self,
        _start: datetime,
        _end: datetime,
        _frequency: Frequency,
        _convention: Convention,
        _termination: datetime_,
        _calendar: CalTypes,
        _adjuster: Adjuster_,
        _stub: bool,
    ):
        if _end < _start:
            raise ValueError("`end` cannot be before `start`.")

        self._start = _start
        self._end = _end
        self._frequency = _frequency
        self._convention = _convention
        self._termination = _termination
        self._calendar = _calendar
        self._stub = _stub
        self._adjuster = _adjuster

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def end(self) -> datetime:
        return self._end

    @property
    def termination(self) -> datetime_:
        return self._termination

    @property
    def adjuster(self) -> Adjuster_:
        return self._adjuster

    @property
    def calendar(self) -> CalTypes:
        return self._calendar

    @property
    def stub(self) -> bool:
        return self._stub

    @property
    def convention(self) -> Convention:
        return self._convention

    @property
    def frequency(self) -> Frequency:
        return self._frequency

    @cached_property
    def dcf(self) -> float:
        """
        float : Calculated with appropriate ``convention`` over the period.
        """
        return dcf(
            start=self.start,
            end=self.end,
            convention=self.convention,
            termination=self.termination,
            frequency=self.frequency,
            stub=self.stub,
            roll=NoInput(0),  # `frequency` is a Frequency.
            calendar=self.calendar,
            adjuster=self.adjuster,
        )
