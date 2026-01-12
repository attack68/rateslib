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
    """Parameters of *Period* cashflows associated with some
    :class:`~rateslib.scheduling.Schedule`.

    Parameters
    ----------
    _start: datetime
        The identified start date of the *Period*.
    _end: datetime
        The identified end date of the *Period*.
    _frequency: Frequency
        The :class:`~rateslib.scheduling.Frequency` associated with the *Period*.
    _convention: Convention
        The day count :class:`~rateslib.scheduling.Convention` associated with the *Period*.
    _termination: datetime, optional
        The termination date of an external :class:`~rateslib.scheduling.Schedule`.
    _calendar: Calendar, optional
         The calendar associated with the *Period*.
    _stub: bool
        Whether the *Period* is defined as a stub according to some external
        :class:`~rateslib.scheduling.Schedule`.
    _adjuster: Adjuster, optional
        The date :class:`~rateslib.scheduling.Adjuster` applied to unadjusted dates in the
        external :class:`~rateslib.scheduling.Schedule` to arrive at adjusted accrual dates.
    """

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
        """The identified start date of the *Period*."""
        return self._start

    @property
    def end(self) -> datetime:
        """The identified end date of the *Period*."""
        return self._end

    @property
    def termination(self) -> datetime_:
        """The termination date of an external :class:`~rateslib.scheduling.Schedule`."""
        return self._termination

    @property
    def adjuster(self) -> Adjuster_:
        """The date :class:`~rateslib.scheduling.Adjuster` applied to unadjusted dates in the
        external :class:`~rateslib.scheduling.Schedule` to arrive at adjusted accrual dates."""
        return self._adjuster

    @property
    def calendar(self) -> CalTypes:
        """The calendar associated with the *Period*."""
        return self._calendar

    @property
    def stub(self) -> bool:
        """Whether the *Period* is defined as a stub according to some external
        :class:`~rateslib.scheduling.Schedule`"""
        return self._stub

    @property
    def convention(self) -> Convention:
        """The day count :class:`~rateslib.scheduling.Convention` associated with the *Period*."""
        return self._convention

    @property
    def frequency(self) -> Frequency:
        """The :class:`~rateslib.scheduling.Frequency` associated with the *Period*."""
        return self._frequency

    @cached_property
    def dcf(self) -> float:
        """
        The DCF of the *Period* determined under its given parameters.
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
