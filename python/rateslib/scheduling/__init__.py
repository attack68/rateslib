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

import rateslib.rs
from rateslib.rs import (
    Adjuster,
    Cal,
    CalendarManager,
    Frequency,
    Imm,
    NamedCal,
    RollDay,
    StubInference,
    UnionCal,
)
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.convention import Convention
from rateslib.scheduling.dcfs import dcf
from rateslib.scheduling.frequency import add_tenor
from rateslib.scheduling.imm import get_imm, next_imm
from rateslib.scheduling.schedule import Schedule

# Patch the namespace for pyo3 pickling: see https://github.com/PyO3/pyo3/discussions/5226
# rateslib.rs.RollDay_Day = rateslib.rs.RollDay.Day  # type: ignore[attr-defined]
# rateslib.rs.RollDay_IMM = rateslib.rs.RollDay.IMM  # type: ignore[attr-defined]


class PicklingContainer:
    pass


rateslib.rs.PyAdjuster = PicklingContainer()  # type: ignore[attr-defined]

rateslib.rs.PyAdjuster.Actual = rateslib.rs.Adjuster.Actual  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.Following = rateslib.rs.Adjuster.Following  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.ModifiedFollowing = rateslib.rs.Adjuster.ModifiedFollowing  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.Previous = rateslib.rs.Adjuster.Previous  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.ModifiedPrevious = rateslib.rs.Adjuster.ModifiedPrevious  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.FollowingSettle = rateslib.rs.Adjuster.FollowingSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.ModifiedFollowingSettle = rateslib.rs.Adjuster.ModifiedFollowingSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.PreviousSettle = rateslib.rs.Adjuster.PreviousSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.ModifiedPreviousSettle = rateslib.rs.Adjuster.ModifiedPreviousSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.BusDaysLagSettle = rateslib.rs.Adjuster.BusDaysLagSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.CalDaysLagSettle = rateslib.rs.Adjuster.CalDaysLagSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.FollowingExLast = rateslib.rs.Adjuster.FollowingExLast  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.FollowingExLastSettle = rateslib.rs.Adjuster.FollowingExLastSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster.BusDaysLagSettleInAdvance = rateslib.rs.Adjuster.BusDaysLagSettleInAdvance  # type: ignore[attr-defined]
#
# rateslib.rs.Frequency_CalDays = rateslib.rs.Frequency.CalDays  # type: ignore[attr-defined]
# rateslib.rs.Frequency_BusDays = rateslib.rs.Frequency.BusDays  # type: ignore[attr-defined]
# rateslib.rs.Frequency_Months = rateslib.rs.Frequency.Months  # type: ignore[attr-defined]
# rateslib.rs.Frequency_Zero = rateslib.rs.Frequency.Zero  # type: ignore[attr-defined]

Imm.__doc__ = """
Enumerable type for International Money-Market (IMM) date definitions.

For further information on these descriptors see the Rust low level docs
for :rust:`Imm <scheduling>`.
"""

StubInference.__doc__ = """
Enumerable type for :class:`~rateslib.scheduling.Schedule` stub inference.
"""

Adjuster.__doc__ = """
Enumerable type for date adjustment rules.

.. rubric:: Variants

.. ipython:: python
   :suppress:

   from rateslib.rs import Adjuster
   variants = [item for item in Adjuster.__dict__ if \\
       "__" != item[:2] and \\
       item not in ['adjust', 'adjusts', 'to_json', 'reverse'] \
   ]

.. ipython:: python

   variants

"""

RollDay.__doc__ = """
Enumerable type for roll days.

.. rubric:: Variants

.. ipython:: python
   :suppress:

   from rateslib.rs import RollDay
   variants = ["Day(int)", "IMM()"]

.. ipython:: python

   variants

"""

Frequency.__doc__ = """
Enumerable type for a scheduling frequency.

.. rubric:: Variants

.. ipython:: python
   :suppress:

   from rateslib.rs import Frequency
   variants = ["BusDays(int, calendar)", "CalDays(int)", "Months(int, rollday | None)", "Zero()"]

.. ipython:: python

   variants

"""

Cal.__doc__ = """
A business day calendar defined by weekends and a holiday list.

Parameters
----------
holidays: list[datetime]
    A list of specific non-business days.
week_mask: list[int]
    A list of days defined as weekends, e.g. [5,6] for Saturday and Sunday.
"""

UnionCal.__doc__ = """
A calendar defined by a business day intersection of multiple :class:`~rateslib.scheduling.Cal`
objects.

Parameters
----------
calendars: list[Cal]
    A list of :class:`~rateslib.scheduling.Cal` objects whose combination will define the
    business and non-business days.
settlement_calendars: list[Cal]
    A list of :class:`~rateslib.scheduling.Cal` objects whose combination will define the
    settleable and non-settleable days.
"""

NamedCal.__doc__ = """
A wrapped :class:`~rateslib.scheduling.Cal` or
:class:`~rateslib.scheduling.UnionCal` constructed with a string parsing syntax.

This instance can only be constructed from named :class:`~rateslib.scheduling.Cal` objects that
have already been populated to the ``calendars`` :class:`~rateslib.scheduling.CalendarManager`.
Each *NamedCal* uses data shared in memory and does **not** reconstruct or copy the entire
list of holidays for every instantiation of this class.

Parameters
----------
name: str
    The names of the calendars to populate the ``calendars`` and ``settlement_calendars``
    arguments of a :class:`~rateslib.scheduling.UnionCal`. The individual calendar names must
    pre-exist in the :class:`~rateslib.scheduling.CalendarManager`. The pipe operator
    separates the two fields.

Examples
--------
.. ipython:: python
   :suppress:

   from rateslib.scheduling import NamedCal, UnionCal

.. ipython:: python

   named_cal = NamedCal("ldn,tgt|fed")
   assert isinstance(named_cal.inner, UnionCal)
   assert len(named_cal.inner.calendars) == 2
   assert len(named_cal.inner.settlement_calendars) == 1
"""

__all__ = (
    "Schedule",
    "Cal",
    "NamedCal",
    "UnionCal",
    "CalendarManager",
    "Adjuster",
    "Convention",
    "Frequency",
    "Imm",
    "RollDay",
    "StubInference",
    "add_tenor",
    "get_calendar",
    "get_imm",
    "next_imm",
    "dcf",
)
