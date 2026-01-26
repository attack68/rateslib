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
from rateslib.rs import Adjuster, Cal, Frequency, Imm, NamedCal, RollDay, StubInference, UnionCal
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.convention import Convention
from rateslib.scheduling.dcfs import dcf
from rateslib.scheduling.frequency import add_tenor
from rateslib.scheduling.imm import get_imm, next_imm
from rateslib.scheduling.schedule import Schedule

# Patch the namespace for pyo3 pickling: see https://github.com/PyO3/pyo3/discussions/5226
rateslib.rs.RollDay_Day = rateslib.rs.RollDay.Day  # type: ignore[attr-defined]
rateslib.rs.RollDay_IMM = rateslib.rs.RollDay.IMM  # type: ignore[attr-defined]

rateslib.rs.PyAdjuster_Actual = rateslib.rs.Adjuster.Actual  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_Following = rateslib.rs.Adjuster.Following  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_ModifiedFollowing = rateslib.rs.Adjuster.ModifiedFollowing  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_Previous = rateslib.rs.Adjuster.Previous  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_ModifiedPrevious = rateslib.rs.Adjuster.ModifiedPrevious  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_FollowingSettle = rateslib.rs.Adjuster.FollowingSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_ModifiedFollowingSettle = rateslib.rs.Adjuster.ModifiedFollowingSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_PreviousSettle = rateslib.rs.Adjuster.PreviousSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_ModifiedPreviousSettle = rateslib.rs.Adjuster.ModifiedPreviousSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_BusDaysLagSettle = rateslib.rs.Adjuster.BusDaysLagSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_CalDaysLagSettle = rateslib.rs.Adjuster.CalDaysLagSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_FollowingExLast = rateslib.rs.Adjuster.FollowingExLast  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_FollowingExLastSettle = rateslib.rs.Adjuster.FollowingExLastSettle  # type: ignore[attr-defined]
rateslib.rs.PyAdjuster_BusDaysLagSettleInAdvance = rateslib.rs.Adjuster.BusDaysLagSettleInAdvance  # type: ignore[attr-defined]

rateslib.rs.Frequency_CalDays = rateslib.rs.Frequency.CalDays  # type: ignore[attr-defined]
rateslib.rs.Frequency_BusDays = rateslib.rs.Frequency.BusDays  # type: ignore[attr-defined]
rateslib.rs.Frequency_Months = rateslib.rs.Frequency.Months  # type: ignore[attr-defined]
rateslib.rs.Frequency_Zero = rateslib.rs.Frequency.Zero  # type: ignore[attr-defined]

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
       item not in ['adjust', 'adjusts'] \
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
A wrapped :class:`~rateslib.scheduling.UnionCal` constructed with a string parsing syntax.

Parameters
----------
name: str
    The names of the calendars to populate the ``calendars`` and ``settlement_calendars``
    arguments of a :class:`~rateslib.scheduling.UnionCal`. The individual calendar names must
    pre-exist in the :ref:`defaults <defaults-arg-input>`. The pipe operator separates the two
    fields.

Examples
--------
.. ipython:: python
   :suppress:

   from rateslib.scheduling import NamedCal

.. ipython:: python

   named_cal = NamedCal("ldn,tgt|fed")
   assert len(named_cal.union_cal.calendars) == 2
   assert len(named_cal.union_cal.settlement_calendars) == 1
"""

__all__ = (
    "Schedule",
    "Cal",
    "NamedCal",
    "UnionCal",
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
