from __future__ import annotations

from rateslib.rs import Adjuster, Cal, Frequency, NamedCal, RollDay, StubInference, UnionCal
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.dcfs import dcf
from rateslib.scheduling.frequency import add_tenor
from rateslib.scheduling.rollday import get_imm, next_imm
from rateslib.scheduling.wrappers import Schedule

RollDay.__doc__ = "Enumerable type for roll day types."

StubInference.__doc__ = "Enumerable type for :class:`~rateslib.scheduling.Schedule` stub inference."

Adjuster.__doc__ = "Enumerable type for date adjustment rules."

Frequency.__doc__ = "Enumerable type for a scheduling frequency."

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
    "add_tenor",
    "Adjuster",
    "Frequency",
    "StubInference",
    "Cal",
    "dcf",
    "NamedCal",
    "RollDay",
    "UnionCal",
    "get_calendar",
    "get_imm",
    "next_imm",
)
