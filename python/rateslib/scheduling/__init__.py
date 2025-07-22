from __future__ import annotations

from rateslib.rs import Adjuster, Cal, NamedCal, RollDay, UnionCal

# from rateslib.scheduling.wrappers import Schedule
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.dcfs import dcf
from rateslib.scheduling.frequency import add_tenor
from rateslib.scheduling.rollday import get_imm, next_imm
from rateslib.scheduling.scheduling import Schedule, _check_regular_swap, _infer_stub_date

RollDay.__doc__ = "Enumerable type for roll day types."
Adjuster.__doc__ = "Enumerable type for date adjustment rules."


__all__ = (
    "Schedule",
    "_infer_stub_date",
    "_check_regular_swap",
    "add_tenor",
    "Adjuster",
    "Cal",
    "dcf",
    "NamedCal",
    "RollDay",
    "UnionCal",
    "get_calendar",
    "get_imm",
    "next_imm",
)
