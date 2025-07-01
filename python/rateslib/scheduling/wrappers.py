from __future__ import annotations

from typing import TYPE_CHECKING

from rateslib.calendars import get_calendar
from rateslib.calendars.rs import _get_rollday
from rateslib.rs import Frequency

if TYPE_CHECKING:
    from rateslib.typing import (
        CalInput,
        int_,
    )


def _get_frequency(frequency: str, roll: str | int_, calendar: CalInput) -> Frequency:
    frequency_: str = frequency.upper()[-1]
    if frequency_ == "D":
        n_: int = int(frequency[:-1])
        return Frequency.CalDays(n_)
    elif frequency_ == "B":
        n_ = int(frequency[:-1])
        return Frequency.BusDays(n_, get_calendar(calendar))
    elif frequency_ == "W":
        n_ = int(frequency[:-1])
        return Frequency.Weeks(n_)
    elif frequency_ == "M":
        return Frequency.Months(1, _get_rollday(roll))
    elif frequency_ == "B":
        return Frequency.Months(2, _get_rollday(roll))
    elif frequency_ == "Q":
        return Frequency.Months(3, _get_rollday(roll))
    elif frequency_ == "T":
        return Frequency.Months(4, _get_rollday(roll))
    elif frequency_ == "S":
        return Frequency.Months(6, _get_rollday(roll))
    elif frequency_ == "A":
        return Frequency.Months(12, _get_rollday(roll))
    elif frequency_ == "Z":
        return Frequency.Zero()
    else:
        raise ValueError("Frequency can not be determined from input.")
