from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rateslib.typing import (  # pragma: no cover
        CalTypes,
        datetime,
    )


def _get_first_bus_day(dates: list[datetime], calendar: CalTypes) -> datetime:
    if len(dates) == 0:
        raise ValueError("The list of `dates` from which to select a business day is empty.")
    for date in dates:
        if calendar.is_bus_day(date):
            return date
    raise ValueError("No valid business days were found in `dates`.")
