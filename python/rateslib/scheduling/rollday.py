from __future__ import annotations

import calendar as calendar_mod
from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.rs import Imm, RollDay
from rateslib.scheduling.calendars import _adjust_date

if TYPE_CHECKING:
    from rateslib.typing import CalInput, Callable, int_


def _get_rollday(roll: RollDay | str | int_) -> RollDay | None:
    """Convert a user str or int into a RollDay enum object."""
    if isinstance(roll, RollDay):
        return roll
    elif isinstance(roll, str):
        return {
            "EOM": RollDay.Day(31),
            "SOM": RollDay.Day(1),
            "IMM": RollDay.IMM(),
        }[roll.upper()]
    elif isinstance(roll, int):
        return RollDay.Day(roll)
    return None


def _get_roll(month: int, year: int, roll: str | int) -> datetime:
    if isinstance(roll, str):
        if roll == "eom":
            date = Imm.Eom.get(year, month)
        elif roll == "som":
            date = datetime(year, month, 1)
        else:  # roll == "imm":
            date = Imm.Wed3.get(year, month)
    else:
        try:
            date = datetime(year, month, roll)
        except ValueError:  # day is out of range for month, i.e. 30 or 31
            date = Imm.Eom.get(year, month)
    return date


def _is_eom(date: datetime) -> bool:
    """
    Test whether a given date is end of month.

    Parameters
    ----------
    date : datetime,
        Date to test

    Returns
    -------
    bool
    """
    return date.day == calendar_mod.monthrange(date.year, date.month)[1]


def _is_eom_cal(date: datetime, cal: CalInput) -> bool:
    """Test whether a given date is end of month under a specific calendar"""
    end_day = calendar_mod.monthrange(date.year, date.month)[1]
    eom = datetime(date.year, date.month, end_day)
    adj_eom = _adjust_date(eom, "P", cal)
    return date == adj_eom


def _is_som(date: datetime) -> bool:
    """
    Test whether a given date is start of month.

    Parameters
    ----------
    date : datetime,
        Date to test

    Returns
    -------
    bool
    """
    return date.day == 1


def _is_imm(date: datetime) -> bool:
    from rateslib.rs import Imm

    return Imm.Wed3.validate(date)


_IS_ROLL: dict[str, Callable[..., bool]] = {
    "eom": _is_eom,
    "som": _is_som,
    "imm": _is_imm,
}
