from __future__ import annotations

from typing import TYPE_CHECKING

from stack_data.utils import cached_property

from rateslib import defaults
from rateslib.calendars import get_calendar
from rateslib.calendars.rs import _get_rollday
from rateslib.default import NoInput, _drb
from rateslib.rs import Adjuster, Frequency
from rateslib.rs import Schedule as Schedule_rs
from rateslib.scheduling.scheduling import _validate_effective, _validate_termination

if TYPE_CHECKING:
    from rateslib.typing import (
        CalInput,
        CalTypes,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
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


class Schedule:
    _obj: Schedule_rs

    def obj(self) -> Schedule_rs:
        """A wrapped instance of Schedule_rs"""

    def __init__(
        self,
        effective: datetime | str,
        termination: datetime | str,
        frequency: str,
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | int_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        eval_date: datetime_ = NoInput(0),
        eval_mode: str_ = NoInput(0),
    ) -> None:
        eom_: bool = _drb(defaults.eom, eom)
        eval_mode_: str = _drb(defaults.eval_mode, eval_mode).lower()
        modifier_: str = _drb(defaults.modifier, modifier).upper()
        # payment_lag_: int = _drb(defaults.payment_lag, payment_lag)
        calendar_: CalTypes = get_calendar(calendar)

        effective_: datetime = _validate_effective(
            effective, eval_mode_, eval_date, modifier_, calendar_, roll
        )
        termination_: datetime = _validate_termination(
            termination, effective_, modifier_, calendar_, roll, eom_
        )

        ueffective: datetime = effective_
        utermination: datetime = termination_

        self.obj = Schedule_rs(
            ueffective=ueffective,
            utermination=utermination,
            frequency=_get_frequency(frequency, roll, calendar_),
            calendar=get_calendar(calendar),
            accrual_adjuster=Adjuster.Actual(),
            payment_adjuster=Adjuster.Actual(),
            ufront_stub=_drb(None, front_stub),
            uback_stub=_drb(None, back_stub),
        )

    @cached_property
    def uschedule(self) -> list[datetime]:
        return self.obj.uschedule

    @cached_property
    def aschedule(self) -> list[datetime]:
        return self.obj.aschedule

    @cached_property
    def pschedule(self) -> list[datetime]:
        return self.obj.pschedule
