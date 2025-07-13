from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame
from stack_data.utils import cached_property

from rateslib import defaults
from rateslib.calendars import get_calendar
from rateslib.calendars.rs import _get_adjuster, _get_rollday
from rateslib.default import NoInput, _drb
from rateslib.rs import Frequency, StubInference
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
        return Frequency.CalDays(n_ * 7)
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


def _get_stub_inference(
    stub: str, front_stub: datetime_, back_stub: datetime_
) -> StubInference | None:
    _map: dict[str, StubInference] = {
        "SHORTFRONT": StubInference.ShortFront,
        "LONGFRONT": StubInference.LongFront,
        "SHORTBACK": StubInference.ShortBack,
        "LONGBACK": StubInference.LongBack,
    }
    stub = stub.upper()
    _ = {v: v in stub for v in _map}
    if not isinstance(front_stub, NoInput):
        # cannot infer front stubs, since it is explicitly provided
        _["SHORTFRONT"] = False
        _["LONGFRONT"] = False
    if not isinstance(back_stub, NoInput):
        # cannot infer back stubs, since it is explicitly provided
        _["SHORTBACK"] = False
        _["LONGBACK"] = False
    ret: StubInference | None = None
    for k, v in _.items():
        if v:
            ret = _map[k]
            break
    return ret


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
        stub: str = _drb(defaults.stub, stub)
        eval_mode_: str = _drb(defaults.eval_mode, eval_mode).lower()
        modifier_: str = _drb(defaults.modifier, modifier).upper()
        payment_lag_: int = _drb(defaults.payment_lag, payment_lag)
        calendar_: CalTypes = get_calendar(calendar)
        effective_: datetime = _validate_effective(
            effective, eval_mode_, eval_date, modifier_, calendar_, roll
        )
        termination_: datetime = _validate_termination(
            termination, effective_, modifier_, calendar_, roll, eom_
        )
        accrual_adjuster = _get_adjuster(modifier_)
        payment_adjuster = _get_adjuster(f"{payment_lag_}B")

        self.obj = Schedule_rs(
            effective=effective_,
            termination=termination_,
            frequency=_get_frequency(frequency, roll, calendar_),
            calendar=calendar_,
            accrual_adjuster=accrual_adjuster,
            payment_adjuster=payment_adjuster,
            front_stub=_drb(None, front_stub),
            back_stub=_drb(None, back_stub),
            eom=eom_,
            stub_inference=_get_stub_inference(stub, front_stub, back_stub),
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

    @cached_property
    def table(self) -> DataFrame:
        """
        DataFrame : Rows of schedule dates and information.
        """
        df = DataFrame(
            {
                defaults.headers["stub_type"]: ["Stub" if _ else "Regular" for _ in self._stubs],
                defaults.headers["u_acc_start"]: self.uschedule[:-1],
                defaults.headers["u_acc_end"]: self.uschedule[1:],
                defaults.headers["a_acc_start"]: self.aschedule[:-1],
                defaults.headers["a_acc_end"]: self.aschedule[1:],
                defaults.headers["payment"]: self.pschedule[1:],
            },
        )
        return df

    @cached_property
    def _stubs(self) -> list[bool]:
        front_stub = self.obj.frequency.is_stub(self.uschedule[0], self.uschedule[1], True)
        back_stub = self.obj.frequency.is_stub(self.uschedule[-2], self.uschedule[-1], False)
        if len(self.uschedule) == 2:  # single period
            return [front_stub or back_stub]
        else:
            return [front_stub] + [False] * (len(self.uschedule) - 3) + [back_stub]

    @cached_property
    def n_periods(self) -> int:
        return len(self.obj.uschedule) - 1
