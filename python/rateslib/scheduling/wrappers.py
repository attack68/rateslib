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
        Adjuster,
        CalInput,
        CalTypes,
        RollDay,
        bool_,
        datetime,
        datetime_,
        int_,
        str_,
    )


def _get_frequency(
    frequency: str | Frequency, roll: str | RollDay | int_, calendar: CalInput
) -> Frequency:
    if isinstance(frequency, Frequency):
        if getattr(frequency, "roll", "no default") is None:
            return Frequency.Months(frequency.number, _get_rollday(roll))  # type: ignore[attr-defined]
        return frequency

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
        raise ValueError("Frequency can not be determined from `frequency` input.")


def _get_stub_inference(
    stub: str, front_stub: datetime_, back_stub: datetime_
) -> StubInference | None:
    """
    Convert `stub` as string to a `StubInference` enum based on what stubs are intended to be
    inferred and what stab dates are already provided. In a stub is provided as a date it
    will never be inferred.

    Parameters
    ----------
    stub: str
        The intention of the schedule for inferred stubs
    front_stub: datetime, optional
        If given StubInference will never contain any front elements.
    back_stub: datetime, optional
        If given StubInference will never contain any back elements.

    Returns
    -------
    StubInference or None
    """
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
    if sum(list(_.values())) > 1:
        raise ValueError("Must supply at least one stub date for dual sided inference.")
    for k, v in _.items():
        if v:
            ret = _map[k]
            break
    return ret


class Schedule:
    _obj: Schedule_rs

    @property
    def obj(self) -> Schedule_rs:
        """A wrapped instance of Schedule_rs"""
        return self._obj

    def __init__(
        self,
        effective: datetime | str,
        termination: datetime | str,
        frequency: str | Frequency,
        *,
        stub: str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | RollDay | int_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: int_ = NoInput(0),
        eval_date: datetime_ = NoInput(0),
        eval_mode: str_ = NoInput(0),
    ) -> None:
        eom_: bool = _drb(defaults.eom, eom)
        stub_: str = _drb(defaults.stub, stub)
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

        self._obj = Schedule_rs(
            effective=effective_,
            termination=termination_,
            frequency=_get_frequency(frequency, roll, calendar_),
            calendar=calendar_,
            accrual_adjuster=accrual_adjuster,
            payment_adjuster=payment_adjuster,
            front_stub=_drb(None, front_stub),
            back_stub=_drb(None, back_stub),
            eom=eom_,
            stub_inference=_get_stub_inference(stub_, front_stub, back_stub),
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
    def frequency(self) -> str:
        return self.obj.frequency.string()

    @cached_property
    def frequency_obj(self) -> Frequency:
        return self.obj.frequency

    def modifier(self) -> Adjuster:
        # legacy alias for Adjuster
        return self.obj.accrual_adjuster

    @cached_property
    def calendar(self) -> CalTypes:
        return self.obj.calendar

    @cached_property
    def accrual_adjuster(self) -> Adjuster:
        return self.obj.accrual_adjuster

    @cached_property
    def payment_adjuster(self) -> Adjuster:
        return self.obj.payment_adjuster

    @cached_property
    def termination(self) -> datetime:
        return self.obj.aschedule[-1]

    @cached_property
    def effective(self) -> datetime:
        return self.obj.aschedule[0]

    @cached_property
    def utermination(self) -> datetime:
        return self.obj.uschedule[-1]

    @cached_property
    def ueffective(self) -> datetime:
        return self.obj.uschedule[0]

    @cached_property
    def ufront_stub(self) -> datetime | None:
        return self.obj.ufront_stub

    @cached_property
    def uback_stub(self) -> datetime | None:
        return self.obj.uback_stub

    @cached_property
    def roll(self) -> str | int | NoInput:
        if isinstance(self.obj.frequency, Frequency.Months):  # type: ignore[arg-type]
            if self.obj.frequency.roll is None:  # type: ignore[attr-defined]
                return NoInput(0)
            else:
                return self.obj.frequency.roll  # type: ignore[attr-defined]
        else:
            return NoInput(0)

    @cached_property
    def table(self) -> DataFrame:
        """
        DataFrame : Rows of schedule dates and information.
        """
        if self.is_regular():
            stubs = ["Regular"] * self.n_periods
        else:
            us = self.uschedule
            front_stub = ["Stub"] if self.frequency.is_stub(us[0], us[1], True) else ["Regular"]
            back_stub = ["Stub"] if self.frequency.is_stub(us[-2], us[-1], False) else ["Regular"]
            if self.n_periods == 1:
                stubs = ["Stub"] if front_stub or back_stub else ["Regular"]
            else:  # self.n_periods >= 2:
                stubs = ["Stub"] if front_stub else ["Regular"]
                stubs += ["Regular"] * (self.n_periods - 2)
                stubs += ["Stub"] if back_stub else ["Regular"]

        df = DataFrame(
            {
                defaults.headers["stub_type"]: stubs,
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

    def __repr__(self) -> str:
        return f"<rl.Schedule at {hex(id(self))}>"

    def __str__(self) -> str:
        f: str = self.frequency_obj.__str__()
        a: str = self.accrual_adjuster.__str__()
        p: str = self.payment_adjuster.__str__()
        str_: str = f"freq: {f}, accrual adjuster: {a}, payment adjuster: {p},\n"
        ret: str = str_ + self.table.__repr__()
        return ret

    def is_regular(self) -> bool:
        return self.obj.is_regular()
