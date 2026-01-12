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

from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING

from pandas import DataFrame

from rateslib import defaults
from rateslib.default import _make_py_json
from rateslib.enums.generics import NoInput, _drb
from rateslib.rs import Adjuster, Frequency, RollDay, StubInference
from rateslib.rs import Schedule as Schedule_rs
from rateslib.scheduling.adjuster import _convert_to_adjuster, _get_adjuster
from rateslib.scheduling.calendars import _is_day_type_tenor, get_calendar
from rateslib.scheduling.frequency import _get_frequency, add_tenor
from rateslib.scheduling.rollday import _is_eom_cal

if TYPE_CHECKING:
    from rateslib.typing import (
        Adjuster_,
        Any,
        CalInput,
        CalTypes,
        bool_,
        datetime_,
        int_,
        str_,
    )


def _get_stub_inference(
    stub: str | StubInference, front_stub: datetime_, back_stub: datetime_
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
    if isinstance(stub, StubInference) or stub is None:
        return stub

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


def _get_adjuster_from_modifier(modifier: Adjuster | str_, mod_days: bool) -> Adjuster:
    if isinstance(modifier, Adjuster):
        return modifier  # use the adjuster as provided
    modifier_: str = _drb(defaults.modifier, modifier).upper()
    return _convert_to_adjuster(modifier_, settlement=False, mod_days=mod_days)


def _should_mod_days(tenor: datetime | str) -> bool:
    """Return whether a specified tenor should be subject to a `modifier`'s modification rule."""
    if isinstance(tenor, str):
        return not _is_day_type_tenor(tenor)
    else:
        # cannot infer any data to issue an overwrite
        return True


def _get_adjuster_from_lag_drb(lag: Adjuster | int_, default: str) -> Adjuster:
    if isinstance(lag, Adjuster):
        return lag
    else:
        lag_: int = _drb(getattr(defaults, default), lag)
        return _get_adjuster(f"{lag_}B")


def _get_adjuster_or_none(lag: Adjuster | None | int_, default: str) -> Adjuster | None:
    if lag is None:
        return None
    else:
        return _get_adjuster_from_lag_drb(lag, default)


class Schedule:
    """
    Generate a schedule of dates according to a regular pattern and calendar inference.

    Parameters
    ----------
    effective : datetime, str
        The unadjusted effective date. If given as adjusted, unadjusted alternatives may be
        inferred. If given as string tenor will be calculated from ``eval_date`` and ``eval_mode``.
    termination : datetime, str
        The unadjusted termination date. If given as adjusted, unadjusted alternatives may be
        inferred. If given as string tenor will be calculated from ``effective``.
    frequency : Frequency, str in {"M", "Q", "S", "A", "Z", "_D", "_B", "_W", "_M", "_Y"}
        The frequency of the schedule.
        If given as string will derive a :class:`~rateslib.scheduling.Frequency` aligning with:
        monthly ("M"), quarterly ("Q"), semi-annually ("S"), annually("A") or zero-coupon ("Z"), or
        a set number of calendar or business days ("_D", "_B"), weeks ("_W"), months ("_M") or
        years ("_Y").
        Where required, the :class:`~rateslib.scheduling.RollDay` is derived as per ``roll``
        and business day calendar as per ``calendar``.
    stub : StubInference, str in {"ShortFront", "LongFront", "ShortBack", "LongBack"}, optional
        The stub type used if stub inference is required. If given as string will derive a
        :class:`~rateslib.scheduling.StubInference`.
    front_stub : datetime, optional
        The unadjusted date for the start stub period. If given as adjusted, unadjusted
        alternatives may be inferred.
    back_stub : datetime, optional
        The unadjusted date for the back stub period. If given as adjusted, unadjusted
        alternatives may be inferred.
        See notes for combining ``stub``, ``front_stub`` and ``back_stub``
        and any automatic stub inference.
    roll : RollDay, int in [1, 31], str in {"eom", "imm", "som"}, optional
        The roll day of the schedule. If not given or not available in ``frequency`` will be
        inferred for monthly frequency variants.
    eom : bool, optional
        Use an end of month preference rather than regular rolls for ``roll`` inference. Set by
        default. Not required if ``roll`` is defined.
    modifier : Adjuster, str in {"NONE", "F", "MF", "P", "MP"}, optional
        The :class:`~rateslib.scheduling.Adjuster` used for adjusting unadjusted schedule dates
        into adjusted dates. If given as string must define simple date rolling rules.
    calendar : calendar, str, optional
        The business day calendar object to use. If string will call
        :meth:`~rateslib.scheduling.get_calendar`.
    payment_lag: Adjuster, int, optional
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        a payment date. If given as integer will define the number of business days to
        lag payments by.
    payment_lag_exchange: Adjuster, int, optional
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        additional payment date. If given as integer will define the number of business days to
        lag payments by.
    extra_lag: Adjuster, int, optional
        The :class:`~rateslib.scheduling.Adjuster` to use to map adjusted schedule dates into
        additional dates, which may be used, for example by fixings schedules. If given as integer
        will define the number of business days to lag dates by.
    eval_date: datetime, optional
        Only required if ``effective`` is given as a string tenor, to provide a point of reference.
    eval_mode: str in {"swaps_align", "swaptions_align"}
        The method for determining the ``effective`` and ``termination`` dates if both are provided
        as string tenors. See notes.

    Examples
    --------

    .. ipython:: python
       :suppress:

       from rateslib import Schedule, RollDay, Frequency, StubInference, Adjuster, NamedCal, dt

    .. tabs::

       .. tab:: Original Inputs

          The **original inputs** allow for a more UI friendly input for the most common schedules.

          .. ipython:: python

             s = Schedule(
                 effective=dt(2024, 1, 3),
                 termination=dt(2024, 11, 29),
                 frequency="Q",
                 stub="ShortFront",
                 modifier="MF",
                 payment_lag=2,
                 calendar="tgt",
                 eom=True,
             )
             print(s)

       .. tab:: Core Inputs

          The **core inputs** utilise the Rust objects directly and may provide more flexibility.

          .. ipython:: python

             s = Schedule(
                 effective=dt(2024, 1, 3),
                 termination=dt(2024, 11, 29),
                 frequency=Frequency.Months(3, None),
                 stub=StubInference.ShortFront,
                 modifier=Adjuster.ModifiedFollowing(),
                 payment_lag=Adjuster.BusDaysLagSettle(2),
                 calendar=NamedCal("tgt"),
                 eom=True,
             )
             print(s)

    Notes
    -----
    **Inference**

    It is not necessary to rely on inference if inputs are defined directly. However three types
    of inference will be performed otherwise:

    - **Unadjusted date inference** if any dates including stubs are given as adjusted.
    - **Frequency inference** if the ``frequency`` is missing properties, such as ``roll``.
    - **Stub date inference** if a regular schedule cannot be defined without stubs one can be
      unambiguously implied.

    *Rateslib* always tries to infer *regular* schedules ahead of *irregular* schedules. Failing
    that, it always tries to infer dates and rolls as close as possible to those given by a user.

    **Dates given as string tenor - The 1Y1Y problem**

    When generating schedules implied from tenor ``effective`` and ``termination`` dates there
    exist different theoretical ways of deriving these dates. *Rateslib* offers two practical
    methods for doing this, configurable by setting the ``eval_mode`` argument to either
    *"swaps_align"* or *"swaptions_align"*.

    .. tabs::

       .. tab:: 'swaps_align'

          This method aligns dates with those implied by a sub-component of a par tenor swap.
          E.g. a 1Y1Y schedule is expected to align with the second half of a 2Y par swap.
          To achieve this, an *unadjusted* ``effective`` date is determined from ``eval_date`` and
          an *unadjusted* ``termination`` date is derived from that ``effective`` date.

          For example, today is Tue 15th Aug '23 and spot is Thu 17th Aug '23:

          - A 1Y has effective, termination and roll of: Tue 17th Aug '23, Mon 19th Aug '24, 17.
          - A 2Y has effective, termination and roll of: Tue 17th Aug '23, Mon 18th Aug '25, 17.
          - A 1Y1Y has effective, termination and roll of: Mon 19th Aug '24, Mon 18th Aug '25, 17.

          .. ipython:: python

             s = Schedule(
                 effective="1Y",
                 termination="1Y",
                 frequency="S",
                 calendar="tgt",
                 eval_date=dt(2023, 8, 17),
                 eval_mode="swaps_align",
             )
             print(s)

       .. tab:: 'swaptions_align'

          A 1Y1Y swaption at expiry is evaluated against the 1Y swap as measured per that expiry
          date. To define this exactly requires more parameters, but this method replicates
          the true swaption expiry instrument about 95% of the time. To achieve this, an
          *adjusted* ``effective`` date is determined from the ``eval_date`` and ``modifier``, and
          an *unadjusted* ``termination`` date is derived from the ``effective`` date.

          For example, today is Tue 15th Aug '23:

          - A 1Y expiring swaption has an expiry on Thu 15th Aug '24.
          - At expiry a spot starting 1Y swap has effective, termination, and roll of:
            Mon 19th Aug '24, Tue 19th Aug '25, 19.

          .. ipython:: python

             s = Schedule(
                 effective="1Y",
                 termination="1Y",
                 frequency="S",
                 calendar="tgt",
                 eval_date=dt(2023, 8, 17),
                 eval_mode="swaptions_align",
             )
             print(s)

    """

    _obj: Schedule_rs

    @property
    def obj(self) -> Schedule_rs:
        """A wrapped instance of the Rust implemented :rust:`Schedule <scheduling>`."""
        return self._obj

    def __init__(
        self,
        effective: datetime | str,
        termination: datetime | str,
        frequency: str | Frequency,
        *,
        stub: StubInference | str_ = NoInput(0),
        front_stub: datetime_ = NoInput(0),
        back_stub: datetime_ = NoInput(0),
        roll: str | RollDay | int_ = NoInput(0),
        eom: bool_ = NoInput(0),
        modifier: Adjuster | str_ = NoInput(0),
        calendar: CalInput = NoInput(0),
        payment_lag: Adjuster | int_ = NoInput(0),
        payment_lag_exchange: Adjuster | int_ = NoInput(0),
        extra_lag: Adjuster | int_ = NoInput(0),
        eval_date: datetime_ = NoInput(0),
        eval_mode: str_ = NoInput(0),
    ) -> None:
        eom_: bool = _drb(defaults.eom, eom)
        stub_: str | StubInference = _drb(defaults.stub, stub)
        eval_mode_: str = _drb(defaults.eval_mode, eval_mode).lower()
        calendar_: CalTypes = get_calendar(calendar)
        frequency_: Frequency = _get_frequency(frequency, roll, calendar_)
        accrual_adjuster = _get_adjuster_from_modifier(modifier, _should_mod_days(termination))
        payment_adjuster = _get_adjuster_from_lag_drb(payment_lag, "payment_lag")
        payment_adjuster2 = _get_adjuster_from_lag_drb(payment_lag_exchange, "payment_lag_exchange")
        payment_adjuster3 = _get_adjuster_or_none(_drb(None, extra_lag), "payment_lag")

        effective_: datetime = _validate_effective(
            effective,
            eval_mode_,
            eval_date,
            accrual_adjuster,
            calendar_,
            roll,
        )
        termination_: datetime = _validate_termination(
            termination,
            effective_,
            accrual_adjuster,
            calendar_,
            roll,
            eom_,
        )

        self._obj = Schedule_rs(
            effective=effective_,
            termination=termination_,
            frequency=frequency_,
            calendar=calendar_,
            accrual_adjuster=accrual_adjuster,
            payment_adjuster=payment_adjuster,
            payment_adjuster2=payment_adjuster2,
            payment_adjuster3=payment_adjuster3,
            front_stub=_drb(None, front_stub),
            back_stub=_drb(None, back_stub),
            eom=eom_,
            stub_inference=_get_stub_inference(stub_, front_stub, back_stub),
        )

    @classmethod
    def __init_from_obj__(cls, obj: Schedule_rs) -> Schedule:
        """Construct the class instance from a given rust object which is wrapped."""
        # create a default instance and overwrite it
        new = cls(datetime(2000, 1, 1), datetime(2000, 2, 1), "M")
        new._obj = obj
        return new

    def __getnewargs__(
        self,
    ) -> tuple[
        datetime,
        datetime,
        Frequency,
        NoInput,
        datetime_,
        datetime_,
        NoInput,
        NoInput,
        Adjuster,
        CalInput,
        Adjuster,
        Adjuster_,
        Adjuster_,
        NoInput,
        NoInput,
    ]:
        return (
            self.ueffective,
            self.utermination,
            self.frequency_obj,
            NoInput(0),
            NoInput(0) if self.ufront_stub is None else self.ufront_stub,
            NoInput(0) if self.uback_stub is None else self.uback_stub,
            NoInput(0),
            NoInput(0),
            self.accrual_adjuster,
            self.calendar,
            self.payment_adjuster,
            NoInput(0) if self.payment_adjuster2 is None else self.payment_adjuster2,
            NoInput(0) if self.payment_adjuster3 is None else self.payment_adjuster3,
            NoInput(0),
            NoInput(0),
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self._obj == other._obj
        else:
            return False

    @cached_property
    def uschedule(self) -> list[datetime]:
        """A list of the *unadjusted* schedule dates."""
        return self.obj.uschedule

    @cached_property
    def aschedule(self) -> list[datetime]:
        """
        A list of the *adjusted accrual* dates.

        These are determined by applying the ``accrual_adjuster`` to ``uschedule``.
        """
        return self.obj.aschedule

    @cached_property
    def pschedule(self) -> list[datetime]:
        """
        A list of the cashflow *payment* dates.

        These are determined by applying the ``payment_adjuster`` to ``aschedule``.
        """
        return self.obj.pschedule

    @cached_property
    def pschedule2(self) -> list[datetime]:
        """
        A list of accrual adjusted dates.

        These are determined by applying the ``payment_adjuster2`` to ``aschedule``.
        """
        return self.obj.pschedule2

    @cached_property
    def pschedule3(self) -> list[datetime]:
        """
        A list of accrual adjusted dates.

        These are determined by applying the ``payment_adjuster3`` to ``aschedule``.
        """
        return self.obj.pschedule3

    @cached_property
    def frequency(self) -> str:
        """Original string representation of the :class:`~rateslib.scheduling.Frequency`."""
        return self.obj.frequency.string()

    @cached_property
    def periods_per_annum(self) -> float:
        """
        Average number of coupons per annum. See
        :meth:`~rateslib.scheduling.Frequency.periods_per_annum`.
        """
        return self.obj.frequency.periods_per_annum()

    @cached_property
    def frequency_obj(self) -> Frequency:
        """The :class:`~rateslib.scheduling.Frequency` object determining the periods."""
        return self.obj.frequency

    @property
    def modifier(self) -> Adjuster:
        """Alias for the ``accrual_adjuster``."""
        return self.accrual_adjuster

    @cached_property
    def calendar(self) -> CalTypes:
        """
        The calendar used for date adjustment by the ``accrual_adjuster`` and
         ``payment_adjuster``.
        """
        return self.obj.calendar

    @cached_property
    def accrual_adjuster(self) -> Adjuster:
        """The :class:`~rateslib.scheduling.Adjuster` object used for accrual date adjustment."""
        return self.obj.accrual_adjuster

    @cached_property
    def payment_adjuster(self) -> Adjuster:
        """The :class:`~rateslib.scheduling.Adjuster` object used for payment date adjustment."""
        return self.obj.payment_adjuster

    @cached_property
    def payment_adjuster2(self) -> Adjuster:
        """The :class:`~rateslib.scheduling.Adjuster` object used for additional date adjustment."""
        return self.obj.payment_adjuster2

    @cached_property
    def payment_adjuster3(self) -> Adjuster | None:
        """The :class:`~rateslib.scheduling.Adjuster` object used for additional date adjustment."""
        return self.obj.payment_adjuster3

    @cached_property
    def termination(self) -> datetime:
        """The *adjusted* termination date of the schedule."""
        return self.obj.aschedule[-1]

    @cached_property
    def effective(self) -> datetime:
        """The *adjusted* effective date of the schedule."""
        return self.obj.aschedule[0]

    @cached_property
    def utermination(self) -> datetime:
        """The *unadjusted* termination date of the schedule."""
        return self.obj.uschedule[-1]

    @cached_property
    def ueffective(self) -> datetime:
        """The *unadjusted* effective date of the schedule."""
        return self.obj.uschedule[0]

    @cached_property
    def ufront_stub(self) -> datetime | None:
        """The *unadjusted* front stub date of the schedule."""
        return self.obj.ufront_stub

    @cached_property
    def uback_stub(self) -> datetime | None:
        """The *unadjusted* back stub date of the schedule."""
        return self.obj.uback_stub

    @cached_property
    def roll(self) -> str | int | NoInput:
        """
        The :class:`~rateslib.scheduling.RollDay` object associated
        with :class:`~rateslib.scheduling.Frequency`, if available.
        """
        if isinstance(self.obj.frequency, Frequency.Months):
            # Frequency.Months on a valid Schedule will always have Some(RollDay).
            if isinstance(self.obj.frequency.roll, RollDay.Day):
                return self.obj.frequency.roll._0
            else:
                return self.obj.frequency.roll.__str__()
        else:
            return NoInput(0)

    @cached_property
    def table(self) -> DataFrame:
        """
        A `DataFrame` of schedule dates and classification.
        """
        df = DataFrame(
            {
                defaults.headers["stub_type"]: [
                    "Stub" if stub else "Regular" for stub in self._stubs
                ],
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
        """A list of boolean flags indication whether periods are stubs (True) or regular (False)"""
        front_stub = self.obj.frequency.is_stub(self.uschedule[0], self.uschedule[1], True)
        back_stub = self.obj.frequency.is_stub(self.uschedule[-2], self.uschedule[-1], False)
        if len(self.uschedule) == 2:  # single period
            return [front_stub or back_stub]
        else:
            return [front_stub] + [False] * (len(self.uschedule) - 3) + [back_stub]

    @cached_property
    def n_periods(self) -> int:
        """The number of periods contained in the schedule."""
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
        """Returns whether the schedule is composed only of regular periods (no stubs)."""
        return self.obj.is_regular()

    def to_json(self) -> str:
        """Return a JSON representation of the object.

        Returns
        -------
        str
        """
        return _make_py_json(self._obj.to_json(), "Schedule")


def _validate_effective(
    effective: datetime | str,
    eval_mode: str,
    eval_date: datetime | NoInput,
    modifier: str | Adjuster,
    calendar: CalTypes,
    roll: int | str | RollDay | NoInput,
) -> datetime:
    """
    Determine the effective date of a schedule if it is given in string form from
    other parameters such as the eval date and the eval mode.
    """
    if isinstance(effective, str):
        if isinstance(eval_date, NoInput):
            raise ValueError(
                "For `effective` given as string tenor, must also supply a base `eval_date`.",
            )
        if eval_mode == "swaps_align":
            # effective date is calculated as unadjusted
            return add_tenor(
                eval_date,
                effective,
                "NONE",
                NoInput(0),
                roll,
            )
        else:  # eval_mode == "swaptions_align":
            return add_tenor(
                eval_date,
                effective,
                modifier,
                calendar,
                roll,
            )
    else:
        return effective


def _validate_termination(
    termination: datetime | str,
    effective: datetime,
    modifier: str | Adjuster,
    calendar: CalTypes,
    roll: int | str | NoInput | RollDay,
    eom: bool,
) -> datetime:
    """
    Determine the termination date of a schedule if it is given in string form from
    """
    if isinstance(termination, str):
        if _is_day_type_tenor(termination):
            termination_: datetime = add_tenor(
                start=effective,
                tenor=termination,
                modifier=modifier,
                calendar=calendar,
                roll=NoInput(0),
                settlement=False,
                mod_days=False,
            )
        else:
            # if termination is string the end date is calculated as unadjusted, which will
            # be used later according to roll inference rules, for monthly and yearly tenors.
            if eom and isinstance(roll, NoInput) and _is_eom_cal(effective, calendar):
                roll_: str | int | NoInput | RollDay = 31
            else:
                roll_ = roll
            termination_ = add_tenor(
                effective,
                termination,
                "NONE",
                calendar,  # calendar is unused for NONE type modifier
                roll_,
            )
    else:
        termination_ = termination

    if termination_ <= effective:
        raise ValueError("Schedule `termination` must be after `effective`.")
    return termination_
