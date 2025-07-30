from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from rateslib.default import NoInput
from rateslib.rs import Adjuster, Frequency, Imm, RollDay
from rateslib.scheduling.adjuster import _convert_to_adjuster
from rateslib.scheduling.calendars import get_calendar
from rateslib.scheduling.rollday import _get_rollday

if TYPE_CHECKING:
    from rateslib.typing import CalInput, datetime_, int_


def add_tenor(
    start: datetime,
    tenor: str | Frequency,
    modifier: str | Adjuster,
    calendar: CalInput = NoInput(0),
    roll: str | int_ | RollDay = NoInput(0),
    settlement: bool = False,
    mod_days: bool = False,
) -> datetime:
    r"""
    Add a tenor to a given date under specific modification rules and holiday calendar.

    .. warning::

       Note this function does not validate the ``roll`` input, but expects it to be correct.
       That is this function acts on ``start`` as an *unchecked* date. See notes.

    Parameters
    ----------
    start : datetime
        The date to which to add the tenor.
    tenor : str | Frequency
        The tenor to add, identified by calendar days, `"D"`, months, `"M"`,
        years, `"Y"` or business days, `"B"`, for example `"10Y"` or `"5B"`.
    modifier : str, optional in {"NONE", "MF", "F", "MP", "P"} | Adjuster
        The modification rule to apply if the tenor is calendar days, months or years.
    calendar : CustomBusinessDay or str, optional
        The calendar for use with business day adjustment and modification.
    roll : str, int, RollDay, optional
        This is only required if the tenor is given in months or years. Ensures the tenor period
        associates with a schedule's roll day.
    settlement : bool, optional
        If ``modifier`` is string this determines whether to enforce the settlement
        with an associated settlement calendar, if provided.
    mod_days : bool, optional
        If ``modifier`` is string and ``tenor`` is a day variant setting this to *False*
        will convert "MF" to "F" and "MP" to "P".

    Returns
    -------
    datetime

    Notes
    ------

    .. ipython:: python
       :suppress:

       from rateslib import add_tenor, get_calendar

       from datetime import datetime as dt
       import pandas as pd
       from pandas import date_range, Series, DataFrame
       pd.set_option("display.float_format", lambda x: '%.2f' % x)
       pd.set_option("display.max_columns", None)
       pd.set_option("display.width", 500)

    This method is a convenience function for coordinating a :class:`~rateslib.scheduling.Frequency`
    date manipulation and an :class:`~rateslib.scheduling.Adjuster`, from simple UI inputs.
    For example the following:

    .. ipython:: python

       add_tenor(dt(2023, 9, 29), "-6m", "MF", NamedCal("bus"), 31)

    is internally mapped to the following, where :meth:`~rateslib.scheduling.Frequency.next`
    performs an *unchecked* date period determination:

    .. ipython:: python

       unadjusted_date = Frequency.Months(-6, RollDay.Day(31)).next(dt(2023, 9, 29))
       Adjuster.ModifiedFollowing().adjust(unadjusted_date, NamedCal("bus"))

    The allowed string inputs *{'B', 'D', 'W', 'M', 'Y'}* for **b**\ usiness days, calendar
    **d**\ ays, **w**\ eeks, **m**\ onths and **y**\ ears are mapped to an appropriate
    :class:`~rateslib.scheduling.Frequency` variant (potentially also mapping a
    :class:`~rateslib.scheduling.RollDay`), and an appropriate
    :class:`~rateslib.scheduling.Adjuster` is derived from the combination of the ``modifier``,
    ``settlement`` and ``mod_days`` arguments.

    Read more about the ``settlement`` argument in the :ref:`calendar user guide <cal-doc>`.

    The ``mod_days`` argument is provided to avoid having to reconfigure *Instrument*
    specifications when a *termination* may differ between months or years, and days or weeks.
    Most *Instruments* will be defined by the typical modified following (*"MF"*) ``modifier``,
    but this would prefer not to always apply.

    .. ipython:: python

       add_tenor(dt(2021, 1, 29), "1M", "MF", "bus")

    while, the following will by default roll into a new month,

    .. ipython:: python

       add_tenor(dt(2021, 1, 22), "8d", "MF", "bus")

    unless day type frequencies are also specifically modified,

    .. ipython:: python

       add_tenor(dt(2021, 1, 22), "8d", "MF", "bus", mod_days=True)

    Examples
    --------

    .. ipython:: python

       add_tenor(dt(2022, 2, 28), "3M", "NONE")
       add_tenor(dt(2022, 12, 28), "4b", "F", get_calendar("ldn"))
       add_tenor(dt(2022, 12, 28), "4d", "F", get_calendar("ldn"))
    """  # noqa: E501
    cal_ = get_calendar(calendar)
    if isinstance(tenor, Frequency):
        frequency: Frequency = tenor
    else:
        tenor = tenor.upper()
        if "D" in tenor:
            frequency = Frequency.CalDays(int(tenor[:-1]))
        elif "W" in tenor:
            frequency = Frequency.CalDays(int(tenor[:-1]) * 7)
        elif "B" in tenor:
            frequency = Frequency.BusDays(int(tenor[:-1]), cal_)
        elif "Y" in tenor:
            roll_ = _get_rollday(roll)
            roll__ = RollDay.Day(start.day) if roll_ is None else roll_
            frequency = Frequency.Months(int(float(tenor[:-1]) * 12), roll__)
        elif "M" in tenor:
            roll_ = _get_rollday(roll)
            roll__ = RollDay.Day(start.day) if roll_ is None else roll_
            frequency = Frequency.Months(int(float(tenor[:-1])), roll__)
        else:
            raise ValueError(
                "`tenor` must identify frequency in {'B', 'D', 'W', 'M', 'Y'} e.g. '1Y'"
            )

    if isinstance(frequency, Frequency.Months | Frequency.Zero):
        mod_days = True

    next_date = frequency.next(start)
    adjuster = _convert_to_adjuster(modifier, settlement, mod_days)
    return adjuster.adjust(next_date, cal_)


def _get_fx_expiry_and_delivery(
    eval_date: datetime_,
    expiry: str | datetime,
    delivery_lag: int | datetime,
    calendar: CalInput,
    modifier: str,
    eom: bool,
) -> tuple[datetime, datetime]:
    """
    Determines the expiry and delivery date of an FX option using the following rules:

    See Foreign Exchange Option Pricing by Iain Clark

    Parameters
    ----------
    eval_date: datetime
        The evaluation date, which is today (if required)
    expiry: str, datetime
        The expiry date
    delivery_lag: int, datetime
        Number of days, e.g. spot = 2, or a specified datetime for FX settlement after expiry.
    calendar: CalInput
        The calendar used for date rolling. This function makes use of the `settlement` option
        within calendars.
    modifier: str
        Date rule, expected to be "MF" for most FX rate tenors.
    eom: bool
        Whether end-of-month is preserved in tenor date determination.

    Returns
    -------
    tuple of datetime
    """
    if isinstance(expiry, str):
        if isinstance(eval_date, NoInput):
            raise ValueError("`expiry` as string tenor requires `eval_date`.")
        # then the expiry will be implied
        e = expiry.upper()
        if "M" in e or "Y" in e:
            # method
            if isinstance(delivery_lag, datetime):
                raise ValueError(
                    "Cannot determine FXOption expiry and delivery with given parameters.\n"
                    "Supply a `delivery_lag` as integer business days and not a datetime, when "
                    "using a string tenor `expiry`.",
                )
            else:
                spot = get_calendar(calendar).lag_bus_days(eval_date, delivery_lag, True)
                roll = "eom" if (eom and Imm.Eom.validate(spot)) else spot.day
                delivery_: datetime = add_tenor(spot, expiry, modifier, calendar, roll, True)
                expiry_ = get_calendar(calendar).add_bus_days(delivery_, -delivery_lag, False)
                return expiry_, delivery_
        else:
            expiry_ = add_tenor(eval_date, expiry, "F", calendar, NoInput(0), False)
    else:
        expiry_ = expiry

    if isinstance(delivery_lag, datetime):
        delivery_ = delivery_lag
    else:
        delivery_ = get_calendar(calendar).lag_bus_days(expiry_, delivery_lag, True)

    return expiry_, delivery_
