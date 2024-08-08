from typing import Union

from rateslib import defaults
from rateslib.default import NoInput
from rateslib.rs import Cal, Modifier, NamedCal, RollDay, UnionCal

CalTypes = Union[Cal, UnionCal, NamedCal]
CalInput = Union[CalTypes, str, NoInput]

Modifier.__doc__ = "Enumerable type for modification rules."
RollDay.__doc__ = "Enumerable type for roll day types."


def _get_rollday(roll: Union[str, int, NoInput]) -> RollDay:
    if isinstance(roll, str):
        return {
            "EOM": RollDay.EoM(),
            "SOM": RollDay.SoM(),
            "IMM": RollDay.IMM(),
        }[roll.upper()]
    elif isinstance(roll, int):
        return RollDay.Int(roll)
    return RollDay.Unspecified()


_M_ALL = {
    "F": Modifier.F,
    "MF": Modifier.ModF,
    "P": Modifier.P,
    "MP": Modifier.ModP,
    "NONE": Modifier.Act,
}
_M_EXCL_DAYS = {
    "F": Modifier.F,
    "MF": Modifier.F,
    "P": Modifier.P,
    "MP": Modifier.P,
    "NONE": Modifier.Act,
}


def _get_modifier(modifier: str, mod_days: bool) -> Modifier:
    if mod_days:
        m = _M_ALL
    else:
        m = _M_EXCL_DAYS

    try:
        return m[modifier.upper()]
    except KeyError:
        raise ValueError("`modifier` must be in {'F', 'MF', 'P', 'MP', 'NONE'}.")


def get_calendar(
    calendar: CalInput,
    kind: bool = False,
    named: bool = True,
) -> Union[CalTypes, tuple[CalTypes, str]]:
    """
    Returns a calendar object either from an available set or a user defined input.

    Parameters
    ----------
    calendar : str, Cal, UnionCal, NamedCal
        If `str`, then the calendar is returned from pre-calculated values.
        If a specific user defined calendar this is returned without modification.
    kind : bool
        If `True` will also return the kind of calculation from `"null", "named",
        "custom"`.
    named : bool
        If `True` will return a :class:`~rateslib.calendars.NamedCal` object, which is more
        compactly serialized, otherwise will parse an input string and return a
        :class:`~rateslib.calendars.Cal` or :class:`~rateslib.calendars.UnionCal` directly.

    Returns
    -------
    NamedCal, Cal, UnionCal or tuple

    Notes
    -----

    The following named calendars are available and have been back tested against the
    publication of RFR indexes in the relevant geography.

    - *"all"*: Every day is defined as business day including weekends.
    - *"bus"*: Regular weekdays are defined as business days. Saturdays and Sunday are
      non-business days.
    - *"tgt"*: Target for Europe's ESTR.
    - *"osl"*: Oslo for Norway's NOWA.
    - *"zur"*: Zurich for Switzerland's SARON.
    - *"nyc"*: New York City for US's SOFR.
    - *"fed"*: Similar to *"nyc"* but omitting Good Friday.
    - *"ldn"*: London for UK's SONIA.
    - *"stk"*: Stockholm for Sweden's SWESTR.
    - *"tro"*: Toronto for Canada's CORRA.
    - *"tyo"*: Tokyo for Japan's TONA.
    - *"syd"*: Sydney for Australia's AONIA.

    Combined calendars can be created with comma separated input, e.g. *"tgt,nyc"*. This would
    be the typical calendar assigned to a cross-currency derivative such as a EUR/USD
    cross-currency swap.

    For short-dated, FX instrument date calculations a concept known as an
    **associated settlement calendars** is introduced. This uses a secondary calendar to determine
    if a calculated date is a valid settlement day, but it is not used in the determination
    of tenor dates. For a EURUSD FX instrument the appropriate calendar combination is *"tgt|nyc"*.
    For a GBPEUR FX instrument the appropriate calendar combination is *"ldn,tgt|nyc"*.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib import get_calendar, dt

    .. ipython:: python

       tgt_cal = get_calendar("tgt", named=False)
       tgt_cal.holidays[300:312]
       tgt_cal.add_bus_days(dt(2023, 1, 3), 5, True)
       type(tgt_cal)

    Calendars can be combined from the pre-existing names using comma separation.

    .. ipython:: python

       tgt_and_nyc_cal = get_calendar("tgt,nyc")
       tgt_and_nyc_cal.holidays[300:312]

    """
    # TODO: rename calendars or make a more generalist statement about their names.
    if isinstance(calendar, str) and named:
        try:
            return _get_calendar_labelled(NamedCal(calendar), "object", kind)
        except ValueError:
            named = False  # try parsing with Python only

    if calendar is NoInput.blank:
        return _get_calendar_labelled(defaults.calendars["all"], "null", kind)
    elif isinstance(calendar, str) and not named:
        # parse the string in Python and return Rust objects directly
        vectors = calendar.split("|")
        if len(vectors) == 1:
            calendars = vectors[0].lower().split(",")
            if len(calendars) == 1:  # only one named calendar is found
                return _get_calendar_labelled(defaults.calendars[calendars[0]], "named", kind)
            else:
                cals = [defaults.calendars[_] for _ in calendars]
                return _get_calendar_labelled(UnionCal(cals, None), "named", kind)
        elif len(vectors) == 2:
            calendars = vectors[0].lower().split(",")
            cals = [defaults.calendars[_] for _ in calendars]
            settlement_calendars = vectors[1].lower().split(",")
            sets = [defaults.calendars[_] for _ in settlement_calendars]
            return _get_calendar_labelled(UnionCal(cals, sets), "named", kind)
        else:
            raise ValueError("Cannot use more than one pipe ('|') operator in `calendar`.")
    else:  # calendar is a Calendar object type
        return _get_calendar_labelled(calendar, "custom", kind)


def _get_calendar_labelled(output, label, kind):
    """Package the return for the get_calendar function"""
    if kind:
        return output, label
    return output
