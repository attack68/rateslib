from rateslib import defaults
from rateslib.default import NoInput
from rateslib.rs import Cal, Modifier, NamedCal, RollDay, UnionCal

CalTypes = Cal | UnionCal | NamedCal
CalInput = CalTypes | str | NoInput


def _get_rollday(roll: str | int | NoInput) -> RollDay:
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
    named: bool = True,
) -> CalTypes:
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
    - *"wlg"*: Wellington for New Zealand's OCR and BKBM.

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
    if isinstance(calendar, str):
        if named:
            try:
                return NamedCal(calendar)
            except ValueError:
                # try parsing with Python only
                pass
        # parse the string in Python and return Rust Cal/UnionCal objects directly
        return _parse_str_calendar(calendar)
    elif isinstance(calendar, NoInput):
        return defaults.calendars["all"]
    else:  # calendar is a Calendar object type
        return calendar


def _parse_str_calendar(calendar: str, named: bool) -> CalTypes:
    """Parse the calendar string using Python and construct calendar objects."""
    vectors = calendar.split("|")
    if len(vectors) == 1:
        calendars = vectors[0].lower().split(",")
        if len(calendars) == 1:  # only one named calendar is found
            return defaults.calendars[calendars[0]]
        else:
            if named:
                return NamedCal(calendar)
            else:
                cals = [defaults.calendars[_] for _ in calendars]
                return UnionCal(cals, None)
    elif len(vectors) == 2:
        if named:
            return NamedCal(calendar)
        else:
            calendars = vectors[0].lower().split(",")
            cals = [defaults.calendars[_] for _ in calendars]
            settlement_calendars = vectors[1].lower().split(",")
            sets = [defaults.calendars[_] for _ in settlement_calendars]
            return UnionCal(cals, sets)
    else:
        raise ValueError("Cannot use more than one pipe ('|') operator in `calendar`.")
