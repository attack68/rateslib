from typing import Union
from rateslib.default import NoInput
from rateslib.rs import Cal, UnionCal, get_named_calendar, RollDay, Modifier

CalTypes = Union[Cal, UnionCal]
CalInput = Union[CalTypes, str, NoInput]


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


def _get_modifier(modifier: str) -> Modifier:
    try:
        return {
            "F": Modifier.F,
            "MF": Modifier.ModF,
            "P": Modifier.P,
            "MP": Modifier.ModP,
            "NONE": Modifier.Act,
        }[modifier.upper()]
    except KeyError:
        raise ValueError("`modifier` must be in {'F', 'MF', 'P', 'MP', 'NONE'}.")


CALENDARS: dict[str, CalTypes] = {
    "all": get_named_calendar("all"),
    "bus": get_named_calendar("bus"),
    "tgt": get_named_calendar("tgt"),
    "ldn": get_named_calendar("ldn"),
    "nyc": get_named_calendar("nyc"),
    "stk": get_named_calendar("stk"),
    "osl": get_named_calendar("osl"),
    "zur": get_named_calendar("zur"),
    "tro": get_named_calendar("tro"),
    "tyo": get_named_calendar("tyo"),
}


def get_calendar(calendar: CalInput, kind: bool = False) -> Union[CalTypes, tuple[CalTypes, str]]:
    """
    Returns a calendar object either from an available set or a user defined input.

    Parameters
    ----------
    calendar : str, Cal, UnionCal
        If `str`, then the calendar is returned from pre-calculated values.
        If a specific user defined calendar this is returned without modification.
    kind : bool
        If `True` will also return the kind of calculation from `"null", "named",
        "custom"`.

    Returns
    -------
    Cal, UnionCal or tuple

    Notes
    -----

    The following named calendars are available and have been back tested against the
    publication of RFR indexes in the relevant geography.

    - *"all"*: Every day is defined as business day including weekends.
    - *"bus"*: Regular weekdays are defined as business days. Saturdays and Sunday are non-business days.
    - *"tgt"*: Target for Europe's ESTR.
    - *"osl"*: Oslo for Norway's NOWA.
    - *"zur"*: Zurich for Switzerland's SARON.
    - *"nyc"*: New York City for US's SOFR.
    - *"ldn"*: London for UK's SONIA.
    - *"stk"*: Stockholm for Sweden's SWESTR.
    - *"tro"*: Toronto for Canada's CORRA.
    - *"tyo"*: Tokyo for Japan's TONA.

    Examples
    --------
    .. ipython:: python
       :suppress:

       from rateslib import get_calendar

    .. ipython:: python

       tgt_cal = get_calendar("tgt")
       tgt_cal.holidays[300:312]
       tgt_cal.add_bus_days(dt(2023, 1, 3), 5, True)
       type(tgt_cal)

    Calendars can be combined from the pre-existing names using comma separation.

    .. ipython:: python

       tgt_and_nyc_cal = get_calendar("tgt,nyc")
       tgt_and_nyc_cal.holidays[300:312]

    """
    # TODO: rename calendars or make a more generalist statement about their names.
    if calendar is NoInput.blank:
        ret = CALENDARS["all"], "null"
    elif isinstance(calendar, str):
        calendars = calendar.lower().split(",")
        if len(calendars) == 1:  # only one named calendar is found
            ret = CALENDARS[calendars[0]], "named"
        else:
            cals = [CALENDARS[_] for _ in calendars]
            ret = UnionCal(cals, None), "named"
    else:  # calendar is a Calendar object type
        ret = calendar, "custom"

    return ret if kind else ret[0]
